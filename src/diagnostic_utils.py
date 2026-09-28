"""Shared utilities for training-time diagnostics.

This module intentionally groups the small diagnostic helpers together:
logging, numeric conversions, scalar matching summaries, and HVP operators.
The full local-quadratic diagnostic lives in its own module.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.func import functional_call, jvp, vjp

from src.model import GPT
from src.muon_optimizer import zeropower_via_newtonschulz5


def _default_print0(s, console=False, log=True):
    if console:
        print(s)


_logger = _default_print0


def set_diagnostic_logger(logger):
    global _logger
    _logger = logger


def print0(s, console=False, log=True):
    return _logger(s, console=console, log=log)


def _ns5_and_shape_scale(buf: Tensor, ns_steps: int) -> tuple[Tensor, float]:
    """Returns (vi_raw, shape_scale) where vi_raw is the Muon update to subtract."""
    buf_2d = buf.view(buf.shape[0], -1).float()
    vi_2d = zeropower_via_newtonschulz5(buf_2d, steps=ns_steps).float()
    shape_scale = max(1.0, vi_2d.size(-2) / vi_2d.size(-1)) ** 0.5
    vi_raw = vi_2d * shape_scale
    return vi_raw.reshape(buf.shape), shape_scale


def _safe_float(x) -> float:
    v = float(x)
    return v if math.isfinite(v) else None


def _safe_float_or_none(x) -> float | None:
    if x is None:
        return None
    return _safe_float(x)


def _tensor_to_matrix(t: Tensor) -> Tensor:
    if t.ndim < 2:
        raise ValueError(f"expected matrix-like tensor, got shape={tuple(t.shape)}")
    if t.ndim == 2:
        return t
    return t.reshape(t.shape[0], -1)


def _pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    den = (vx * vy) ** 0.5
    if den <= 1e-30:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def _vector_match_stats(xs: list[float], ts: list[float]) -> dict:
    finite_pairs = [
        (float(x), float(t))
        for x, t in zip(xs, ts)
        if math.isfinite(float(x)) and math.isfinite(float(t))
    ]
    if not finite_pairs:
        return {
            "n": 0,
            "cosine": None,
            "pearson": None,
            "best_scale_to_t": None,
            "relative_l2_after_best_scale": None,
            "direct_relative_l2": None,
            "mean_x": None,
            "mean_t": None,
            "std_x": None,
            "std_t": None,
        }

    x_vals = [p[0] for p in finite_pairs]
    t_vals = [p[1] for p in finite_pairs]
    dot_xt = sum(x * t for x, t in finite_pairs)
    dot_xx = sum(x * x for x in x_vals)
    dot_tt = sum(t * t for t in t_vals)
    norm_prod = math.sqrt(max(dot_xx, 0.0) * max(dot_tt, 0.0))
    cosine = dot_xt / norm_prod if norm_prod > 1e-30 else None
    pearson = _pearson_corr(x_vals, t_vals)
    best_scale = dot_xt / dot_xx if dot_xx > 1e-30 else None
    if best_scale is not None and dot_tt > 1e-30:
        rel_l2 = math.sqrt(sum((best_scale * x - t) ** 2 for x, t in finite_pairs) / dot_tt)
    else:
        rel_l2 = None
    direct_rel_l2 = (
        math.sqrt(sum((x - t) ** 2 for x, t in finite_pairs) / dot_tt)
        if dot_tt > 1e-30 else None
    )
    mean_x = sum(x_vals) / len(x_vals)
    mean_t = sum(t_vals) / len(t_vals)
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_vals) / len(x_vals))
    std_t = math.sqrt(sum((t - mean_t) ** 2 for t in t_vals) / len(t_vals))
    return {
        "n": len(finite_pairs),
        "cosine": _safe_float_or_none(cosine),
        "pearson": _safe_float_or_none(pearson),
        "best_scale_to_t": _safe_float_or_none(best_scale),
        "relative_l2_after_best_scale": _safe_float_or_none(rel_l2),
        "direct_relative_l2": _safe_float_or_none(direct_rel_l2),
        "mean_x": _safe_float(mean_x),
        "mean_t": _safe_float(mean_t),
        "std_x": _safe_float(std_x),
        "std_t": _safe_float(std_t),
    }




def _make_block_ggn_operator(
    *,
    model: GPT,
    diag_inputs: Tensor,
    param_name: str,
    num_tokens_diag: int,
    ggn_microbatch_size: int,
):
    """Return H_GGN[delta] for one parameter block using chunked JVP/VJP on logits."""
    params = {name: p.detach().float() for name, p in model.named_parameters()}
    buffers = {
        name: b.detach().float() if b.is_floating_point() else b.detach()
        for name, b in model.named_buffers()
    }
    base_param = params[param_name]
    inv_tokens = 1.0 / float(max(num_tokens_diag, 1))
    chunk_size = max(1, int(ggn_microbatch_size))

    def operator(delta: Tensor) -> Tensor:
        delta = delta.to(device=base_param.device, dtype=base_param.dtype)
        ggn_vec = torch.zeros_like(base_param, dtype=torch.float32)
        for start in range(0, diag_inputs.size(0), chunk_size):
            inputs_chunk = diag_inputs[start:start + chunk_size]

            def logits_from_param(param_value: Tensor) -> Tensor:
                local_params = dict(params)
                local_params[param_name] = param_value
                if inputs_chunk.is_cuda:
                    sdp_ctx = sdpa_kernel(SDPBackend.MATH)
                else:
                    sdp_ctx = nullcontext()
                with sdp_ctx:
                    logits = functional_call(
                        model,
                        (local_params, buffers),
                        args=(inputs_chunk, None),
                        tie_weights=True,
                        strict=False,
                    )
                return logits.float().reshape(-1, logits.shape[-1])

            base_logits_flat, vjp_fn = vjp(logits_from_param, base_param)
            probs = torch.softmax(base_logits_flat.detach().float(), dim=-1)
            _, logits_tangent = jvp(logits_from_param, (base_param,), (delta,))
            tangent = logits_tangent.float().reshape_as(probs)
            centered = tangent - torch.sum(probs * tangent, dim=-1, keepdim=True)
            hz_jv = probs * centered * inv_tokens
            (chunk_vec,) = vjp_fn(hz_jv.to(dtype=base_logits_flat.dtype))
            ggn_vec.add_(chunk_vec.detach().float())
        if dist.is_initialized():
            dist.all_reduce(ggn_vec, op=dist.ReduceOp.SUM)
        return ggn_vec

    return operator


def _exact_block_hessian_vec(
    *,
    model: GPT,
    diag_inputs: Tensor,
    diag_targets: Tensor,
    param: Tensor,
    vector: Tensor,
    num_tokens_diag: int,
    hvp_microbatch_size: int,
) -> Tensor:
    """Return H_ii @ vector for the diagnostic per-token mean loss."""
    hv = torch.zeros_like(param, dtype=torch.float32)
    v = vector.detach().to(device=param.device, dtype=param.dtype)
    chunk_size = max(1, int(hvp_microbatch_size))

    for start in range(0, diag_inputs.size(0), chunk_size):
        inputs_chunk = diag_inputs[start:start + chunk_size]
        targets_chunk = diag_targets[start:start + chunk_size]
        sdp_ctx = sdpa_kernel(SDPBackend.MATH) if inputs_chunk.is_cuda else nullcontext()
        with torch.enable_grad(), sdp_ctx:
            loss_sum = model(inputs_chunk, targets_chunk)
            loss_mean_global = loss_sum / float(max(num_tokens_diag, 1))
            (g,) = torch.autograd.grad(
                loss_mean_global,
                param,
                create_graph=True,
                allow_unused=True,
            )
            if g is None:
                continue
            gv = torch.sum(g * v)
            if not gv.requires_grad:
                continue
            (hv_chunk,) = torch.autograd.grad(gv, param, allow_unused=True)
        if hv_chunk is not None:
            hv.add_(hv_chunk.detach().float())

    if dist.is_initialized():
        dist.all_reduce(hv, op=dist.ReduceOp.SUM)
    return hv


def _exact_source_block_hessian_vecs(
    *,
    model: GPT,
    diag_inputs: Tensor,
    diag_targets: Tensor,
    source_param: Tensor,
    source_vector: Tensor,
    target_params: list[Tensor],
    num_tokens_diag: int,
    hvp_microbatch_size: int,
) -> dict[int, Tensor]:
    """Return {id(target): H_target,source @ source_vector} for the diagnostic loss."""
    hv_by_pid = {
        id(p): torch.zeros_like(p, dtype=torch.float32)
        for p in target_params
    }
    v = source_vector.detach().to(device=source_param.device, dtype=source_param.dtype)
    chunk_size = max(1, int(hvp_microbatch_size))

    for start in range(0, diag_inputs.size(0), chunk_size):
        inputs_chunk = diag_inputs[start:start + chunk_size]
        targets_chunk = diag_targets[start:start + chunk_size]
        sdp_ctx = sdpa_kernel(SDPBackend.MATH) if inputs_chunk.is_cuda else nullcontext()
        with torch.enable_grad(), sdp_ctx:
            loss_sum = model(inputs_chunk, targets_chunk)
            loss_mean_global = loss_sum / float(max(num_tokens_diag, 1))
            (g_source,) = torch.autograd.grad(
                loss_mean_global,
                source_param,
                create_graph=True,
                allow_unused=True,
            )
            if g_source is None:
                continue
            gv = torch.sum(g_source * v)
            if not gv.requires_grad:
                continue
            hv_chunks = torch.autograd.grad(gv, target_params, allow_unused=True)
        for target_param, hv_chunk in zip(target_params, hv_chunks):
            if hv_chunk is not None:
                hv_by_pid[id(target_param)].add_(hv_chunk.detach().float())

    if dist.is_initialized():
        for hv in hv_by_pid.values():
            dist.all_reduce(hv, op=dist.ReduceOp.SUM)
    return hv_by_pid


def _estimate_block_lmax(
    *,
    ggn_operator,
    shape: torch.Size,
    device: torch.device,
    power_iters: int,
    seed: int,
    floor: float,
    initial_vector: Tensor | None = None,
    eps: float = 1e-12,
) -> dict:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    if initial_vector is not None and tuple(initial_vector.shape) == tuple(shape):
        u = initial_vector.to(device=device, dtype=torch.float32, non_blocking=True)
        vector_source = "warm_start"
    else:
        u = torch.randn(shape, device=device, dtype=torch.float32, generator=gen)
        vector_source = "random_init"
    u = u / torch.linalg.vector_norm(u).clamp_min(eps)

    last_norm = None
    for _ in range(max(1, int(power_iters))):
        z = ggn_operator(u)
        z_norm = torch.linalg.vector_norm(z)
        last_norm = float(z_norm)
        if not math.isfinite(last_norm) or last_norm <= eps:
            u = torch.randn(shape, device=device, dtype=torch.float32, generator=gen)
            u = u / torch.linalg.vector_norm(u).clamp_min(eps)
        else:
            u = (z / z_norm).detach()

    h_u = ggn_operator(u)
    raw_L = float(torch.sum(u * h_u).item())
    floor_applied = (not math.isfinite(raw_L)) or raw_L <= floor
    L = max(raw_L if math.isfinite(raw_L) else floor, floor)
    return {
        "L_i": L,
        "L_i_raw": raw_L if math.isfinite(raw_L) else None,
        "L_i_floor_applied": bool(floor_applied),
        "power_last_hvp_norm": last_norm,
        "power_vector_cpu": u.detach().cpu(),
        "power_vector_source": vector_source,
    }



safe_float = _safe_float
safe_float_or_none = _safe_float_or_none
tensor_to_matrix = _tensor_to_matrix
