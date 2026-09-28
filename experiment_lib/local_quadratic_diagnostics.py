"""Exact-HVP local quadratic diagnostic measured during training."""

from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.distributed as dist

from experiment_lib.diagnostic_utils import print0
from experiment_lib.diagnostic_utils import (
    _ns5_and_shape_scale,
    _safe_float,
    _safe_float_or_none,
    _tensor_to_matrix,
    _vector_match_stats,
)
from experiment_lib.diagnostic_utils import (
    _exact_block_hessian_vec,
    _exact_source_block_hessian_vecs,
)
from experiment_lib.model import GPT
from experiment_lib.muon_optimizer import DistributedMuonLKindwise


def compute_local_quadratic_diagnostics(
    model: GPT,
    optimizer2: DistributedMuonLKindwise,
    diag_inputs: Tensor,
    diag_targets: Tensor,
    diag_data_source: str,
    train_batch_tokens: int,
    diag_grad_microbatch_size: int,
    hvp_microbatch_size: int,
    lr_current: float,
    diag_index: int,
    step: int,
    train_loss: float,
    scheme: str,
    seed: int,
    ns_steps: int,
    max_blocks: int,
    curvature_method: str,
    fd_probes: int,
    fd_rel_eps: float,
    fd_abs_eps: float,
    eps_w: float,
    eps_den: float,
    exact_hvp_eager_diagnostic_model: bool,
    measure_cross_curvature: bool,
    save_cross_curvature_matrix: bool,
) -> dict:
    """
    Measure how well three scalar choices x_i match

        t_i = <G_i, V_i> / (||W_i||_F * <V_i, H_ii V_i>)

    for the basic Muon shape-scaled update

        -eta * r_i * ||W_i||_F * V_i.

    Called every DIAG_FREQ steps, before optimizer.step().
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        cuda_allocated_start = int(torch.cuda.memory_allocated())
        cuda_reserved_start = int(torch.cuda.memory_reserved())
    else:
        cuda_allocated_start = None
        cuda_reserved_start = None

    world_size = dist.get_world_size()
    num_tokens_diag = diag_inputs.numel() * world_size
    momentum_debug_scale = 1.0 / train_batch_tokens

    block_params = []
    group_cfg_by_pid = {}
    for group in optimizer2.param_groups:
        base_lr = float(group.get("initial_lr", group["lr"]))
        raw_wd = float(group["weight_decay"])
        beta = float(group["momentum"])
        nesterov = bool(group["nesterov"])
        for p in group["params"]:
            if p.ndim < 2:
                continue
            state = optimizer2.state[p]
            if "momentum_buffer" not in state:
                continue
            name = optimizer2._param_name_by_id.get(id(p), f"id_{id(p)}")
            kind = optimizer2._matrix_kind(p)
            block_idx = optimizer2._param_block_by_id.get(id(p), -1)
            block_params.append((p, name, kind, block_idx, base_lr, raw_wd))
            group_cfg_by_pid[id(p)] = {"momentum": beta, "nesterov": nesterov}
    block_params.sort(key=lambda item: (item[3], item[1]))
    if max_blocks > 0:
        block_params = block_params[:max_blocks]

    def compute_diag_gradient() -> tuple[float, float, dict[int, Tensor]]:
        saved_grads = [(p, None if p.grad is None else p.grad.detach().clone())
                       for p in model.parameters()]
        loss_sum_global = torch.zeros((), device=diag_inputs.device, dtype=torch.float32)
        chunk_size = max(1, int(diag_grad_microbatch_size))
        g_diag_accum: dict[int, Tensor] = {}

        for start in range(0, diag_inputs.size(0), chunk_size):
            model.zero_grad(set_to_none=True)
            inputs_chunk = diag_inputs[start:start + chunk_size]
            targets_chunk = diag_targets[start:start + chunk_size]
            loss_sum_local = model(inputs_chunk, targets_chunk)
            loss_sum_global.add_(loss_sum_local.detach().float())
            chunk_tokens_global = inputs_chunk.numel() * world_size
            chunk_weight = float(chunk_tokens_global) / float(max(num_tokens_diag, 1))
            (loss_sum_local / float(max(chunk_tokens_global, 1))).backward()

            for p, *_ in block_params:
                if p.grad is None:
                    raise RuntimeError(f"missing diagnostic gradient for {optimizer2._param_name_by_id.get(id(p), id(p))}")
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                g_chunk = p.grad.detach().float().clone()
                pid = id(p)
                if pid not in g_diag_accum:
                    g_diag_accum[pid] = g_chunk.mul(chunk_weight)
                else:
                    g_diag_accum[pid].add_(g_chunk, alpha=chunk_weight)

        dist.all_reduce(loss_sum_global, op=dist.ReduceOp.SUM)

        g_diag: dict[int, Tensor] = {}
        for p, *_ in block_params:
            g_diag[id(p)] = g_diag_accum[id(p)]

        model.zero_grad(set_to_none=True)
        for p, g in saved_grads:
            p.grad = g
        f_sum = float(loss_sum_global)
        return f_sum, f_sum / num_tokens_diag, g_diag

    def compute_diag_loss_only() -> tuple[float, float]:
        loss_sum_global = torch.zeros((), device=diag_inputs.device, dtype=torch.float32)
        chunk_size = max(1, int(hvp_microbatch_size))
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for start in range(0, diag_inputs.size(0), chunk_size):
                inputs_chunk = diag_inputs[start:start + chunk_size]
                targets_chunk = diag_targets[start:start + chunk_size]
                loss_sum_local = model(inputs_chunk, targets_chunk)
                loss_sum_global.add_(loss_sum_local.detach().float())
        if was_training:
            model.train()
        dist.all_reduce(loss_sum_global, op=dist.ReduceOp.SUM)
        f_sum = float(loss_sum_global)
        return f_sum, f_sum / num_tokens_diag

    f_0_sum, f_0, g_diag = compute_diag_gradient()

    block_info: dict[int, dict] = {}
    by_kind: dict[str, list[int]] = {}
    for block_ord, (p, name, kind, block_idx, _lr, _wd) in enumerate(block_params):
        state = optimizer2.state[p]
        if p.grad is None:
            raise RuntimeError(f"missing current training gradient for {name}")

        beta = group_cfg_by_pid[id(p)]["momentum"]
        nesterov = group_cfg_by_pid[id(p)]["nesterov"]
        train_grad = p.grad.detach().float()
        buf_old = state["momentum_buffer"].detach().float()
        buf_next = torch.lerp(buf_old, train_grad, 1.0 - beta)
        update_buffer = torch.lerp(train_grad, buf_next, beta) if nesterov else buf_next
        vi_raw, shape_scale = _ns5_and_shape_scale(update_buffer, ns_steps)
        vi_raw = vi_raw.float()
        vi_raw_norm = float(torch.linalg.vector_norm(vi_raw).item())
        if vi_raw_norm <= eps_den or not math.isfinite(vi_raw_norm):
            v_i = torch.zeros_like(vi_raw, dtype=torch.float32)
        else:
            v_i = vi_raw / vi_raw_norm

        w = p.detach().float()
        w_norm = float(torch.linalg.vector_norm(w).item())
        m_i, n_i = int(_tensor_to_matrix(w).shape[0]), int(_tensor_to_matrix(w).shape[1])
        d_i = min(m_i, n_i)
        r_i = vi_raw_norm / max(w_norm + eps_w, 1e-30)

        g_i = g_diag[id(p)].float()
        g_dot_v = float(torch.sum(g_i * v_i).item())
        fd_epsilon = max(float(fd_abs_eps), float(fd_rel_eps) * max(w_norm, 1.0))

        block_info[id(p)] = {
            "param_name": name,
            "type": kind,
            "block_idx": block_idx,
            "m_i": m_i,
            "n_i": n_i,
            "d_i": d_i,
            "w_norm": w_norm,
            "r_i": r_i,
            "shape_scale": shape_scale,
            "vi_raw_norm": vi_raw_norm,
            "vi_unit_norm": float(torch.linalg.vector_norm(v_i).item()),
            "train_grad_frob_norm": float(torch.linalg.vector_norm(train_grad).item()),
            "diag_g_frob_norm": float(torch.linalg.vector_norm(g_i).item()),
            "g_dot_v": g_dot_v,
            "v_i_tensor": v_i,
            "fd_epsilon": fd_epsilon,
            "base_lr": _lr,
            "weight_decay_raw": _wd,
        }
        by_kind.setdefault(kind, []).append(id(p))

    fd_probe_losses = []

    cross_curvature_matrix = None
    cross_curvature_order = None
    if measure_cross_curvature and curvature_method != "exact_hvp":
        raise ValueError("cross curvature measurement currently requires MYOPT_LOCAL_QUADRATIC_CURVATURE=exact_hvp")

    if curvature_method == "exact_hvp" and measure_cross_curvature:
        n_blocks = len(block_params)
        target_params = [p for p, *_ in block_params]
        cross_curvature_matrix = [
            [float("nan") for _ in range(n_blocks)]
            for _ in range(n_blocks)
        ]
        cross_curvature_order = [
            {
                "block_name": name,
                "type": kind,
                "block_idx": int(block_idx),
            }
            for p, name, kind, block_idx, _lr, _wd in block_params
        ]
        for source_ord, (source_p, source_name, _source_kind, _source_block_idx, _lr, _wd) in enumerate(block_params):
            source_info = block_info[id(source_p)]
            source_v = source_info["v_i_tensor"]
            if dist.get_rank() == 0 and n_blocks > 8 and source_ord % 12 == 0:
                print0(
                    f"local_quadratic_diag step:{step} cross_hvp_source:{source_ord + 1}/{n_blocks} {source_name}",
                    console=True,
                    log=False,
                )
            hv_by_pid = _exact_source_block_hessian_vecs(
                model=model,
                diag_inputs=diag_inputs,
                diag_targets=diag_targets,
                source_param=source_p,
                source_vector=source_v,
                target_params=target_params,
                num_tokens_diag=num_tokens_diag,
                hvp_microbatch_size=hvp_microbatch_size,
            )
            for target_ord, (target_p, *_target_rest) in enumerate(block_params):
                target_info = block_info[id(target_p)]
                hv_target = hv_by_pid[id(target_p)]
                cij = float(torch.sum(target_info["v_i_tensor"] * hv_target).item())
                cross_curvature_matrix[target_ord][source_ord] = cij
                if target_ord == source_ord:
                    target_info.update({
                        "curvature_method": "exact_hvp",
                        "hvp_frob_norm": float(torch.linalg.vector_norm(hv_target).item()),
                        "v_h_v": cij,
                        "v_h_v_raw_values": [cij],
                        "v_h_v_std": None,
                    })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for target_ord, (target_p, *_target_rest) in enumerate(block_params):
            target_info = block_info[id(target_p)]
            wi = target_info["w_norm"]
            mi = float(target_info["m_i"])
            kind_i = target_info["type"]
            shape_wi = wi / math.sqrt(max(mi, 1e-30))
            typed_cross_sum = 0.0
            all_cross_sum = 0.0
            for source_ord, (source_p, *_source_rest) in enumerate(block_params):
                if source_ord == target_ord:
                    continue
                source_info = block_info[id(source_p)]
                wj = source_info["w_norm"]
                mj = float(source_info["m_i"])
                cij = cross_curvature_matrix[target_ord][source_ord]
                if source_info["type"] == kind_i:
                    typed_den = wi + wj
                    if abs(typed_den) > eps_den:
                        typed_cross_sum += (2.0 * wj / typed_den) * cij
                shape_wj = wj / math.sqrt(max(mj, 1e-30))
                all_den = shape_wi + shape_wj
                if abs(all_den) > eps_den:
                    all_cross_sum += (
                        2.0
                        * math.sqrt(max(mj / max(mi, 1e-30), 0.0))
                        * shape_wj
                        / all_den
                    ) * cij
            self_curv = target_info["v_h_v"]
            target_info["cross_curvature_typed_sum"] = typed_cross_sum
            target_info["cross_curvature_all_sum"] = all_cross_sum
            target_info["weighted_adjusted_curvature_typed"] = self_curv + typed_cross_sum
            target_info["weighted_adjusted_curvature_all"] = self_curv + all_cross_sum

    elif curvature_method == "exact_hvp":
        for block_ord, (p, name, kind, block_idx, _lr, _wd) in enumerate(block_params):
            info = block_info[id(p)]
            v_i = info["v_i_tensor"]
            if dist.get_rank() == 0 and len(block_params) > 8 and block_ord % 12 == 0:
                print0(
                    f"local_quadratic_diag step:{step} hvp_block:{block_ord + 1}/{len(block_params)} {name}",
                    console=True,
                    log=False,
                )
            h_v = _exact_block_hessian_vec(
                model=model,
                diag_inputs=diag_inputs,
                diag_targets=diag_targets,
                param=p,
                vector=v_i,
                num_tokens_diag=num_tokens_diag,
                hvp_microbatch_size=hvp_microbatch_size,
            )
            v_h_v = float(torch.sum(v_i * h_v).item())
            info.update({
                "curvature_method": "exact_hvp",
                "hvp_frob_norm": float(torch.linalg.vector_norm(h_v).item()),
                "v_h_v": v_h_v,
                "v_h_v_raw_values": [v_h_v],
                "v_h_v_std": None,
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elif curvature_method == "block_loss_fd":
        for block_ord, (p, name, kind, block_idx, _lr, _wd) in enumerate(block_params):
            info = block_info[id(p)]
            v_i = info["v_i_tensor"].to(device=p.device, dtype=p.dtype)
            eps_i = float(info["fd_epsilon"])
            if dist.get_rank() == 0 and len(block_params) > 8 and block_ord % 12 == 0:
                print0(
                    f"local_quadratic_diag step:{step} loss_fd_block:{block_ord + 1}/{len(block_params)} {name}",
                    console=True,
                    log=False,
                )
            with torch.no_grad():
                p.add_(v_i, alpha=eps_i)
            f_plus_sum, f_plus = compute_diag_loss_only()
            with torch.no_grad():
                p.add_(v_i, alpha=-2.0 * eps_i)
            f_minus_sum, f_minus = compute_diag_loss_only()
            with torch.no_grad():
                p.add_(v_i, alpha=eps_i)

            v_h_v = (f_plus - 2.0 * f_0 + f_minus) / max(eps_i * eps_i, 1e-30)
            info.update({
                "curvature_method": "block_loss_fd",
                "hvp_frob_norm": None,
                "v_h_v": v_h_v,
                "v_h_v_raw_values": [v_h_v],
                "v_h_v_std": None,
                "fd_loss_base": f_0,
                "fd_loss_base_sum": f_0_sum,
                "fd_loss_plus": f_plus,
                "fd_loss_minus": f_minus,
                "fd_loss_plus_sum": f_plus_sum,
                "fd_loss_minus_sum": f_minus_sum,
            })

    elif curvature_method == "random_grad_fd":
        probe_count = max(1, int(fd_probes))
        curv_values: dict[int, list[float]] = {id(p): [] for p, *_ in block_params}
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed * 1_000_003 + step * 9_176 + diag_index * 101))

        def apply_probe(signs: list[float], scale: float) -> None:
            with torch.no_grad():
                for sign, (p, *_rest) in zip(signs, block_params):
                    info = block_info[id(p)]
                    eps_i = info["fd_epsilon"]
                    v_i = info["v_i_tensor"].to(device=p.device, dtype=p.dtype)
                    p.add_(v_i, alpha=float(scale) * float(sign) * float(eps_i))

        for probe_idx in range(probe_count):
            signs_tensor = torch.randint(
                0, 2, (len(block_params),), generator=gen, dtype=torch.int8
            )
            signs = [1.0 if int(v) else -1.0 for v in signs_tensor.tolist()]
            apply_probe(signs, +1.0)
            f_plus_sum, f_plus, g_plus = compute_diag_gradient()
            apply_probe(signs, -2.0)
            f_minus_sum, f_minus, g_minus = compute_diag_gradient()
            apply_probe(signs, +1.0)
            fd_probe_losses.append({
                "probe_index": int(probe_idx),
                "loss_plus": _safe_float(f_plus),
                "loss_minus": _safe_float(f_minus),
                "loss_plus_sum": _safe_float(f_plus_sum),
                "loss_minus_sum": _safe_float(f_minus_sum),
            })

            for sign, (p, *_rest) in zip(signs, block_params):
                info = block_info[id(p)]
                pid = id(p)
                eps_i = info["fd_epsilon"]
                v_i = info["v_i_tensor"]
                grad_diff = g_plus[pid].float() - g_minus[pid].float()
                dot_diff = float(torch.sum(v_i * grad_diff).item())
                h_hat = float(sign) * dot_diff / max(2.0 * eps_i, 1e-30)
                curv_values[pid].append(h_hat)

        for p, name, kind, block_idx, _lr, _wd in block_params:
            info = block_info[id(p)]
            vals = curv_values[id(p)]
            mean_vhv = sum(vals) / len(vals) if vals else None
            std_vhv = (
                math.sqrt(sum((v - mean_vhv) ** 2 for v in vals) / len(vals))
                if vals and mean_vhv is not None else None
            )
            info.update({
                "curvature_method": "random_grad_fd",
                "hvp_frob_norm": None,
                "v_h_v": mean_vhv,
                "v_h_v_raw_values": vals,
                "v_h_v_std": std_vhv,
            })

    else:
        raise ValueError(f"unknown local_quadratic curvature method {curvature_method!r}")

    if curvature_method == "exact_hvp":
        nonzero_hvp_blocks = sum(
            1
            for info in block_info.values()
            if (info.get("hvp_frob_norm") is not None and info["hvp_frob_norm"] > eps_den)
        )
        if block_info and nonzero_hvp_blocks == 0:
            raise RuntimeError(
                "exact_hvp produced zero HVP for every measured block; "
                "this indicates a broken second-order graph, often from a compiled model path."
            )

    for p, *_rest in block_params:
        info = block_info[id(p)]
        v_h_v = info["v_h_v"]
        denominator = info["w_norm"] * v_h_v if v_h_v is not None else None
        valid_t = (
            v_h_v is not None
            and math.isfinite(info["g_dot_v"])
            and math.isfinite(denominator)
            and abs(denominator) > eps_den
        )
        info["denominator_w_norm_v_h_v"] = denominator
        info["t_i"] = info["g_dot_v"] / denominator if valid_t else None
        info["t_i_valid"] = bool(valid_t)

    typed_x_by_kind: dict[str, float] = {}
    typed_sqrt_m_by_kind: dict[str, float] = {}
    mean_w_by_kind: dict[str, float] = {}
    for kind, pids in by_kind.items():
        sqrt_m = math.sqrt(float(block_info[pids[0]]["m_i"]))
        mean_w = sum(block_info[pid]["w_norm"] for pid in pids) / len(pids)
        typed_sqrt_m_by_kind[kind] = sqrt_m
        mean_w_by_kind[kind] = mean_w
        typed_x_by_kind[kind] = sqrt_m / mean_w

    # Paper-side Muon ratios use the ideal shape-scaled update norm sqrt(m_i).
    # Keep the finite-Newton--Schulz measurement separately for sanity checks;
    # all alignment candidates below use one coherent convention.
    ideal_r_by_pid = {
        pid: math.sqrt(float(info["m_i"])) / info["w_norm"]
        for pid, info in block_info.items()
    }
    ideal_r_values = list(ideal_r_by_pid.values())
    r_bar_all = len(ideal_r_values) / sum(1.0 / value for value in ideal_r_values)

    blocks_out = []
    t_vals = []
    x_plain = []
    x_typed = []
    x_all = []

    for p, name, kind, block_idx, _lr, _wd in block_params:
        info = block_info[id(p)]
        rc = typed_x_by_kind[kind]
        ra = r_bar_all
        r_ideal = ideal_r_by_pid[id(p)]
        ti = info["t_i"]
        if ti is not None:
            t_vals.append(ti)
            x_plain.append(r_ideal)
            x_typed.append(rc)
            x_all.append(ra)

        blocks_out.append({
            "block_name": name,
            "type": kind,
            "block_idx": block_idx,
            "m_i": info["m_i"],
            "n_i": info["n_i"],
            "d_i": info["d_i"],
            "w_norm": _safe_float(info["w_norm"]),
            "r_i": _safe_float(r_ideal),
            "r_i_ideal": _safe_float(r_ideal),
            "r_i_measured": _safe_float(info["r_i"]),
            "u_norm_ideal": _safe_float(math.sqrt(float(info["m_i"]))),
            "vi_raw_norm_over_ideal": _safe_float(
                info["vi_raw_norm"] / math.sqrt(float(info["m_i"]))
            ),
            "rbar_C": _safe_float(rc),
            "rbar_all": _safe_float(ra),
            "x_plain": _safe_float(r_ideal),
            "x_typed": _safe_float(rc),
            "x_all": _safe_float(ra),
            "t_i": _safe_float_or_none(ti),
            "t_i_valid": bool(info["t_i_valid"]),
            "g_dot_v": _safe_float(info["g_dot_v"]),
            "v_h_v": _safe_float_or_none(info["v_h_v"]),
            "cross_curvature_typed_sum": _safe_float_or_none(info.get("cross_curvature_typed_sum")),
            "cross_curvature_all_sum": _safe_float_or_none(info.get("cross_curvature_all_sum")),
            "weighted_adjusted_curvature_typed": _safe_float_or_none(
                info.get("weighted_adjusted_curvature_typed")
            ),
            "weighted_adjusted_curvature_all": _safe_float_or_none(
                info.get("weighted_adjusted_curvature_all")
            ),
            "v_h_v_raw_values": [
                _safe_float(v) for v in info.get("v_h_v_raw_values", [])
                if math.isfinite(float(v))
            ],
            "v_h_v_std": _safe_float_or_none(info.get("v_h_v_std")),
            "denominator_w_norm_v_h_v": _safe_float_or_none(info["denominator_w_norm_v_h_v"]),
            "hvp_frob_norm": _safe_float_or_none(info["hvp_frob_norm"]),
            "curvature_method": info["curvature_method"],
            "fd_epsilon": _safe_float(info["fd_epsilon"]),
            "fd_loss_base": _safe_float_or_none(info.get("fd_loss_base")),
            "fd_loss_plus": _safe_float_or_none(info.get("fd_loss_plus")),
            "fd_loss_minus": _safe_float_or_none(info.get("fd_loss_minus")),
            "fd_loss_base_sum": _safe_float_or_none(info.get("fd_loss_base_sum")),
            "fd_loss_plus_sum": _safe_float_or_none(info.get("fd_loss_plus_sum")),
            "fd_loss_minus_sum": _safe_float_or_none(info.get("fd_loss_minus_sum")),
            "diag_g_frob_norm": _safe_float(info["diag_g_frob_norm"]),
            "train_grad_frob_norm": _safe_float(info["train_grad_frob_norm"]),
            "shape_scale": _safe_float(info["shape_scale"]),
            "vi_raw_norm": _safe_float(info["vi_raw_norm"]),
            "vi_unit_norm": _safe_float(info["vi_unit_norm"]),
            "typed_weight_mean": _safe_float(mean_w_by_kind[kind]),
            "base_lr": _safe_float(info["base_lr"]),
            "lr_current": _safe_float(lr_current),
            "weight_decay_raw": _safe_float(info["weight_decay_raw"]),
            "weight_decay_included_in_direction": False,
        })

    plain_stats = _vector_match_stats(x_plain, t_vals)
    typed_stats = _vector_match_stats(x_typed, t_vals)
    all_stats = _vector_match_stats(x_all, t_vals)

    groups_data = {}
    for kind, pids in by_kind.items():
        valid_pids = [pid for pid in pids if block_info[pid]["t_i"] is not None]
        groups_data[kind] = {
            "rbar_C": _safe_float(typed_x_by_kind[kind]),
            "num_blocks": len(pids),
            "valid_t_block_count": len(valid_pids),
            "mean_r_i": _safe_float(sum(ideal_r_by_pid[pid] for pid in pids) / len(pids)),
            "mean_r_i_measured": _safe_float(
                sum(block_info[pid]["r_i"] for pid in pids) / len(pids)
            ),
            "typed_ideal_sqrt_m": _safe_float(typed_sqrt_m_by_kind[kind]),
            "typed_weight_mean": _safe_float(mean_w_by_kind[kind]),
            "mean_t_i_valid_only": _safe_float_or_none(
                sum(block_info[pid]["t_i"] for pid in valid_pids) / len(valid_pids)
                if valid_pids else None
            ),
            "block_names": [
                block_info[pid]["param_name"]
                for pid in sorted(pids, key=lambda pid: block_info[pid]["block_idx"])
            ],
        }

    return {
        "diagnostic": "muon_shape_scaled_t_scalar_match",
        "diag_index": int(diag_index),
        "step": step,
        "train_loss": _safe_float(train_loss),
        "diag_loss": _safe_float(f_0),
        "diag_loss_sum": _safe_float(f_0_sum),
        "lr_current": _safe_float(lr_current),
        "optimizer": scheme,
        "seed": seed,
        "diag_data_source": diag_data_source,
        "diag_sequence_count_local": int(diag_inputs.size(0)),
        "diag_token_count_global": int(num_tokens_diag),
        "gradient_source": "diagnostic_gradient_per_token_loss",
        "loss_reduction": "per_token_mean",
        "direction_source": "basic_muon_shape_scaled_current_train_direction",
        "direction_formula": "-eta * r_i * ||W_i||_F * V_i",
        "target_scalar_formula": "<G_i,V_i> / (||W_i||_F * <V_i,H_ii V_i>)",
        "hvp": {
            "operator": (
                "exact_block_hessian_of_diagnostic_per_token_loss"
                if curvature_method == "exact_hvp"
                else "blockwise_central_finite_difference_of_diagnostic_loss"
                if curvature_method == "block_loss_fd"
                else "random_sign_central_finite_difference_of_diagnostic_gradient"
            ),
            "microbatch_size": int(hvp_microbatch_size),
            "sdpa_backend": "math" if curvature_method == "exact_hvp" else "default",
            "second_order_quantity": "<V_i,H_ii V_i>",
            "curvature_method": curvature_method,
            "cross_curvature_enabled": bool(measure_cross_curvature),
            "cross_curvature_quantity": "C_ij = <V_i, H_ij V_j>",
            "weighted_adjusted_curvature_definitions": {
                "typed": (
                    "tilde_L_i = C_ii + sum_{j same type, j!=i} "
                    "2||W_j||_F/(||W_i||_F+||W_j||_F) C_ij"
                ),
                "all": (
                    "tilde_L_i = C_ii + sum_{j!=i} "
                    "2 sqrt(m_j/m_i) (||W_j||_F/sqrt(m_j)) / "
                    "((||W_i||_F/sqrt(m_i))+(||W_j||_F/sqrt(m_j))) C_ij"
                ),
            },
            "fd_probe_count": int(fd_probes) if curvature_method == "random_grad_fd" else 0,
            "fd_rel_eps": _safe_float(fd_rel_eps),
            "fd_abs_eps": _safe_float(fd_abs_eps),
            "fd_probe_losses": fd_probe_losses,
            "exact_hvp_eager_diagnostic_model": bool(exact_hvp_eager_diagnostic_model),
        },
        "cuda_memory": {
            "allocated_start_bytes": cuda_allocated_start,
            "reserved_start_bytes": cuda_reserved_start,
            "allocated_peak_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "reserved_peak_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
            "allocated_peak_gb": (
                _safe_float(torch.cuda.max_memory_allocated() / 1024**3)
                if torch.cuda.is_available() else None
            ),
            "reserved_peak_gb": (
                _safe_float(torch.cuda.max_memory_reserved() / 1024**3)
                if torch.cuda.is_available() else None
            ),
        },
        "x_definitions": {
            "plain": "x_i = ideal r_i = sqrt(m_i) / ||W_i||_F",
            "all": "x_i = harmonic_mean_all_blocks(ideal r_i)",
            "typed": "x_i = sqrt(m_G) / mean_{j in G}(||W_j||_F)",
        },
        "muon_norm_convention": (
            "alignment candidates use ideal Muon norm sqrt(m_i); measured finite-NS "
            "ratios are stored only in r_i_measured for sanity checks"
        ),
        "momentum_signal_scale": _safe_float(momentum_debug_scale),
        "momentum_debug_scale": _safe_float(momentum_debug_scale),
        "diag_grad_microbatch_size": int(diag_grad_microbatch_size),
        "included_block_count": len(blocks_out),
        "valid_t_block_count": len(t_vals),
        "included_block_rule": (
            "basic Muon matrix blocks in optimizer2 only; AdamW fallback params excluded"
            if max_blocks <= 0 else
            f"debug subset: first {max_blocks} basic Muon matrix blocks after deterministic ordering"
        ),
        "rbar_all": _safe_float(r_bar_all),
        "cross_curvature": {
            "enabled": bool(measure_cross_curvature),
            "matrix_orientation": "rows are target i, columns are source j; entry is C_ij=<V_i,H_ij V_j>",
            "order": cross_curvature_order if measure_cross_curvature else None,
            "matrix": (
                [
                    [_safe_float(v) for v in row]
                    for row in cross_curvature_matrix
                ]
                if measure_cross_curvature and save_cross_curvature_matrix else None
            ),
            "matrix_saved": bool(measure_cross_curvature and save_cross_curvature_matrix),
        },
        "similarity_metric_primary": "cosine; higher is more similar",
        "distance_metric": "relative_l2_after_best_scale; lower is better",
        "plain": plain_stats,
        "typed": typed_stats,
        "all": all_stats,
        "cosine_plain_t": plain_stats["cosine"],
        "cosine_typed_t": typed_stats["cosine"],
        "cosine_all_t": all_stats["cosine"],
        "relative_l2_plain_t": plain_stats["relative_l2_after_best_scale"],
        "relative_l2_typed_t": typed_stats["relative_l2_after_best_scale"],
        "relative_l2_all_t": all_stats["relative_l2_after_best_scale"],
        "pearson_plain_t": plain_stats["pearson"],
        "pearson_typed_t": typed_stats["pearson"],
        "pearson_all_t": all_stats["pearson"],
        "ordering": {
            "cosine_typed_gt_plain": bool(
                typed_stats["cosine"] is not None and plain_stats["cosine"] is not None
                and typed_stats["cosine"] > plain_stats["cosine"]
            ),
            "cosine_all_gt_plain": bool(
                all_stats["cosine"] is not None and plain_stats["cosine"] is not None
                and all_stats["cosine"] > plain_stats["cosine"]
            ),
            "cosine_typed_gt_all": bool(
                typed_stats["cosine"] is not None and all_stats["cosine"] is not None
                and typed_stats["cosine"] > all_stats["cosine"]
            ),
            "l2_typed_lt_plain": bool(
                typed_stats["relative_l2_after_best_scale"] is not None
                and plain_stats["relative_l2_after_best_scale"] is not None
                and typed_stats["relative_l2_after_best_scale"] < plain_stats["relative_l2_after_best_scale"]
            ),
            "l2_all_lt_plain": bool(
                all_stats["relative_l2_after_best_scale"] is not None
                and plain_stats["relative_l2_after_best_scale"] is not None
                and all_stats["relative_l2_after_best_scale"] < plain_stats["relative_l2_after_best_scale"]
            ),
            "l2_typed_lt_all": bool(
                typed_stats["relative_l2_after_best_scale"] is not None
                and all_stats["relative_l2_after_best_scale"] is not None
                and typed_stats["relative_l2_after_best_scale"] < all_stats["relative_l2_after_best_scale"]
            ),
        },
        "groups": groups_data,
        "blocks": blocks_out,
    }

