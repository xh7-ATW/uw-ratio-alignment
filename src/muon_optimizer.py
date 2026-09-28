"""Muon-family optimizers and u-w floor logic used by the paper."""

from __future__ import annotations

import re

import torch
from torch import Tensor
import torch.distributed as dist


_BLOCK_WEIGHT_RE = re.compile(
    r"^blocks\.(\d+)\.(attn\.(?:q|k|v|proj)|mlp\.(?:fc|proj))\.weight$"
)


def _matrix_kind_from_name(name: str) -> str | None:
    match = _BLOCK_WEIGHT_RE.match(name)
    return match.group(2).replace(".", "_") if match else None


def _block_idx_from_name(name: str) -> int | None:
    match = _BLOCK_WEIGHT_RE.match(name)
    return int(match.group(1)) if match else None


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 12, eps: float = 1e-7) -> Tensor:
    assert G.ndim == 2
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + eps)
    a, b, c = 2, -1.5, 0.5
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class DistributedMuonLKindwise(torch.optim.Optimizer):
    """
    Muon optimizer for the paper's matrix-update policies.

    ``plain`` is baseline Muon. ``kindwise`` is t-MuonL: matrices are rescaled
    by their type-level weight-norm ratio. ``alllayer`` is MuonL: matrices are
    rescaled by their global u-w ratio alignment coefficient. ``normuon`` is
    the benchmark #9 NorMuon-lite update: per-row/column second-moment
    normalization after Newton-Schulz, followed by Frobenius renormalization.
    Optional fixed u-w flooring acts on the pre-learning-rate update direction.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0.0,
        momentum=0.95,
        ns_steps=12,
        nesterov=True,
        rho=1.0,
        interval=5,
        param_names=None,
        eps=1e-12,
        scheme="kindwise",
        scale_ema_beta=0.9,
        normuon_beta2=0.95,
        orscale_r_min=0.1,
        orscale_r_max=5.0,
        orscale_general_shape_scale="main",
        uw_floor_mode="off",
        uw_floor_c=0.0,
        uw_floor_eps=1e-12,
    ):
        assert scheme in (
            "plain", "kindwise", "alllayer", "normuon",
            "orscale_general", "orscale_general_decoupled_clipped",
        ), (
            f"unknown scheme {scheme!r}"
        )
        uw_floor_mode = str(uw_floor_mode).strip().lower()
        if uw_floor_mode not in ("off", "fixed"):
            raise ValueError("uw_floor_mode must be off or fixed")
        if uw_floor_mode == "fixed" and float(uw_floor_c) <= 0.0:
            raise ValueError("fixed uw_floor_mode requires uw_floor_c > 0")
        if float(uw_floor_eps) <= 0.0:
            raise ValueError("uw_floor_eps must be > 0")
        if float(orscale_r_min) > float(orscale_r_max):
            raise ValueError("orscale_r_min must be <= orscale_r_max")
        orscale_general_shape_scale = str(orscale_general_shape_scale).lower()
        if orscale_general_shape_scale not in ("unit", "main"):
            raise ValueError("orscale_general_shape_scale must be unit or main")

        defaults = dict(
            lr=float(lr),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
            ns_steps=max(1, int(ns_steps)),
            nesterov=bool(nesterov),
            rho=float(rho) if scheme != "plain" else 0.0,
            interval=int(interval),
            eps=float(eps),
            scheme=scheme,
            scale_ema_beta=float(scale_ema_beta),
            normuon_beta2=float(normuon_beta2),
            orscale_r_min=float(orscale_r_min),
            orscale_r_max=float(orscale_r_max),
            orscale_general_shape_scale=orscale_general_shape_scale,
            uw_floor_mode=uw_floor_mode,
            uw_floor_c=float(uw_floor_c),
            uw_floor_eps=float(uw_floor_eps),
        )
        super().__init__(params, defaults)
        self.global_step = 0
        self.scheme = scheme
        self._param_kind_by_id: dict[int, str] = {}
        self._param_block_by_id: dict[int, int] = {}
        self._param_name_by_id: dict[int, str] = {}

        if param_names is not None:
            if isinstance(param_names, dict):
                name_by_id = param_names
            else:
                all_params = [p for group in self.param_groups for p in group["params"]]
                name_by_id = {id(p): name for name, p in zip(param_names, all_params)}
            for group in self.param_groups:
                for p in group["params"]:
                    name = name_by_id.get(id(p))
                    if name is None:
                        continue
                    self._param_name_by_id[id(p)] = name
                    self._param_kind_by_id[id(p)] = (
                        _matrix_kind_from_name(name) or f"shape:{tuple(p.shape)}"
                    )
                    block_idx = _block_idx_from_name(name)
                    if block_idx is not None:
                        self._param_block_by_id[id(p)] = block_idx

    def _matrix_kind(self, p: Tensor) -> str:
        return self._param_kind_by_id.get(id(p), f"shape:{tuple(p.shape)}")

    def _ensure_param_state(self, p: Tensor) -> dict:
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)
            state.setdefault("matrix_kind", self._matrix_kind(p))
            state.setdefault("cached_w_norm", torch.ones((), dtype=torch.float32, device=p.device))
            state.setdefault("cached_relative_scale", torch.ones((), dtype=torch.float32, device=p.device))
        return state

    def _apply_normuon_normalization(self, p: Tensor, update: Tensor) -> Tensor:
        """Apply benchmark #9's NorMuon-lite row/column variance normalization."""

        state = self.state[p]
        if "normuon_second_moment" not in state:
            if update.size(-2) >= update.size(-1):
                shape = (*update.shape[:-1], 1)
            else:
                shape = (*update.shape[:-2], 1, update.shape[-1])
            state["normuon_second_moment"] = torch.zeros(
                shape, dtype=torch.float32, device=update.device
            )

        if update.size(-2) >= update.size(-1):
            variance = update.float().square().mean(dim=-1, keepdim=True)
        else:
            variance = update.float().square().mean(dim=-2, keepdim=True)

        second_moment = state["normuon_second_moment"]
        beta2 = float(self.defaults["normuon_beta2"])
        second_moment.lerp_(variance, 1.0 - beta2)

        original_norm = torch.linalg.vector_norm(update.float())
        normalized = update * second_moment.clamp_min(1e-10).rsqrt().to(update.dtype)
        normalized_norm = torch.linalg.vector_norm(normalized.float()).clamp_min(1e-10)
        return normalized * (original_norm / normalized_norm).to(normalized.dtype)

    def _update_relative_scale_ema(self, p: Tensor, raw_scale: Tensor, beta: float) -> Tensor:
        state = self.state[p]
        raw_scale = raw_scale.to(device=p.device, dtype=torch.float32)
        if beta <= 0.0:
            state["relative_scale_ema"] = raw_scale.clone()
            state["relative_scale_ema_updates"] = 1
            return raw_scale
        if "relative_scale_ema" not in state:
            state["relative_scale_ema"] = torch.zeros((), dtype=torch.float32, device=p.device)
            state["relative_scale_ema_updates"] = 0
        ema = state["relative_scale_ema"]
        ema.mul_(beta).add_(raw_scale, alpha=1.0 - beta)
        updates = int(state.get("relative_scale_ema_updates", 0)) + 1
        state["relative_scale_ema_updates"] = updates
        bias_correction = max(float(state.get("eps", 1e-12)), 1.0 - beta ** updates)
        return ema / bias_correction

    def _uw_norms_and_ratio(self, p: Tensor, update: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eps = float(self.defaults["uw_floor_eps"])
        p_norm = torch.linalg.vector_norm(p.detach().float()).clamp_min(eps)
        update_norm = torch.linalg.vector_norm(update.detach().float())
        return p_norm, update_norm, update_norm / p_norm

    def _active_uw_floor_target(self) -> float | None:
        if str(self.defaults["uw_floor_mode"]) == "fixed":
            return float(self.defaults["uw_floor_c"])
        return None

    def _apply_uw_floor_to_update(
        self,
        p: Tensor,
        update: Tensor,
        lr: float,
        target: float | None,
    ) -> Tensor:
        mode = str(self.defaults["uw_floor_mode"])
        state = self.state[p]
        p_norm, update_norm, raw_ratio = self._uw_norms_and_ratio(p, update)
        if target is None or target <= 0.0 or lr <= 0.0:
            state["uw_floor_mode"] = mode
            state["uw_floor_w_norm_pre_update"] = p_norm.detach().float()
            state["uw_floor_update_norm_pre_lr"] = update_norm.detach().float()
            state["uw_floor_raw_ratio"] = raw_ratio.detach().float()
            state["uw_floor_target"] = None
            state["uw_floor_scale"] = torch.ones((), device=p.device, dtype=torch.float32)
            state["uw_floor_effective_ratio"] = raw_ratio.detach().float()
            state["uw_floor_applied"] = False
            return update

        eps = float(self.defaults["uw_floor_eps"])
        target_t = torch.tensor(float(target), device=p.device, dtype=torch.float32)
        scale = torch.where(
            raw_ratio < target_t,
            target_t * p_norm / update_norm.clamp_min(eps),
            torch.ones_like(raw_ratio),
        )
        state["uw_floor_mode"] = mode
        state["uw_floor_w_norm_pre_update"] = p_norm.detach().float()
        state["uw_floor_update_norm_pre_lr"] = update_norm.detach().float()
        state["uw_floor_raw_ratio"] = raw_ratio.detach().float()
        state["uw_floor_target"] = target_t.detach()
        state["uw_floor_scale"] = scale.detach().float()
        state["uw_floor_effective_ratio"] = (raw_ratio * scale).detach().float()
        state["uw_floor_applied"] = bool((scale > 1.0).item())
        return update * scale.to(dtype=update.dtype)

    def _apply_weight_decay_and_update(
        self,
        p: Tensor,
        update: Tensor,
        lr: float,
        weight_decay: float,
    ) -> None:
        if weight_decay != 0.0:
            p.mul_(1 - lr * weight_decay)
        p.add_(update.reshape_as(p), alpha=-lr)

    @torch.no_grad()
    def _refresh_kindwise_scales(self) -> None:
        if self.scheme != "kindwise":
            return
        interval = self.defaults["interval"]
        needs_refresh = (
            any(
                "cached_relative_scale" not in self.state[p]
                for group in self.param_groups
                for p in group["params"]
                if p.ndim >= 2
            )
            or self.global_step % interval == 0
        )
        if not needs_refresh:
            return

        eps = self.defaults["eps"]
        beta = self.defaults["scale_ema_beta"]
        all_matrix = [
            (p, group) for group in self.param_groups for p in group["params"] if p.ndim >= 2
        ]
        norms = {id(p): torch.linalg.vector_norm(p.detach().float()) for p, _ in all_matrix}
        by_kind: dict[str, list[Tensor]] = {}
        for p, _ in all_matrix:
            by_kind.setdefault(self._matrix_kind(p), []).append(p)
        for kind, params_k in by_kind.items():
            avg_norm_unclamped = torch.stack([norms[id(p)] for p in params_k]).mean()
            avg_k = avg_norm_unclamped.clamp_min(eps)
            zero_kind = bool(avg_norm_unclamped <= eps)
            for p in params_k:
                raw = torch.ones_like(norms[id(p)]) if zero_kind else norms[id(p)] / avg_k
                self.state[p]["cached_relative_scale"] = self._update_relative_scale_ema(
                    p, raw, beta
                )
                self.state[p]["cached_w_norm"] = norms[id(p)]
                self.state[p]["matrix_kind"] = kind

    @torch.no_grad()
    def step(self):
        if self.scheme in ("orscale_general", "orscale_general_decoupled_clipped"):
            self._step_orscale_general()
            return
        if self.scheme == "alllayer":
            self._step_alllayer_ratio_aligned()
            return

        world_size = dist.get_world_size()
        rank = dist.get_rank()
        self._refresh_kindwise_scales()
        uw_floor_target = self._active_uw_floor_target()

        for group in self.param_groups:
            beta = float(group["momentum"])
            ns_steps = max(1, int(group["ns_steps"]))
            nesterov = bool(group["nesterov"])
            rho = float(group["rho"])
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)

            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    if p.grad is not None:
                        state = self._ensure_param_state(p)
                        grad = p.grad
                        buf = state["momentum_buffer"]
                        buf.lerp_(grad, 1 - beta)
                        update = grad.lerp_(buf, beta) if nesterov else buf
                        state["last_update_buffer_pre_ns"] = update.detach().clone()
                        update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                        if self.scheme == "normuon":
                            update = update.bfloat16()
                        update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
                        if self.scheme == "normuon":
                            update = self._apply_normuon_normalization(p, update)
                        if rho > 0.0:
                            rel = state["cached_relative_scale"].to(dtype=update.dtype)
                            update = update * rel.pow(rho)
                        update = self._apply_uw_floor_to_update(
                            p,
                            update,
                            float(group["lr"]),
                            uw_floor_target,
                        )
                        self._apply_weight_decay_and_update(
                            p,
                            update,
                            float(group["lr"]),
                            float(group["weight_decay"]),
                        )

                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])

        self.global_step += 1

    @torch.no_grad()
    def _step_orscale_general(self) -> None:
        """Historical Figure-8 OrScale update, exposed through the public runner."""
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        eps = float(self.defaults["eps"])
        r_min = float(self.defaults["orscale_r_min"])
        r_max = float(self.defaults["orscale_r_max"])
        decoupled = self.scheme == "orscale_general_decoupled_clipped"

        for group in self.param_groups:
            beta = float(group["momentum"])
            ns_steps = max(1, int(group["ns_steps"]))
            nesterov = bool(group["nesterov"])
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (
                world_size - len(params) % world_size
            )
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    if p.grad is not None:
                        state = self._ensure_param_state(p)
                        grad = p.grad
                        buf = state["momentum_buffer"]
                        buf.mul_(beta).add_(grad)
                        pre_ns = grad.add(buf, alpha=beta) if nesterov else buf
                        q = zeropower_via_newtonschulz5(pre_ns, steps=ns_steps).float()
                        if self.defaults["orscale_general_shape_scale"] == "main":
                            q.mul_(max(1.0, q.size(-2) / q.size(-1)) ** 0.5)
                        direction = q
                        if wd and not decoupled:
                            direction = direction.add(p.detach().float(), alpha=wd)
                        w_norm = torch.linalg.vector_norm(p.detach().float())
                        direction_norm = torch.linalg.vector_norm(direction.detach().float())
                        raw_ratio = w_norm.clamp_min(eps) / direction_norm.add(eps)
                        ratio = torch.clamp(raw_ratio, min=r_min, max=r_max)
                        state["last_update_buffer_pre_ns"] = pre_ns.detach().clone()
                        state["cached_w_norm"] = w_norm.detach()
                        state["cached_relative_scale"] = ratio.detach()
                        state["matrix_kind"] = self._matrix_kind(p)
                        state["orscale_general_raw_trust_ratio"] = raw_ratio.detach()
                        state["orscale_general_effective_trust_ratio"] = ratio.detach()
                        scaled = direction * ratio.to(direction.dtype)
                        if decoupled and wd:
                            p.mul_(1 - lr * wd)
                        p.add_(scaled.reshape_as(p).to(p.dtype), alpha=-lr)
                dist.all_gather(
                    params_pad[base_i:base_i + world_size],
                    params_pad[base_i + rank],
                )
        self.global_step += 1

    @torch.no_grad()
    def _step_alllayer_ratio_aligned(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        eps = float(self.defaults["eps"])
        interval = int(self.defaults["interval"])
        beta_ema = float(self.defaults["scale_ema_beta"])
        needs_refresh = self.global_step % interval == 0

        first_param = next(p for group in self.param_groups for p in group["params"])
        ratio_sum = torch.zeros((), dtype=torch.float32, device=first_param.device)
        ratio_count = torch.zeros((), dtype=torch.float32, device=first_param.device)
        pending_by_group = []

        for group in self.param_groups:
            beta = float(group["momentum"])
            ns_steps = max(1, int(group["ns_steps"]))
            nesterov = bool(group["nesterov"])
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            pending = []

            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    if p.grad is not None:
                        state = self._ensure_param_state(p)
                        if "relative_scale_ema" not in state:
                            needs_refresh = True
                        grad = p.grad
                        buf = state["momentum_buffer"]
                        buf.lerp_(grad, 1 - beta)
                        update = grad.lerp_(buf, beta) if nesterov else buf
                        state["last_update_buffer_pre_ns"] = update.detach().clone()
                        update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                        update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

                        w_norm = torch.linalg.vector_norm(p.detach().float())
                        update_norm = torch.linalg.vector_norm(update.detach().float()).clamp_min(eps)
                        zero_weight = bool(float(w_norm.item()) <= eps)
                        ratio = None
                        if needs_refresh and not zero_weight:
                            ratio = w_norm / update_norm
                            ratio_sum.add_(ratio.detach().float())
                            ratio_count.add_(1.0)

                        state["cached_w_norm"] = w_norm.detach().to(device=p.device, dtype=torch.float32)
                        state["matrix_kind"] = self._matrix_kind(p)
                        pending.append((p, update, state, ratio))

            pending_by_group.append((group, params, params_pad, pending))

        ratio_mean = None
        zero_all = False
        if needs_refresh:
            dist.all_reduce(ratio_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(ratio_count, op=dist.ReduceOp.SUM)
            ratio_mean = ratio_sum / ratio_count.clamp_min(1.0)
            zero_all = bool(float(ratio_mean.item()) <= eps)

        for group, params, params_pad, pending in pending_by_group:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            rho = float(group["rho"])
            for p, update, state, ratio in pending:
                if needs_refresh:
                    if ratio is None or rho <= 0.0 or zero_all:
                        raw_coeff = torch.ones((), dtype=torch.float32, device=p.device)
                    else:
                        raw_coeff = (ratio / ratio_mean.clamp_min(eps)).pow(rho)
                        raw_coeff = raw_coeff.to(device=p.device, dtype=torch.float32)
                    state["cached_relative_scale"] = self._update_relative_scale_ema(
                        p, raw_coeff, beta_ema
                    )
                coeff = state["cached_relative_scale"].to(device=p.device, dtype=update.dtype)
                self._apply_weight_decay_and_update(p, update * coeff, lr, wd)

            for base_i in range(0, len(params), world_size):
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])

        self.global_step += 1
