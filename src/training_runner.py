"""Training runner for u-w ratio alignment experiments.

The public script in ``scripts/train/train_alignment.py`` calls ``main()`` here.
This module owns experiment setup, logging, validation, and the training loop;
model, data, optimizer, and diagnostic kernels live in neighboring modules.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from torch.optim import AdamW
import torch.distributed as dist

from src.data import distributed_data_generator
from src.model import GPT
from src.muon_optimizer import DistributedMuonLKindwise
from src.utils import iclr_code_dir, repository_root
from src.diagnostic_utils import (
    safe_float as _safe_float,
    safe_float_or_none as _safe_float_or_none,
    set_diagnostic_logger,
)
from src.local_quadratic_diagnostics import compute_local_quadratic_diagnostics

MODULE_PATH = Path(__file__).resolve()
ICLR_DIR = iclr_code_dir(MODULE_PATH)
REPO_ROOT = repository_root(ICLR_DIR)
DATA_DIR = Path(os.getenv("MYOPT_DATA_DIR", str(REPO_ROOT / "data" / "fineweb10B"))).expanduser().resolve()
ENTRYPOINT = ICLR_DIR / "scripts" / "train" / "train_alignment.py"


def _source_snapshot_for_log() -> str:
    source_paths = [
        ENTRYPOINT,
        MODULE_PATH,
        ICLR_DIR / "src" / "data.py",
        ICLR_DIR / "src" / "model.py",
        ICLR_DIR / "src" / "muon_optimizer.py",
        ICLR_DIR / "src" / "diagnostic_utils.py",
        ICLR_DIR / "src" / "local_quadratic_diagnostics.py",
    ]
    chunks = []
    for path in source_paths:
        try:
            chunks.append(f"# === {path.relative_to(ICLR_DIR)} ===\n" + path.read_text())
        except OSError as exc:
            chunks.append(f"# === {path} unavailable: {exc} ===")
    return "\n\n".join(chunks)


def main() -> None:
    ########################################
    #                Setup                 #
    ########################################

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    assert 8 % dist.get_world_size() == 0

    _result_dir_override = os.getenv("MYOPT_OUTPUT_DIR", "").strip()
    if _result_dir_override:
        RESULT_DIR = Path(_result_dir_override)
        if not RESULT_DIR.is_absolute():
            RESULT_DIR = REPO_ROOT / RESULT_DIR
    else:
        RESULT_DIR = ENTRYPOINT.parent
    logfile = None
    val_loss_file = None
    u_over_w_file = None
    weight_norm_file = None

    def print0(s, console=False, log=True):
        if dist.get_rank() == 0:
            if console:
                print(s)
            if log and logfile is not None:
                with open(logfile, "a") as f:
                    print(s, file=f)


    set_diagnostic_logger(print0)


    ########################################
    #           Hyperparameters            #
    ########################################

    FIXED_SEED = int(os.getenv("MYOPT_SEED", "42"))
    if FIXED_SEED < 0:
        raise ValueError("MYOPT_SEED must be non-negative")
    MYOPT_SCHEME = os.getenv("MYOPT_SCHEME", "plain_muon")
    VALID_MYOPT_SCHEMES = (
        "plain_muon",
        "normuon",
        "typed_muonl",
        "alllayer_muonl",
        "orscale_general",
        "orscale_general_decoupled_clipped",
    )
    assert MYOPT_SCHEME in VALID_MYOPT_SCHEMES, \
        f"MYOPT_SCHEME must be one of {VALID_MYOPT_SCHEMES}, got {MYOPT_SCHEME!r}"

    # Translate scheme name to optimizer internals
    SCHEME_TO_OPT = {
        "plain_muon":                         ("plain",    0.0),
        "normuon":                            ("normuon",  0.0),
        "typed_muonl":                        ("kindwise", 1.0),
        "alllayer_muonl":                     ("alllayer", 1.0),
        "orscale_general":                    ("orscale_general", 1.0),
        "orscale_general_decoupled_clipped":  ("orscale_general_decoupled_clipped", 1.0),
    }
    OPT_MODE, OPT_RHO = SCHEME_TO_OPT[MYOPT_SCHEME]

    MYOPT_SAVE_DIAG = bool(int(os.getenv("MYOPT_SAVE_DIAG", "0")))
    MYOPT_DIAG_MODE = os.getenv("MYOPT_DIAG_MODE", "local_quadratic").strip().lower()
    if MYOPT_DIAG_MODE != "local_quadratic":
        raise ValueError("public paper code only supports MYOPT_DIAG_MODE=local_quadratic")
    if MYOPT_SAVE_DIAG and MYOPT_SCHEME != "plain_muon":
        raise ValueError("local quadratic diagnostics are defined on the Muon trajectory")

    MYOPT_MOMENTUM     = 0.95
    MYOPT_NESTEROV     = bool(int(os.getenv("MYOPT_NESTEROV", "1")))
    MYOPT_NS_STEPS     = int(os.getenv("MYOPT_NS_STEPS", "12"))
    MYOPT_INTERVAL     = 5
    MYOPT_SCALE_EMA_BETA = 0.9
    MYOPT_NORMUON_BETA2 = float(os.getenv("MYOPT_NORMUON_BETA2", "0.95"))
    MYOPT_ORSCALE_R_MIN = float(os.getenv("MYOPT_ORSCALE_R_MIN", "0.1"))
    MYOPT_ORSCALE_R_MAX = float(os.getenv("MYOPT_ORSCALE_R_MAX", "5.0"))
    MYOPT_ORSCALE_GENERAL_SHAPE_SCALE = os.getenv(
        "MYOPT_ORSCALE_GENERAL_SHAPE_SCALE", "main"
    )
    MYOPT_LR           = float(os.getenv("MYOPT_LR", "0.03"))
    MYOPT_WD           = float(os.getenv("MYOPT_WD", "0.025"))
    MYOPT_QKV_LR       = float(os.getenv("MYOPT_QKV_LR", str(MYOPT_LR)))
    MYOPT_ATTN_PROJ_LR = float(os.getenv("MYOPT_ATTN_PROJ_LR", str(MYOPT_LR)))
    MYOPT_MLP_FC_LR    = float(os.getenv("MYOPT_MLP_FC_LR", str(MYOPT_LR)))
    MYOPT_MLP_PROJ_LR  = float(os.getenv("MYOPT_MLP_PROJ_LR", str(MYOPT_LR)))
    MYOPT_QKV_WD       = float(os.getenv("MYOPT_QKV_WD", str(MYOPT_WD)))
    MYOPT_ATTN_PROJ_WD = float(os.getenv("MYOPT_ATTN_PROJ_WD", str(MYOPT_WD)))
    MYOPT_MLP_FC_WD    = float(os.getenv("MYOPT_MLP_FC_WD", str(MYOPT_WD)))
    MYOPT_MLP_PROJ_WD  = float(os.getenv("MYOPT_MLP_PROJ_WD", str(MYOPT_WD)))
    MYOPT_COOLDOWN_FRAC = float(os.getenv("MYOPT_COOLDOWN_FRAC", "0.7"))
    if not (0.0 < MYOPT_COOLDOWN_FRAC <= 1.0):
        raise ValueError("MYOPT_COOLDOWN_FRAC must be in (0, 1]")
    MYOPT_UW_FLOOR_MODE = os.getenv("MYOPT_UW_FLOOR_MODE", "off").strip().lower()
    if MYOPT_UW_FLOOR_MODE in ("0", "false", "none", "no"):
        MYOPT_UW_FLOOR_MODE = "off"
    if MYOPT_UW_FLOOR_MODE in ("1", "true", "constant", "const", "c"):
        MYOPT_UW_FLOOR_MODE = "fixed"
    if MYOPT_UW_FLOOR_MODE not in ("off", "fixed"):
        raise ValueError("MYOPT_UW_FLOOR_MODE must be off or fixed")
    MYOPT_UW_FLOOR_C = float(os.getenv("MYOPT_UW_FLOOR_C", "0.0"))
    MYOPT_UW_FLOOR_EPS = float(os.getenv("MYOPT_UW_FLOOR_EPS", "1e-12"))
    if MYOPT_UW_FLOOR_MODE == "fixed" and MYOPT_UW_FLOOR_C <= 0.0:
        raise ValueError("fixed MYOPT_UW_FLOOR_MODE requires MYOPT_UW_FLOOR_C > 0")
    if MYOPT_UW_FLOOR_EPS <= 0.0:
        raise ValueError("MYOPT_UW_FLOOR_EPS must be > 0")
    DIAG_FREQ          = int(os.getenv("MYOPT_DIAG_FREQ", "250"))
    LOCAL_QUADRATIC_HVP_MBS = int(os.getenv("MYOPT_LOCAL_QUADRATIC_HVP_MBS", "1"))
    LOCAL_QUADRATIC_MAX_BLOCKS = int(os.getenv("MYOPT_LOCAL_QUADRATIC_MAX_BLOCKS", "0"))
    LOCAL_QUADRATIC_EPS_W = float(os.getenv("MYOPT_LOCAL_QUADRATIC_EPS_W", "1e-12"))
    LOCAL_QUADRATIC_EPS_DEN = float(os.getenv("MYOPT_LOCAL_QUADRATIC_EPS_DEN", "1e-30"))
    LOCAL_QUADRATIC_CURVATURE_METHOD = os.getenv("MYOPT_LOCAL_QUADRATIC_CURVATURE", "exact_hvp").strip().lower()
    if LOCAL_QUADRATIC_CURVATURE_METHOD != "exact_hvp":
        raise ValueError("public paper diagnostics use MYOPT_LOCAL_QUADRATIC_CURVATURE=exact_hvp")
    LOCAL_QUADRATIC_CROSS_CURVATURE = bool(int(os.getenv("MYOPT_LOCAL_QUADRATIC_CROSS_CURVATURE", "0")))
    LOCAL_QUADRATIC_SAVE_CROSS_CURVATURE_MATRIX = bool(int(os.getenv("MYOPT_LOCAL_QUADRATIC_SAVE_CROSS_CURVATURE_MATRIX", "1")))
    LOCAL_QUADRATIC_DATA_SOURCE = os.getenv("MYOPT_LOCAL_QUADRATIC_DATA_SOURCE", "train_batch").strip().lower()
    if LOCAL_QUADRATIC_DATA_SOURCE in ("train", "current_train", "current_training_batch"):
        LOCAL_QUADRATIC_DATA_SOURCE = "train_batch"
    if LOCAL_QUADRATIC_DATA_SOURCE in ("val", "fixed_val", "validation"):
        LOCAL_QUADRATIC_DATA_SOURCE = "fixed_val"
    if LOCAL_QUADRATIC_DATA_SOURCE not in ("train_batch", "fixed_val"):
        raise ValueError("MYOPT_LOCAL_QUADRATIC_DATA_SOURCE must be train_batch or fixed_val")
    LOCAL_QUADRATIC_EXACT_HVP_EAGER_DIAG_MODEL = bool(
        MYOPT_SAVE_DIAG and LOCAL_QUADRATIC_CURVATURE_METHOD == "exact_hvp"
    )
    MYOPT_SAVE_LOG = bool(int(os.getenv("MYOPT_SAVE_LOG", "1" if MYOPT_SAVE_DIAG else "0")))
    MYOPT_SAVE_U_OVER_W_JSON = bool(int(os.getenv("MYOPT_SAVE_U_OVER_W_JSON", "0")))
    MYOPT_U_OVER_W_FREQ = int(os.getenv("MYOPT_U_OVER_W_FREQ", "5"))
    if MYOPT_U_OVER_W_FREQ <= 0:
        raise ValueError("MYOPT_U_OVER_W_FREQ must be > 0")
    MYOPT_SAVE_VAL_LOSS_JSONL = bool(int(os.getenv("MYOPT_SAVE_VAL_LOSS_JSONL", "0")))
    MYOPT_SAVE_WEIGHT_NORMS = bool(int(os.getenv("MYOPT_SAVE_WEIGHT_NORMS", "0")))
    MYOPT_WEIGHT_NORM_FREQ = int(os.getenv("MYOPT_WEIGHT_NORM_FREQ", "1"))
    if MYOPT_WEIGHT_NORM_FREQ <= 0:
        raise ValueError("MYOPT_WEIGHT_NORM_FREQ must be > 0")

    EARLY_VAL_STEP_FREQ = 125
    MID_VAL_STEP_FREQ   = int(os.getenv("MYOPT_MID_VAL_FREQ", "25"))
    MID_VAL_START_STEP  = int(os.getenv("MYOPT_MID_VAL_START", "3000"))
    LATE_VAL_STEP_FREQ  = int(os.getenv("MYOPT_LATE_VAL_FREQ", "10"))
    LATE_VAL_START_STEP = int(os.getenv("MYOPT_LATE_VAL_START", "3200"))
    train_steps         = int(os.getenv("MYOPT_TRAIN_STEPS", "3300"))
    schedule_steps      = int(os.getenv("MYOPT_SCHEDULE_STEPS", str(train_steps)))

    sequence_length = 1024
    train_microbatches = 8
    batch_size  = 8 * 64 * sequence_length
    train_microbatch_sequences = batch_size // (sequence_length * train_microbatches)
    diag_mbs    = int(os.getenv("MYOPT_DIAG_MBS", "64"))
    diag_grad_mbs = int(os.getenv("MYOPT_DIAG_GRAD_MBS", str(train_microbatch_sequences)))
    val_tokens  = int(os.getenv("MYOPT_VAL_TOKENS", str(20 * 524288)))

    LOCAL_QUADRATIC_DONATED_BUFFER_DISABLED = False
    if MYOPT_DIAG_MODE == "local_quadratic":
        try:
            import torch._functorch.config as _functorch_config
            _functorch_config.donated_buffer = False
            LOCAL_QUADRATIC_DONATED_BUFFER_DISABLED = True
        except Exception:
            LOCAL_QUADRATIC_DONATED_BUFFER_DISABLED = False

    ########################################
    #          Data & Model Init           #
    ########################################

    val_inputs, val_targets = next(distributed_data_generator(
        str(DATA_DIR / "fineweb_val_*.bin"), val_tokens))

    # Diagnostic batch: first diag_mbs sequences from val data (fixed, same across all steps)
    diag_inputs  = val_inputs[:diag_mbs]
    diag_targets = val_targets[:diag_mbs]

    init_mode = os.getenv("MYOPT_INIT_MODE", "paper_explicit").strip().lower()
    if init_mode == "benchmark_default":
        torch.manual_seed(FIXED_SEED)
        torch.cuda.manual_seed_all(FIXED_SEED)

    model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()

    if init_mode == "benchmark_default":
        for name, p in model.named_parameters():
            if "proj" in name:
                p.data.zero_()
    elif init_mode == "paper_explicit":
        # Preserve the initialization order used by the existing paper runs.
        torch.manual_seed(FIXED_SEED)
        torch.cuda.manual_seed_all(FIXED_SEED)
        for name, p in model.named_parameters():
            w = p.data
            if name.endswith("weight"):
                if "proj" in name:
                    w.zero_()
                elif "embed" in name:
                    w.normal_()
                else:
                    w.normal_(std=0.33**0.5 / w.size(-1)**0.5)
            elif name.endswith("bias"):
                w.zero_()
            elif name.endswith("gains"):
                w.normal_(mean=1, std=0)
            else:
                raise Exception(f"Uninitialized parameter: {name}")
    else:
        raise ValueError("MYOPT_INIT_MODE must be paper_explicit or benchmark_default")

    train_model = torch.compile(model, dynamic=False)

    ########################################
    #            Optimizers                #
    ########################################

    optimizer1 = AdamW(
        [dict(params=[model.embed.weight], lr=0.3),
         dict(params=[model.proj.weight], lr=1/320),
         dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
        betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)

    named_block_params = [(n, p) for n, p in model.named_parameters()
                          if n.startswith("blocks.") and p.ndim >= 2]
    qkv_named      = [(n, p) for n, p in named_block_params
                      if n.endswith((".attn.q.weight", ".attn.k.weight", ".attn.v.weight"))]
    attn_out_named = [(n, p) for n, p in named_block_params if n.endswith(".attn.proj.weight")]
    mlp_fc_named   = [(n, p) for n, p in named_block_params if n.endswith(".mlp.fc.weight")]
    mlp_proj_named = [(n, p) for n, p in named_block_params if n.endswith(".mlp.proj.weight")]
    param_name_by_id = {id(p): n for n, p in named_block_params}

    optimizer2 = DistributedMuonLKindwise(
        [dict(params=[p for _, p in qkv_named],      lr=MYOPT_QKV_LR,       weight_decay=MYOPT_QKV_WD,       label="q/k/v"),
         dict(params=[p for _, p in attn_out_named],  lr=MYOPT_ATTN_PROJ_LR, weight_decay=MYOPT_ATTN_PROJ_WD, label="attn.proj"),
         dict(params=[p for _, p in mlp_fc_named],    lr=MYOPT_MLP_FC_LR,    weight_decay=MYOPT_MLP_FC_WD,    label="mlp.fc"),
         dict(params=[p for _, p in mlp_proj_named],  lr=MYOPT_MLP_PROJ_LR,  weight_decay=MYOPT_MLP_PROJ_WD,  label="mlp.proj")],
        param_names=param_name_by_id,
        momentum=MYOPT_MOMENTUM,
        ns_steps=MYOPT_NS_STEPS,
        nesterov=MYOPT_NESTEROV,
        rho=OPT_RHO,
        interval=MYOPT_INTERVAL,
        scheme=OPT_MODE,
        scale_ema_beta=MYOPT_SCALE_EMA_BETA,
        normuon_beta2=MYOPT_NORMUON_BETA2,
        orscale_r_min=MYOPT_ORSCALE_R_MIN,
        orscale_r_max=MYOPT_ORSCALE_R_MAX,
        orscale_general_shape_scale=MYOPT_ORSCALE_GENERAL_SHAPE_SCALE,
        uw_floor_mode=MYOPT_UW_FLOOR_MODE,
        uw_floor_c=MYOPT_UW_FLOOR_C,
        uw_floor_eps=MYOPT_UW_FLOOR_EPS,
    )
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    ########################################
    #          Logging setup               #
    ########################################

    diag_file = None
    if dist.get_rank() == 0 and (
        MYOPT_SAVE_LOG
        or MYOPT_SAVE_DIAG
        or MYOPT_SAVE_U_OVER_W_JSON
        or MYOPT_SAVE_VAL_LOSS_JSONL
        or MYOPT_SAVE_WEIGHT_NORMS
    ):
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        log_stem_override = os.getenv("MYOPT_LOG_STEM", "").strip()
        default_log_stem = log_stem_override or f"seed{FIXED_SEED}_{MYOPT_SCHEME}"
        if MYOPT_SAVE_LOG or MYOPT_SAVE_DIAG:
            logfile = RESULT_DIR / f"{default_log_stem}.txt"
        if MYOPT_SAVE_DIAG:
            diag_json_override = os.getenv("MYOPT_DIAG_JSON", "").strip()
            if diag_json_override:
                diag_file = Path(diag_json_override)
                if not diag_file.is_absolute():
                    diag_file = RESULT_DIR / diag_file
            else:
                diag_file = RESULT_DIR / f"{default_log_stem}_diag.json"
        if MYOPT_SAVE_U_OVER_W_JSON:
            u_over_w_override = os.getenv("MYOPT_U_OVER_W_JSON", "").strip()
            if u_over_w_override:
                u_over_w_file = Path(u_over_w_override)
                if not u_over_w_file.is_absolute():
                    u_over_w_file = RESULT_DIR / u_over_w_file
            else:
                u_over_w_file = RESULT_DIR / f"{default_log_stem}_u_over_w.json"
            u_over_w_file.parent.mkdir(parents=True, exist_ok=True)
            u_over_w_file.write_text("")
        if MYOPT_SAVE_VAL_LOSS_JSONL:
            val_loss_file = RESULT_DIR / f"{default_log_stem}_val_loss.jsonl"
            val_loss_file.write_text("")
        if MYOPT_SAVE_WEIGHT_NORMS:
            weight_norm_file = RESULT_DIR / f"{default_log_stem}_weight_norms.jsonl"
            weight_norm_file.write_text("")
        if logfile is not None:
            logfile.write_text("")
            print(f"logfile: {logfile}")
        if diag_file is not None:
            print(f"diag_file: {diag_file}")
        if u_over_w_file is not None:
            print(f"u_over_w_file: {u_over_w_file}")
        if val_loss_file is not None:
            print(f"val_loss_file: {val_loss_file}")
        if weight_norm_file is not None:
            print(f"weight_norm_file: {weight_norm_file}")
    elif dist.get_rank() == 0:
        print("save_log:false save_diag:false")

    if logfile is not None:
        print0(_source_snapshot_for_log())
    print0("=" * 100)
    print0(f"scheme:{MYOPT_SCHEME} opt_mode:{OPT_MODE} rho:{OPT_RHO}"
           + f" ns_steps:{MYOPT_NS_STEPS} momentum:{MYOPT_MOMENTUM}"
           + f" nesterov:{MYOPT_NESTEROV}"
           + f" scale_ema_beta:{MYOPT_SCALE_EMA_BETA} interval:{MYOPT_INTERVAL}"
           + f" normuon_beta2:{MYOPT_NORMUON_BETA2} init_mode:{init_mode}",
           console=True)
    print0(
        f"matrix_lr_base:{MYOPT_LR} matrix_weight_decay_base:{MYOPT_WD}"
        " matrix_weight_decay_mode:decoupled",
        console=True,
    )
    for _group in optimizer2.param_groups:
        print0(
            f"matrix_group:{_group.get('label', '')}"
            f" initial_lr:{_group['initial_lr']}"
            f" weight_decay:{_group['weight_decay']}",
            console=True,
        )
    print0(
        f"uw_floor_mode:{MYOPT_UW_FLOOR_MODE}"
        f" uw_floor_c:{MYOPT_UW_FLOOR_C}"
        f" uw_floor_eps:{MYOPT_UW_FLOOR_EPS}",
        console=True,
    )
    print0(
        f"schedule_cooldown_frac:{MYOPT_COOLDOWN_FRAC}",
        console=True,
    )
    print0(
        f"validation_frequency early:{EARLY_VAL_STEP_FREQ}"
        f" mid:{MID_VAL_STEP_FREQ}@{MID_VAL_START_STEP}"
        f" late:{LATE_VAL_STEP_FREQ}@{LATE_VAL_START_STEP}",
        console=True,
    )
    print0(f"seed:{FIXED_SEED} train_steps:{train_steps} schedule_steps:{schedule_steps} diag_freq:{DIAG_FREQ}", console=True)
    print0(
        f"batch_tokens:{batch_size} train_microbatches:{train_microbatches}"
        f" train_microbatch_sequences:{train_microbatch_sequences} val_tokens:{val_tokens}",
        console=True,
    )
    print0(
        f"save_log:{bool(MYOPT_SAVE_LOG)} save_diag:{bool(MYOPT_SAVE_DIAG)}"
        f" save_u_over_w_json:{bool(MYOPT_SAVE_U_OVER_W_JSON)}"
        f" u_over_w_freq:{MYOPT_U_OVER_W_FREQ}"
        f" save_val_loss_jsonl:{bool(MYOPT_SAVE_VAL_LOSS_JSONL)}",
        console=True,
    )
    if MYOPT_SAVE_DIAG:
        print0(f"diag_mode:{MYOPT_DIAG_MODE}", console=True)
        print0(
            f"diag_batch_size:{diag_mbs} diag_tokens:{diag_mbs*sequence_length}"
            f" diag_grad_mbs:{diag_grad_mbs} local_quadratic_data_source:{LOCAL_QUADRATIC_DATA_SOURCE}",
            console=True,
        )
        if MYOPT_DIAG_MODE == "local_quadratic" and LOCAL_QUADRATIC_DATA_SOURCE == "train_batch":
            print0(
                "local_quadratic_train_batch_scope:full_optimizer_step"
                f" local_quadratic_train_batch_sequences:{train_microbatches * train_microbatch_sequences}"
                f" local_quadratic_train_batch_tokens:{batch_size}",
                console=True,
            )
        print0(
            f"local_quadratic_hvp_mbs:{LOCAL_QUADRATIC_HVP_MBS}"
            f" local_quadratic_max_blocks:{LOCAL_QUADRATIC_MAX_BLOCKS}"
            f" local_quadratic_eps_w:{LOCAL_QUADRATIC_EPS_W}"
            f" local_quadratic_eps_den:{LOCAL_QUADRATIC_EPS_DEN}"
            f" local_quadratic_curvature:{LOCAL_QUADRATIC_CURVATURE_METHOD}"
            f" local_quadratic_cross_curvature:{LOCAL_QUADRATIC_CROSS_CURVATURE}"
            f" local_quadratic_save_cross_curvature_matrix:{LOCAL_QUADRATIC_SAVE_CROSS_CURVATURE_MATRIX}"
            f" local_quadratic_donated_buffer_disabled:{LOCAL_QUADRATIC_DONATED_BUFFER_DISABLED}",
            console=True,
        )
    print0("step:S val_loss is after S optimizer updates", console=True)


    def _write_json_atomic(path: Path, payload) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(payload, f)
        tmp.replace(path)


    def _state_float_or_none(state: dict, key: str) -> float | None:
        value = state.get(key)
        if value is None:
            return None
        return _safe_float_or_none(value)


    def _append_jsonl(path: Path, payload: dict) -> None:
        with path.open("a") as f:
            json.dump(payload, f)
            f.write("\n")

    @torch.no_grad()
    def _append_weight_norm_record(step_index: int) -> None:
        if not (
            dist.get_rank() == 0
            and MYOPT_SAVE_WEIGHT_NORMS
            and weight_norm_file is not None
        ):
            return
        blocks = []
        for name, p in named_block_params:
            suffix = name.split(".", 2)[-1].removesuffix(".weight").replace(".", "_")
            blocks.append({
                "name": name,
                "type": suffix,
                "w_norm": float(torch.linalg.vector_norm(p.detach().float()).item()),
            })
        _append_jsonl(weight_norm_file, {"step": step_index, "blocks": blocks})


    U_OVER_W_KIND_KEYS = {
        "attn_q": "q",
        "attn_k": "k",
        "attn_v": "v",
        "attn_proj": "attnproj",
        "mlp_fc": "mlpfc",
        "mlp_proj": "mlpproj",
    }


    @torch.no_grad()
    def _append_u_over_w_record(
        step_index: int,
        train_loss: float | None,
        lr_current: float,
        event: str,
    ) -> None:
        if not (
            dist.get_rank() == 0
            and MYOPT_SAVE_U_OVER_W_JSON
            and u_over_w_file is not None
        ):
            return

        raw_blocks: dict[str, dict[str, float | None]] = {}
        effective_blocks: dict[str, dict[str, float | None]] = {}
        scale_blocks: dict[str, dict[str, float | None]] = {}
        target_blocks: dict[str, dict[str, float | None]] = {}
        param_blocks: list[dict] = []

        for group in optimizer2.param_groups:
            label = str(group.get("label", ""))
            for p in group["params"]:
                if p.ndim < 2:
                    continue
                state = optimizer2.state.get(p, {})
                if "uw_floor_raw_ratio" not in state:
                    continue

                name = optimizer2._param_name_by_id.get(id(p), f"id_{id(p)}")
                kind = optimizer2._matrix_kind(p)
                plot_key = U_OVER_W_KIND_KEYS.get(kind, kind)
                block_idx = optimizer2._param_block_by_id.get(id(p), -1)
                if block_idx is None or int(block_idx) < 0:
                    continue

                block_key = str(int(block_idx))
                raw_ratio = _state_float_or_none(state, "uw_floor_raw_ratio")
                effective_ratio = _state_float_or_none(state, "uw_floor_effective_ratio")
                scale = _state_float_or_none(state, "uw_floor_scale")
                target = _state_float_or_none(state, "uw_floor_target")

                raw_blocks.setdefault(block_key, {})[plot_key] = raw_ratio
                effective_blocks.setdefault(block_key, {})[plot_key] = effective_ratio
                scale_blocks.setdefault(block_key, {})[plot_key] = scale
                target_blocks.setdefault(block_key, {})[plot_key] = target
                param_blocks.append({
                    "param_name": name,
                    "group_label": label,
                    "type": kind,
                    "plot_key": plot_key,
                    "block_idx": int(block_idx),
                    "raw_ratio": raw_ratio,
                    "effective_ratio": effective_ratio,
                    "target": target,
                    "scale": scale,
                    "applied": bool(state.get("uw_floor_applied", False)),
                    "w_norm_pre_update": _state_float_or_none(
                        state, "uw_floor_w_norm_pre_update"
                    ),
                    "update_norm_pre_lr": _state_float_or_none(
                        state, "uw_floor_update_norm_pre_lr"
                    ),
                })

        param_blocks.sort(key=lambda b: (b["block_idx"], b["type"], b["param_name"]))
        u_over_w_records.append({
            "step": int(step_index),
            "event": event,
            "scheme": MYOPT_SCHEME,
            "opt_mode": OPT_MODE,
            "seed": int(FIXED_SEED),
            "train_loss": _safe_float_or_none(train_loss),
            "matrix_lr": _safe_float(lr_current),
            "blocks": effective_blocks,
            "raw_blocks": raw_blocks,
            "scale_blocks": scale_blocks,
            "target_blocks": target_blocks,
            "param_blocks": param_blocks,
        })

        _write_json_atomic(u_over_w_file, {
            "metadata": {
                "description": (
                    "Per-block ||U||_F / ||W||_F recorder for Figure 1b. "
                    "blocks stores post-floor effective ratios; raw_blocks stores "
                    "pre-floor ratios."
                ),
                "scheme": MYOPT_SCHEME,
                "opt_mode": OPT_MODE,
                "seed": int(FIXED_SEED),
                "train_steps": int(train_steps),
                "schedule_steps": int(schedule_steps),
                "record_interval": int(MYOPT_U_OVER_W_FREQ),
                "ratio": "||U||_F / ||W||_F before learning rate",
                "blocks_field": "post-floor effective ratio",
                "raw_blocks_field": "pre-floor raw ratio",
                "matrix_kinds": [
                    "q",
                    "k",
                    "v",
                    "attnproj",
                    "mlpfc",
                    "mlpproj",
                ],
                "uw_floor_mode": MYOPT_UW_FLOOR_MODE,
                "uw_floor_c": _safe_float(MYOPT_UW_FLOOR_C),
                "target_uw": (
                    _safe_float(MYOPT_UW_FLOOR_C)
                    if MYOPT_UW_FLOOR_MODE == "fixed" else
                    None
                ),
                "matrix_lrs": {
                    str(group.get("label", "")): _safe_float(
                        group.get("initial_lr", group["lr"])
                    )
                    for group in optimizer2.param_groups
                },
                "matrix_weight_decays": {
                    str(group.get("label", "")): _safe_float(group["weight_decay"])
                    for group in optimizer2.param_groups
                },
            },
            "records": u_over_w_records,
        })


    def _append_val_loss_record(
        step_index: int,
        val_loss_per_tok: float,
        train_time: float,
        step_avg_ms: float,
    ) -> None:
        if not (
            dist.get_rank() == 0
            and MYOPT_SAVE_VAL_LOSS_JSONL
            and val_loss_file is not None
        ):
            return
        _append_jsonl(val_loss_file, {
            "step": int(step_index),
            "scheme": MYOPT_SCHEME,
            "opt_mode": OPT_MODE,
            "seed": int(FIXED_SEED),
            "val_loss": _safe_float(val_loss_per_tok),
            "train_time": _safe_float(train_time),
            "step_avg_ms": _safe_float(step_avg_ms),
            "matrix_lrs": {
                str(group.get("label", "")): _safe_float(group.get("initial_lr", group["lr"]))
                for group in optimizer2.param_groups
            },
            "matrix_weight_decays": {
                str(group.get("label", "")): _safe_float(group["weight_decay"])
                for group in optimizer2.param_groups
            },
            "uw_floor_mode": MYOPT_UW_FLOOR_MODE,
            "uw_floor_target": (
                _safe_float(MYOPT_UW_FLOOR_C)
                if MYOPT_UW_FLOOR_MODE == "fixed" else
                None
            ),
        })


    for p in model.parameters():
        dist.broadcast(p.detach(), 0)

    ########################################
    #         LR schedule                  #
    ########################################

    def schedule_multiplier(step, cooldown_frac=None):
        cooldown_frac = MYOPT_COOLDOWN_FRAC if cooldown_frac is None else float(cooldown_frac)
        progress = step / schedule_steps
        return 1.0 if progress < 1 - cooldown_frac else (1 - progress) / cooldown_frac

    def set_hparams(step, cooldown_frac=None):
        eta = schedule_multiplier(step, cooldown_frac)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta

    ########################################
    #         Training loop                #
    ########################################

    diag_records: list[dict] = []
    u_over_w_records: list[dict] = []
    diag_counter = 0
    train_loader = distributed_data_generator(str(DATA_DIR / "fineweb_train_*.bin"), batch_size)
    training_time = 0.0
    last_val_step = 0
    dist.barrier()
    t0 = time.perf_counter()
    if MYOPT_SAVE_WEIGHT_NORMS:
        _append_weight_norm_record(0)

    for step in range(train_steps + 1):

        # -------- validation --------
        if step >= LATE_VAL_START_STEP:
            val_step_freq = LATE_VAL_STEP_FREQ
        elif step >= MID_VAL_START_STEP:
            val_step_freq = MID_VAL_STEP_FREQ
        else:
            val_step_freq = EARLY_VAL_STEP_FREQ
        if step == train_steps or step % val_step_freq == 0:
            dist.barrier()
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            train_model.eval()
            val_loss = torch.zeros((), device=device)
            with torch.no_grad():
                for i in range(len(val_inputs) // train_microbatch_sequences):
                    start = i * train_microbatch_sequences
                    end = (i + 1) * train_microbatch_sequences
                    val_loss += train_model(val_inputs[start:end], val_targets[start:end])
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss_per_tok = float(val_loss) / val_tokens
            print0(f"step:{step}/{train_steps} val_loss:{val_loss_per_tok:.5f}"
                   f" train_time:{training_time:.3f}s step_avg:{1000*step_avg:.2f}ms", console=True)
            _append_val_loss_record(
                step,
                val_loss_per_tok,
                training_time,
                1000 * step_avg,
            )
            model.train()
            train_model.train()
            dist.barrier()
            t0 = time.perf_counter()

        if step == train_steps:
            break

        # -------- training step --------
        inputs, targets = next(train_loader)
        assert len(inputs) % train_microbatches == 0
        current_microbatch_sequences = len(inputs) // train_microbatches
        assert current_microbatch_sequences == train_microbatch_sequences
        step_train_loss = 0.0
        for i in range(train_microbatches):
            start = i * current_microbatch_sequences
            end = (i + 1) * current_microbatch_sequences
            loss = train_model(inputs[start:end], targets[start:end])
            step_train_loss += loss.item()
            loss.backward()
        step_train_loss /= (len(inputs) * sequence_length)  # per token

        for name_p, p in model.named_parameters():
            assert p.grad is not None, name_p
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        current_step_index = step + 1
        set_hparams(step)
        current_lr = optimizer2.param_groups[0]["lr"]

        # -------- diagnostics every DIAG_FREQ steps (all ranks participate) --------
        # Measure before optimizer.step so G_diag(W_t) and the Muon direction V_t
        # are defined at the same checkpoint.
        if MYOPT_SAVE_DIAG and (step + 1) % DIAG_FREQ == 0:
            diag_counter += 1
            if LOCAL_QUADRATIC_DATA_SOURCE == "train_batch":
                local_quadratic_inputs = inputs
                local_quadratic_targets = targets
                local_quadratic_data_source = "current_training_batch"
            else:
                local_quadratic_inputs = diag_inputs
                local_quadratic_targets = diag_targets
                local_quadratic_data_source = "fixed_validation_batch"
            record = compute_local_quadratic_diagnostics(
                model=model,
                optimizer2=optimizer2,
                diag_inputs=local_quadratic_inputs,
                diag_targets=local_quadratic_targets,
                diag_data_source=local_quadratic_data_source,
                train_batch_tokens=batch_size,
                diag_grad_microbatch_size=diag_grad_mbs,
                hvp_microbatch_size=LOCAL_QUADRATIC_HVP_MBS,
                lr_current=current_lr,
                diag_index=diag_counter,
                step=step + 1,
                train_loss=step_train_loss,
                scheme=MYOPT_SCHEME,
                seed=FIXED_SEED,
                ns_steps=MYOPT_NS_STEPS,
                max_blocks=LOCAL_QUADRATIC_MAX_BLOCKS,
                curvature_method=LOCAL_QUADRATIC_CURVATURE_METHOD,
                fd_probes=0,
                fd_rel_eps=0.0,
                fd_abs_eps=0.0,
                eps_w=LOCAL_QUADRATIC_EPS_W,
                eps_den=LOCAL_QUADRATIC_EPS_DEN,
                exact_hvp_eager_diagnostic_model=LOCAL_QUADRATIC_EXACT_HVP_EAGER_DIAG_MODEL,
                measure_cross_curvature=LOCAL_QUADRATIC_CROSS_CURVATURE,
                save_cross_curvature_matrix=LOCAL_QUADRATIC_SAVE_CROSS_CURVATURE_MATRIX,
            )
            if dist.get_rank() == 0 and MYOPT_SAVE_DIAG:
                diag_records.append(record)
                _write_json_atomic(diag_file, diag_records)

        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        if MYOPT_SAVE_WEIGHT_NORMS and current_step_index % MYOPT_WEIGHT_NORM_FREQ == 0:
            _append_weight_norm_record(current_step_index)
        if MYOPT_SAVE_U_OVER_W_JSON and current_step_index % MYOPT_U_OVER_W_FREQ == 0:
            _append_u_over_w_record(
                current_step_index,
                step_train_loss,
                current_lr,
                event="post_update",
            )
        approx_time = training_time + (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_loss:{step_train_loss:.5f}"
               f" train_time:{approx_time:.3f}s step_avg:{1000*approx_time/(step+1):.2f}ms",
               console=True, log=False)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
