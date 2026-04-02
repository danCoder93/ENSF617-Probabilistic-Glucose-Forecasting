from __future__ import annotations

# AI-assisted maintenance note:
# This module owns the narrow layer that turns already-resolved runtime policy
# into low-level backend actions.
#
# Why this file exists separately from `profiles.py`:
# - `profiles.py` decides *what* the default policy should be for one machine
# - this file decides *how* that policy is applied to Torch/runtime surfaces
# - keeping those concerns apart makes it easier to reason about "selection"
#   versus "execution"
#
# Responsibility boundary:
# - translate `TrainConfig` tuning fields into environment variables or Torch
#   backend calls
# - record what was actually applied versus skipped so runs remain observable
# - keep backend-specific behavior centralized rather than scattered across the
#   CLI, trainer wrapper, and diagnostics
#
# What does *not* live here:
# - detecting the host environment
# - deciding which profile to use
# - validating whether a chosen configuration is sensible
#
# Important disclaimer:
# these knobs are best-effort throughput and compatibility aids, not guarantees
# of faster training. Backend behavior can vary across Torch versions, driver
# stacks, and model architectures, so this module is intentionally defensive
# and records skipped actions rather than failing eagerly.

from dataclasses import dataclass
import os
from typing import Any

from config.runtime import TrainConfig
from environment.types import RuntimeEnvironment


# ============================================================================
# Lightweight Tuning Report
# ============================================================================
# The rest of the repository should be able to answer:
# - which backend knobs were actually set?
# - which ones were requested but unavailable here?
# without importing Torch backend objects directly.
@dataclass(frozen=True)
class RuntimeTuningReport:
    """
    Small summary of runtime/backend tuning applied before training.

    Context:
    this keeps the low-level tuning layer observable without exposing raw torch
    backend objects to the rest of the codebase.
    """

    applied: dict[str, Any]
    skipped: dict[str, str]


# ============================================================================
# Environment Variable Overrides
# ============================================================================
# Some backend controls are consumed only at import/runtime initialization time.
# Apple Silicon's MPS allocator behavior is the main example here, so those
# settings must be exported through `os.environ` before later Torch code relies
# on them.
def apply_runtime_environment_overrides(
    *,
    train_config: TrainConfig,
) -> RuntimeTuningReport:
    """
    Apply environment-variable-based runtime tuning before torch is imported.
    """

    applied: dict[str, Any] = {}
    skipped: dict[str, str] = {}

    if train_config.mps_high_watermark_ratio is not None:
        # PyTorch reads this value from the environment rather than exposing a
        # stable Python setter, so we treat it as an import-time/runtime
        # bootstrap control.
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(
            train_config.mps_high_watermark_ratio
        )
        applied["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = (
            train_config.mps_high_watermark_ratio
        )
    if train_config.mps_low_watermark_ratio is not None:
        # The low watermark complements the high watermark by telling the MPS
        # allocator where it should begin releasing pressure more aggressively.
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(
            train_config.mps_low_watermark_ratio
        )
        applied["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = train_config.mps_low_watermark_ratio
    if train_config.enable_mps_fallback is not None:
        # Fallback lets unsupported MPS ops drop to CPU rather than crashing.
        # That can help compatibility, but it may also hide performance cliffs,
        # which is why we still surface the choice through the report.
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
            "1" if train_config.enable_mps_fallback else "0"
        )
        applied["PYTORCH_ENABLE_MPS_FALLBACK"] = train_config.enable_mps_fallback

    return RuntimeTuningReport(applied=applied, skipped=skipped)


# ============================================================================
# Torch Backend Knobs
# ============================================================================
# These controls are applied after Torch is importable and the runtime has
# already been detected. Most are cheap global setters that affect math
# kernels, threading, or backend heuristics for the rest of the process.
def apply_runtime_tuning(
    *,
    environment: RuntimeEnvironment,
    train_config: TrainConfig,
) -> RuntimeTuningReport:
    """
    Apply host/backend-specific torch runtime tuning before training starts.
    """

    try:
        import torch
    except ImportError:
        return RuntimeTuningReport(
            applied={},
            skipped={"torch": "PyTorch is not installed in the active environment."},
        )

    applied: dict[str, Any] = {}
    skipped: dict[str, str] = {}

    set_matmul_precision = getattr(torch, "set_float32_matmul_precision", None)
    if train_config.matmul_precision is not None and callable(set_matmul_precision):
        # This affects float32 matmul kernel selection. It is most interesting
        # for CPU and CUDA paths where users want a broad "lean toward
        # throughput" or "lean toward highest precision" control without
        # rewriting model code.
        set_matmul_precision(train_config.matmul_precision)
        applied["matmul_precision"] = train_config.matmul_precision

    cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
    cudnn_backend = getattr(getattr(torch, "backends", None), "cudnn", None)
    if train_config.allow_tf32 is not None:
        # TF32 is only meaningful on CUDA devices that expose those fast math
        # paths. We record a skip on non-CUDA hosts so the caller can see that
        # the request was understood but not relevant here.
        if not environment.cuda_available:
            skipped["allow_tf32"] = "TF32 tuning is only relevant when CUDA is available."
        if environment.cuda_available and cuda_backend is not None:
            matmul_backend = getattr(cuda_backend, "matmul", None)
            if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
                matmul_backend.allow_tf32 = train_config.allow_tf32
                applied["cuda.matmul.allow_tf32"] = train_config.allow_tf32
        if (
            environment.cuda_available
            and cudnn_backend is not None
            and hasattr(cudnn_backend, "allow_tf32")
        ):
            cudnn_backend.allow_tf32 = train_config.allow_tf32
            applied["cudnn.allow_tf32"] = train_config.allow_tf32

    if train_config.cudnn_benchmark is not None:
        # cuDNN benchmark can improve throughput when batch/input shapes are
        # stable, but it conflicts with the spirit of deterministic execution.
        # The deterministic guard keeps the effective behavior explicit.
        if cudnn_backend is not None and hasattr(cudnn_backend, "benchmark"):
            cudnn_backend.benchmark = (
                train_config.cudnn_benchmark and not train_config.deterministic
            )
            applied["cudnn.benchmark"] = cudnn_backend.benchmark
        else:
            skipped["cudnn_benchmark"] = "cuDNN backend controls are not available."

    if train_config.intraop_threads is not None:
        try:
            # Intra-op threads control how much parallelism one Torch operator
            # may use internally. This is especially relevant on CPU-heavy runs.
            torch.set_num_threads(train_config.intraop_threads)
            applied["intraop_threads"] = train_config.intraop_threads
        except RuntimeError as exc:
            skipped["intraop_threads"] = str(exc)

    if train_config.interop_threads is not None and hasattr(
        torch, "set_num_interop_threads"
    ):
        try:
            # Inter-op threads control how many operators may execute in
            # parallel. Keeping this separate from intra-op threads lets local
            # CPU and Apple Silicon runs avoid oversubscribing the host.
            torch.set_num_interop_threads(train_config.interop_threads)
            applied["interop_threads"] = train_config.interop_threads
        except RuntimeError as exc:
            skipped["interop_threads"] = str(exc)

    return RuntimeTuningReport(applied=applied, skipped=skipped)


def synchronize_runtime_device(
    *,
    environment: RuntimeEnvironment,
) -> None:
    """
    Best-effort device synchronization for timing-sensitive runtime flows.

    Context:
    this is intentionally not part of model code. Synchronization is a runtime
    orchestration concern used mainly for benchmark boundaries so async CUDA
    kernels do not distort wall-clock measurements.
    """

    try:
        import torch
    except ImportError:
        return

    if environment.cuda_available and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


# ============================================================================
# Optional Compilation
# ============================================================================
# Compilation is treated as a thin wrapper around `torch.compile(...)`, not a
# guaranteed optimization stage. The trainer wrapper owns fallback behavior when
# compilation itself raises, while this function focuses on:
# - deciding whether compilation should happen at all
# - choosing a backend-aware default mode when the caller enabled compile but
#   did not specify an exact mode
def maybe_compile_model(
    model: Any,
    *,
    train_config: TrainConfig,
    environment: RuntimeEnvironment,
) -> Any:
    """
    Compile the model when explicitly requested and supported by this torch build.
    """

    if not train_config.compile_model:
        return model

    try:
        import torch
    except ImportError:
        return model

    compile_fn = getattr(torch, "compile", None)
    if not callable(compile_fn):
        return model

    compile_kwargs: dict[str, Any] = {}
    compile_mode = train_config.compile_mode
    if compile_mode is None:
        # CUDA generally benefits from the standard compile policy, while CPU
        # commonly prefers the lighter `reduce-overhead` mode for smaller local
        # experiments. MPS stays conservative unless the caller opts in.
        if environment.cuda_available:
            compile_mode = "default"
        elif environment.accelerator_type == "cpu" or not environment.accelerator_available:
            compile_mode = "reduce-overhead"
    if compile_mode is not None:
        compile_kwargs["mode"] = compile_mode
    if train_config.compile_fullgraph:
        # Full-graph compilation is more restrictive but can be useful for
        # advanced experimentation when the model is known to be compiler-
        # friendly.
        compile_kwargs["fullgraph"] = True

    return compile_fn(model, **compile_kwargs)
