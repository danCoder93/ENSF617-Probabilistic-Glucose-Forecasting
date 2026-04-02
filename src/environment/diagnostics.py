from __future__ import annotations

# AI-assisted maintenance note:
# This module owns runtime validation and failure explanation for the
# environment package. It answers:
# - does the selected runtime configuration look valid here?
# - if something failed, does it look environment-related?
#
# Responsibility boundary:
# - validate resolved runtime choices against the detected environment
# - convert environment-looking failures into friendlier diagnostic hints
# - format diagnostics for CLI/notebook/reporting surfaces
#
# What does *not* live here:
# - backend detection itself
# - profile defaulting / config mutation
#
# Important disclaimer:
# these diagnostics are heuristic, not omniscient. They are designed to catch
# common environment mismatches and make failures easier to understand, but
# they do not replace the underlying stack trace or backend documentation.

from typing import Sequence

from config.data import DataConfig
from config.observability import ObservabilityConfig
from config.runtime import TrainConfig
from environment.types import RuntimeDiagnostic, RuntimeEnvironment


# ============================================================================
# Validation Helpers
# ============================================================================
# The diagnostics layer stays lightweight by operating on already-normalized
# config objects and environment metadata. These helpers keep the public
# functions easier to scan without creating a deeper abstraction tree.
def _requested_device_count(devices: str | int | list[int]) -> int | None:
    if devices == "auto":
        return None
    if isinstance(devices, int):
        return devices
    if isinstance(devices, list):
        return len(devices)
    return None


def collect_runtime_diagnostics(
    *,
    requested_profile: str,
    resolved_profile: str,
    environment: RuntimeEnvironment,
    train_config: TrainConfig,
    data_config: DataConfig,
    observability_config: ObservabilityConfig,
) -> tuple[RuntimeDiagnostic, ...]:
    """
    Validate the selected runtime setup against the detected environment.

    Context:
    this is the preflight validation path used before training starts. It
    favors actionable, environment-shaped messages over deep backend-specific
    wording so users can quickly see what likely needs to change.
    """

    diagnostics: list[RuntimeDiagnostic] = []

    # Dependency presence is checked first because many later diagnostics only
    # make sense once the core training stack exists.
    #
    # In other words, these are "can the repository even launch the intended
    # runtime stack?" checks, not backend-policy checks yet.
    if not environment.torch_available:
        diagnostics.append(
            RuntimeDiagnostic(
                severity="error",
                code="missing_torch",
                message="PyTorch is not installed in the active environment.",
                suggestion="Install `torch` before running training or diagnostics.",
            )
        )
    if not environment.pytorch_lightning_available:
        diagnostics.append(
            RuntimeDiagnostic(
                severity="error",
                code="missing_lightning",
                message="PyTorch Lightning is not installed in the active environment.",
                suggestion=(
                    "Install `pytorch-lightning` before launching the training workflow."
                ),
            )
        )

    accelerator = train_config.accelerator
    requested_device_count = _requested_device_count(train_config.devices)

    # Backend compatibility checks answer the question:
    # "does the requested runtime policy make sense on this machine?"
    #
    # These are treated as hard errors when the mismatch is strong enough that
    # the run is very likely to fail or behave incorrectly.
    if accelerator == "gpu" and not environment.cuda_available:
        diagnostics.append(
            RuntimeDiagnostic(
                severity="error",
                code="cuda_unavailable",
                message="GPU acceleration was requested, but CUDA is not available.",
                suggestion="Use a CPU/MPS profile or install a CUDA-enabled PyTorch build.",
            )
        )
    if accelerator == "mps" and not environment.mps_available:
        diagnostics.append(
            RuntimeDiagnostic(
                severity="error",
                code="mps_unavailable",
                message="MPS acceleration was requested, but Apple Silicon MPS is not available.",
                suggestion="Use `local-cpu` or install an MPS-capable PyTorch build on macOS.",
            )
        )
    if (
        accelerator == "gpu"
        and requested_device_count is not None
        and requested_device_count > environment.cuda_device_count
    ):
        diagnostics.append(
            RuntimeDiagnostic(
                severity="error",
                code="gpu_count_mismatch",
                message=(
                    "The requested GPU device count exceeds the number of visible CUDA devices."
                ),
                suggestion=(
                    "Reduce `--devices`, adjust `CUDA_VISIBLE_DEVICES`, or request more GPUs."
                ),
            )
        )
    if (
        accelerator == "cpu"
        and str(train_config.precision).endswith("-mixed")
        and not (
            str(train_config.precision) == "bf16-mixed"
            and environment.cpu_supports_bf16
        )
    ):
        # Generic CPU mixed precision is not something we want to "maybe let
        # slide." The repository only blesses the BF16 CPU path when the host
        # reports support for it; otherwise plain FP32 is the clear baseline.
        diagnostics.append(
            RuntimeDiagnostic(
                severity="error",
                code="cpu_mixed_precision",
                message="The requested mixed-precision mode is not supported for this CPU run.",
                suggestion="Use precision `32`, or `bf16-mixed` on CPUs that support BF16.",
            )
        )
    if (
        accelerator == "cpu"
        and str(train_config.precision) == "bf16-mixed"
        and not environment.cpu_supports_bf16
    ):
        diagnostics.append(
            RuntimeDiagnostic(
                severity="error",
                code="cpu_bf16_unavailable",
                message="BF16 mixed precision was requested, but the active CPU does not report BF16 support.",
                suggestion="Use precision `32` on this CPU.",
            )
        )
    if accelerator == "mps" and str(train_config.precision).endswith("-mixed"):
        # MPS stays intentionally conservative here. Even if some mixed-
        # precision combinations can work in certain versions, the repo's
        # profile policy is to keep Apple Silicon defaults simpler and easier
        # to reason about.
        diagnostics.append(
            RuntimeDiagnostic(
                severity="error",
                code="mps_mixed_precision",
                message="Mixed precision was requested for MPS, which uses conservative defaults here.",
                suggestion="Use precision `32` for Apple Silicon runs.",
            )
        )
    if (
        accelerator == "gpu"
        and str(train_config.precision) == "bf16-mixed"
        and not environment.cuda_supports_bf16
    ):
        diagnostics.append(
            RuntimeDiagnostic(
                severity="error",
                code="cuda_bf16_unavailable",
                message=(
                    "BF16 mixed precision was requested, but the active CUDA setup "
                    "does not report BF16 support."
                ),
                suggestion="Use precision `16-mixed` or `32` for this GPU.",
            )
        )

    # Environment-shape warnings answer the slightly different question:
    # "even if this might run, does the selected profile match the current
    # environment well?"
    #
    # These are warnings rather than errors when the situation is "plausibly
    # runnable but probably the wrong shape of defaults."
    if resolved_profile == "apple-silicon" and not environment.is_apple_silicon:
        diagnostics.append(
            RuntimeDiagnostic(
                severity="error",
                code="apple_profile_without_apple_silicon",
                message="The Apple Silicon profile was selected on a non-Apple-Silicon host.",
                suggestion="Use `local-cpu`, `local-cuda`, or a Slurm profile instead.",
            )
        )
    if resolved_profile.startswith("slurm-") and not environment.is_slurm:
        diagnostics.append(
            RuntimeDiagnostic(
                severity="warning",
                code="slurm_profile_without_slurm",
                message="A Slurm profile was selected, but no Slurm allocation was detected.",
                suggestion="Use a local profile or launch the job through Slurm.",
            )
        )
    if resolved_profile.startswith("colab-") and not environment.is_colab:
        diagnostics.append(
            RuntimeDiagnostic(
                severity="warning",
                code="colab_profile_without_colab",
                message="A Colab profile was selected, but the runtime does not look like Google Colab.",
                suggestion="Use a local profile unless this notebook is actually running in Colab.",
            )
        )
    if environment.is_colab and data_config.num_workers > 2:
        diagnostics.append(
            RuntimeDiagnostic(
                severity="warning",
                code="colab_worker_count_high",
                message="Colab runs often become unstable with high DataLoader worker counts.",
                suggestion="Keep `num_workers` at 0-2 in Colab unless you have verified stability.",
            )
        )
    if data_config.persistent_workers and data_config.num_workers == 0:
        # The DataLoader path already guards this at construction time, so this
        # warning is less about preventing a crash and more about surfacing an
        # internally contradictory config.
        diagnostics.append(
            RuntimeDiagnostic(
                severity="warning",
                code="persistent_workers_without_workers",
                message="Persistent workers were enabled with `num_workers=0`.",
                suggestion="Disable persistent workers or raise `num_workers` above zero.",
            )
        )
    if data_config.prefetch_factor is not None and data_config.num_workers == 0:
        # `prefetch_factor` silently does nothing without worker processes. A
        # warning keeps the run summary honest about which knobs are actually
        # meaningful for the selected loader shape.
        diagnostics.append(
            RuntimeDiagnostic(
                severity="warning",
                code="prefetch_without_workers",
                message="A DataLoader prefetch factor was set while `num_workers=0`.",
                suggestion="Unset `prefetch_factor` or raise `num_workers` above zero.",
            )
        )
    if environment.is_apple_silicon and data_config.num_workers > 2:
        # This remains a heuristic warning rather than an error because some
        # Apple Silicon machines may benchmark well above this. The point is to
        # document the repo's "start small, benchmark upward" stance.
        diagnostics.append(
            RuntimeDiagnostic(
                severity="warning",
                code="apple_worker_count_high",
                message="Apple Silicon runs often work best with only a small number of DataLoader workers.",
                suggestion="Keep `num_workers` around 0-2 on Apple Silicon unless benchmarking proves otherwise.",
            )
        )
    if environment.mps_available and data_config.pin_memory:
        diagnostics.append(
            RuntimeDiagnostic(
                severity="warning",
                code="mps_pin_memory",
                message="Pinned host memory is usually not helpful for MPS runs.",
                suggestion="Set `pin_memory=False` on Apple Silicon unless benchmarking proves otherwise.",
            )
        )
    if train_config.deterministic and train_config.cudnn_benchmark:
        # This warning exists because the two settings express different goals:
        # deterministic mode asks for repeatability, while cuDNN benchmark asks
        # for fastest kernel selection based on runtime shape probing.
        diagnostics.append(
            RuntimeDiagnostic(
                severity="warning",
                code="deterministic_with_cudnn_benchmark",
                message="Deterministic mode was requested together with cuDNN benchmark mode.",
                suggestion="Disable `cudnn_benchmark` when reproducibility matters more than throughput.",
            )
        )
    if accelerator == "mps" and train_config.compile_model:
        # Compile on MPS is not forbidden, but it is called out because this is
        # still one of the more likely "advanced optimization" knobs to cause
        # surprise regressions or unsupported-operator paths on Apple Silicon.
        diagnostics.append(
            RuntimeDiagnostic(
                severity="warning",
                code="mps_compile_experimental",
                message="`torch.compile` remains more experimental on MPS than on CPU/CUDA.",
                suggestion="Disable compile on Apple Silicon if you see regressions or unsupported-operator failures.",
            )
        )
    if environment.is_slurm and train_config.enable_progress_bar:
        diagnostics.append(
            RuntimeDiagnostic(
                severity="warning",
                code="slurm_progress_bar",
                message="Interactive progress bars can create noisy batch logs on Slurm jobs.",
                suggestion="Disable progress bars for batch jobs unless you are using an interactive allocation.",
            )
        )
    if observability_config.enable_torchview and not environment.torchview_available:
        diagnostics.append(
            RuntimeDiagnostic(
                severity="warning",
                code="torchview_missing",
                message="Torchview rendering is enabled, but the `torchview` package is not installed.",
                suggestion="Install `torchview` or disable torchview rendering for this environment.",
            )
        )
    if observability_config.enable_tensorboard and not environment.tensorboard_available:
        diagnostics.append(
            RuntimeDiagnostic(
                severity="warning",
                code="tensorboard_missing",
                message="TensorBoard logging is enabled, but the `tensorboard` package is not installed.",
                suggestion="Install `tensorboard` or rely on the CSV fallback logger.",
            )
        )

    if requested_profile == "auto":
        diagnostics.append(
            RuntimeDiagnostic(
                severity="info",
                code="auto_profile_resolved",
                message=f"`auto` resolved to `{resolved_profile}` for this environment.",
            )
        )

    return tuple(diagnostics)


def has_error_diagnostics(diagnostics: Sequence[RuntimeDiagnostic]) -> bool:
    """
    Return whether the diagnostic set contains at least one hard error.
    """
    return any(diagnostic.severity == "error" for diagnostic in diagnostics)


def format_runtime_diagnostics(diagnostics: Sequence[RuntimeDiagnostic]) -> str:
    """
    Render diagnostics into a compact multi-line string for CLI and exceptions.
    """

    if not diagnostics:
        return "No runtime diagnostics were recorded."

    lines: list[str] = []
    for diagnostic in diagnostics:
        prefix = diagnostic.severity.upper()
        line = f"[{prefix}] {diagnostic.code}: {diagnostic.message}"
        if diagnostic.suggestion:
            line += f" Suggested fix: {diagnostic.suggestion}"
        lines.append(line)
    return "\n".join(lines)


def analyze_runtime_failure(
    exc: BaseException,
    *,
    requested_profile: str,
    resolved_profile: str,
    environment: RuntimeEnvironment,
) -> tuple[RuntimeDiagnostic, ...]:
    """
    Classify a runtime exception into likely environment-related causes.

    Context:
    this is intentionally a coarse classifier layered on top of the original
    exception. It tries to answer "does this look like an environment/setup
    problem?" without pretending it can replace the underlying error message.
    """

    message = str(exc).lower()
    diagnostics: list[RuntimeDiagnostic] = []

    # The matching rules below are intentionally simple and transparent. That
    # makes them easier to maintain than a more opaque classifier and reduces
    # the risk of overconfident explanations.
    if "no module named" in message:
        diagnostics.append(
            RuntimeDiagnostic(
                severity="error",
                code="runtime_missing_dependency",
                message="Execution failed because a required Python dependency is missing.",
                suggestion="Activate the correct environment and install the project requirements.",
            )
        )
    if "cuda" in message and any(
        pattern in message
        for pattern in (
            "not compiled with cuda",
            "found no nvidia driver",
            "cuda error",
            "invalid device ordinal",
        )
    ):
        diagnostics.append(
            RuntimeDiagnostic(
                severity="error",
                code="runtime_cuda_failure",
                message="Execution failed in a way that looks CUDA-related.",
                suggestion=(
                    "Verify the requested profile, the visible GPUs, and the installed CUDA-enabled PyTorch build."
                ),
            )
        )
    if "mps" in message and "available" in message:
        diagnostics.append(
            RuntimeDiagnostic(
                severity="error",
                code="runtime_mps_failure",
                message="Execution failed in a way that looks MPS-related.",
                suggestion="Switch to CPU or confirm that MPS is available in the active PyTorch build.",
            )
        )
    if "permission denied" in message or "read-only file system" in message:
        diagnostics.append(
            RuntimeDiagnostic(
                severity="error",
                code="runtime_filesystem_failure",
                message="Execution failed because the runtime could not write to the requested filesystem location.",
                suggestion="Use a writable output directory or adjust job permissions.",
            )
        )
    if environment.is_slurm and (
        "slurm" in message or "cuda_visible_devices" in message
    ):
        diagnostics.append(
            RuntimeDiagnostic(
                severity="error",
                code="runtime_slurm_allocation_failure",
                message="Execution failed in a way that may reflect a Slurm allocation or device visibility mismatch.",
                suggestion="Confirm the Slurm resource request and the job's visible devices.",
            )
        )

    if not diagnostics:
        diagnostics.append(
            RuntimeDiagnostic(
                severity="warning",
                code="runtime_environment_unknown",
                message=(
                    "Execution failed, but the failure did not match a known environment signature."
                ),
                suggestion=(
                    f"Review the stack trace together with the `{resolved_profile}` profile and recorded environment summary."
                ),
            )
        )

    if requested_profile == "auto":
        diagnostics.append(
            RuntimeDiagnostic(
                severity="info",
                code="runtime_auto_profile_context",
                message=(
                    f"The run started with `auto`, which resolved to `{resolved_profile}`."
                ),
            )
        )

    return tuple(diagnostics)
