from __future__ import annotations

# AI-assisted maintenance note:
# This module maps detected environments onto project-specific runtime
# profiles. It owns the environment-aware defaulting logic, but does not own
# hardware detection or diagnostic explanation.
#
# Responsibility boundary:
# - choose a concrete profile from the detected environment
# - apply profile defaults to runtime-facing config objects
# - preserve explicit user overrides
#
# What does *not* live here:
# - probing torch/Lightning/backend state
# - deciding whether a resolved setup is valid
# - formatting diagnostics for users
#
# Important disclaimer:
# these profiles are repository-specific convenience policies. They are meant
# to provide safer starting points for common environments, not to claim that
# the chosen settings are universally optimal for performance or stability.

from dataclasses import replace
from typing import Any

from config.data import DataConfig
from config.observability import ObservabilityConfig
from config.runtime import TrainConfig
from environment.types import DeviceProfileResolution, RuntimeEnvironment


# ============================================================================
# Profile Selection
# ============================================================================
# The `auto` profile is intentionally conservative in its priority order:
# 1. notebook / scheduler environments with strong external context
# 2. Apple Silicon MPS
# 3. local CUDA
# 4. plain local CPU
#
# This keeps explicit special environments from being mistaken for generic
# local runs.
def infer_device_profile(
    requested_profile: str,
    environment: RuntimeEnvironment,
) -> str:
    """
    Resolve `auto` into one concrete device profile for the current environment.
    """

    # Explicit profile requests always win over auto-detection.
    if requested_profile != "auto":
        return requested_profile

    # Hosted notebook environments are checked before generic local hardware so
    # their profile-specific policies are preserved.
    if environment.is_colab:
        return "colab-cuda" if environment.cuda_available else "colab-cpu"

    # Scheduler-managed jobs are checked before generic local CUDA / CPU so the
    # resolved profile reflects the batch-execution environment.
    if environment.is_slurm:
        return "slurm-cuda" if environment.cuda_available else "slurm-cpu"

    # Apple Silicon gets its own profile because MPS has different memory and
    # runtime tradeoffs than CUDA and plain CPU execution.
    if (
        environment.accelerator_type == "mps"
        or (environment.mps_available and environment.is_apple_silicon)
    ):
        return "apple-silicon"

    # If CUDA is available outside Colab / Slurm, treat the machine as a local
    # CUDA environment.
    if environment.accelerator_type == "cuda" or environment.cuda_available:
        return "local-cuda"

    # Fall back to a generic local CPU profile.
    return "local-cpu"


def _slurm_worker_default(environment: RuntimeEnvironment) -> int:
    """Choose a conservative DataLoader worker default for scheduler-managed jobs."""
    # Slurm allocations often represent the most trustworthy CPU budget signal
    # we have for cluster jobs because they reflect scheduler intent rather
    # than raw host capacity. We still leave one CPU free when possible so the
    # process is less likely to consume every allocated core with DataLoader
    # workers alone.
    if environment.slurm_cpus_per_task is None:
        return 2 # smaller value is safer for general slurms
    return max(0, min(environment.slurm_cpus_per_task - 1, 8))


def _cpu_parallelism_budget(environment: RuntimeEnvironment) -> int:
    """Estimate the usable host-side parallelism budget from physical or logical core counts."""
    # Physical cores are preferred over logical cores because loader workers
    # and Torch intra-op threads tend to saturate "real" core capacity first.
    # Logical count is still a useful fallback on platforms where physical-core
    # discovery is unavailable or unreliable.
    for count in (environment.cpu_count_physical, environment.cpu_count_logical):
        if count is not None and count > 0:
            return count
    return 1


def _local_cpu_worker_default(environment: RuntimeEnvironment) -> int:
    """Choose a modest worker count for local CPU-only runs."""
    # Local CPU runs need to balance two competing consumers of host parallelism:
    # DataLoader workers and Torch operator threads. Using roughly half the
    # machine for workers and capping at a modest value keeps the default
    # useful for laptops and workstations without oversubscribing the host.
    return max(0, min((_cpu_parallelism_budget(environment) // 2) - 1, 4))


def _local_cuda_worker_default(environment: RuntimeEnvironment) -> int:
    """Choose a more throughput-oriented worker count for local CUDA runs."""
    # CUDA runs usually benefit from a somewhat more aggressive loader pool
    # because the GPU can sit idle if batches are not prepared fast enough.
    # We still cap the count so the default remains sane on large hosts.
    return max(1, min(_cpu_parallelism_budget(environment) // 2, 8))


def _apple_silicon_worker_default(environment: RuntimeEnvironment) -> int:
    """Choose a smaller worker pool for Apple Silicon MPS runs."""
    # Apple Silicon often responds better to a smaller worker pool than CUDA
    # systems do. The platform can be sensitive to host-side loader overhead,
    # and the MPS path frequently benefits more from "few workers, simple
    # pipeline" than from scaling worker count upward.
    return max(0, min(_cpu_parallelism_budget(environment) // 4, 2))


def _persistent_workers_enabled(num_workers: int) -> bool:
    """Return whether persistent workers make semantic sense for the given worker count."""
    # Persistent workers only make semantic sense when there are workers to
    # persist. Centralizing that rule here keeps the profile table from having
    # to repeat the same guard in every branch.
    return num_workers > 0


def _prefetch_factor_default(num_workers: int, *, accelerator: str) -> int | None:
    """Choose the default DataLoader prefetch depth for the current accelerator class."""
    # `prefetch_factor` only matters when multiprocessing workers are active.
    # GPU runs get a deeper queue because keeping the device fed is usually the
    # higher priority, while CPU / MPS paths stay more conservative.
    if num_workers <= 0:
        return None
    if accelerator == "gpu":
        return 4
    return 2


def _cuda_compute_capability_major(environment: RuntimeEnvironment) -> int:
    """
    Extract the CUDA compute capability major version from strings like "7.5" or "8.0".

    Returns 0 when the value is unavailable or malformed so downstream policy
    stays conservative.
    """
    capability = environment.cuda_capability
    if not capability:
        return 0

    try:
        major_text, *_ = capability.split(".")
        return int(major_text)
    except (TypeError, ValueError):
        return 0


def _cuda_can_use_bf16(environment: RuntimeEnvironment) -> bool:
    """
    BF16 should only be used when both:
    - PyTorch reports support, and
    - the GPU architecture is Ampere+ (sm_80 or newer).

    This avoids selecting bf16 on older GPUs such as Tesla T4 (sm_75),
    which can still reach code paths that fail during Triton/PTX compilation.
    """
    return (
        environment.cuda_supports_bf16
        and _cuda_compute_capability_major(environment) >= 8
    )


def _cuda_precision_default(environment: RuntimeEnvironment) -> str:
    """
    Choose the preferred mixed-precision policy for CUDA-capable runs.

    BF16 is preferred only on Ampere+ GPUs with confirmed runtime support.
    Older CUDA GPUs fall back to FP16 mixed precision.
    """
    return "bf16-mixed" if _cuda_can_use_bf16(environment) else "16-mixed"


def _cpu_precision_default(environment: RuntimeEnvironment) -> int | str:
    """Choose the preferred precision policy for CPU-only runs."""
    # CPU BF16 is intentionally opt-in via capability detection because plain
    # CPU mixed precision is much less universal than CUDA mixed precision.
    # When support is unclear, the safer default remains full FP32.
    return "bf16-mixed" if environment.cpu_supports_bf16 else 32


def _cpu_intraop_threads_default(environment: RuntimeEnvironment) -> int:
    """Choose a default PyTorch intra-op thread budget for host-side execution."""
    # Intra-op threads govern how much parallelism a single operator may use.
    # Capping them avoids one local run consuming every host thread by default.
    return max(1, min(_cpu_parallelism_budget(environment), 8))


def _cpu_interop_threads_default(environment: RuntimeEnvironment) -> int:
    """Choose a default PyTorch inter-op thread budget for host-side execution."""
    # Inter-op parallelism should usually stay lower than intra-op parallelism.
    # This helps CPU and MPS runs avoid multiplying "operators in flight" by
    # "threads per operator" into an oversubscribed mess.
    return max(1, min(_cpu_parallelism_budget(environment) // 2, 4))


def _compile_defaults_for_profile(
    profile: str,
    environment: RuntimeEnvironment,
) -> tuple[bool, str | None]:
    """Return whether `torch.compile` should be enabled by default for the given profile."""
    # Compilation is treated as a profile-sensitive throughput experiment, not
    # a universal truth. Local CUDA gets the most aggressive default because
    # hardware is usually known and controlled. Slurm CUDA stays more cautious
    # because cluster jobs may land on heterogeneous or older GPUs.

    # Local CUDA → keep aggressive (you control hardware)
    if profile == "local-cuda":
        return True, "default"

    # Slurm CUDA → be conservative (heterogeneous GPUs)
    # Only enable compile by default on Ampere+ GPUs where the CUDA backend is
    # less likely to trip older-hardware compiler issues.
    if profile == "slurm-cuda":
        if _cuda_compute_capability_major(environment) >= 8:  # Ampere+
            return True, "default"
        return False, None  # Disable compile on T4, etc.

    # CPU
    if profile == "local-cpu":
        return True, "reduce-overhead"

    return False, None


def resolve_device_profile(
    *,
    requested_profile: str,
    environment: RuntimeEnvironment,
    train_config: TrainConfig,
    data_config: DataConfig,
    observability_config: ObservabilityConfig,
    explicit_overrides: set[str] | None = None,
) -> DeviceProfileResolution:
    """
    Apply one high-level device profile to the runtime-facing config objects.

    Context:
    this function is the translation layer from "one user-facing runtime
    choice" to concrete config updates across Trainer, DataLoader, and
    observability policy.
    """

    overrides = explicit_overrides or set()
    resolved_profile = infer_device_profile(requested_profile, environment)

    # These dictionaries collect only the fields that this profile wants to
    # change. They are later applied with `dataclasses.replace(...)`.
    train_updates: dict[str, Any] = {}
    data_updates: dict[str, Any] = {}
    observability_updates: dict[str, Any] = {}
    applied_defaults: dict[str, Any] = {}

    def maybe_update(
        destination: dict[str, Any],
        field_name: str,
        value: Any,
        *,
        cli_dest: str | None = None,
    ) -> None:
        """Apply one profile default only when the user did not already override that field."""
        # Explicit user inputs remain authoritative. Profiles only provide
        # defaults for settings the caller did not specify directly.
        #
        # This is the core reason profiles can coexist with low-level flags:
        # the profile gives the repo a strong environment-specific starting
        # point, but one experiment can still override only the handful of
        # knobs it cares about.
        override_name = field_name if cli_dest is None else cli_dest
        if override_name in overrides:
            return
        destination[field_name] = value
        applied_defaults[field_name] = value

    # The profile table is intentionally explicit rather than data-driven. That
    # makes it easier to answer:
    # - what does each environment profile actually do?
    # - which knobs differ between profiles?
    # - where should a future profile-specific tweak be made?
    #
    # The comments inside each branch focus on the "why" of the profile, not
    # just the "what". That is important here because these are performance and
    # stability heuristics, not immutable truths.
    if resolved_profile == "local-cpu":
        num_workers = _local_cpu_worker_default(environment)
        compile_model, compile_mode = _compile_defaults_for_profile(
            resolved_profile,
            environment,
        )

        # Local CPU defaults lean conservative on memory movement and slightly
        # more aggressive on thread tuning. The goal is to help a workstation
        # or laptop make decent use of host compute without pretending it is a
        # GPU-first throughput environment.

        # Run all training and inference work on the CPU.
        maybe_update(train_updates, "accelerator", "cpu")

        # Use one device because this profile is not a distributed CPU policy.
        maybe_update(train_updates, "devices", 1)

        # Prefer BF16 mixed precision on CPU only when the runtime reports
        # support. Otherwise stay with full FP32.
        maybe_update(train_updates, "precision", _cpu_precision_default(environment))

        # Hint PyTorch toward higher internal matmul precision heuristics.
        maybe_update(train_updates, "matmul_precision", "high")

        # Limit how many host threads one operator may use.
        maybe_update(
            train_updates,
            "intraop_threads",
            _cpu_intraop_threads_default(environment),
        )

        # Limit how many separate operators may execute in parallel.
        maybe_update(
            train_updates,
            "interop_threads",
            _cpu_interop_threads_default(environment),
        )

        # Enable or disable `torch.compile` according to profile policy.
        maybe_update(train_updates, "compile_model", compile_model)

        # Choose the compile strategy when compilation is enabled.
        maybe_update(train_updates, "compile_mode", compile_mode)

        # Number of multiprocessing DataLoader workers.
        maybe_update(data_updates, "num_workers", num_workers)

        # Pinned memory mainly benefits host-to-GPU transfer, so keep it off on CPU.
        maybe_update(data_updates, "pin_memory", False)

        # Keep workers alive across epochs only when workers exist.
        maybe_update(
            data_updates,
            "persistent_workers",
            _persistent_workers_enabled(num_workers),
        )

        # Queue a modest number of prefetched batches per worker.
        maybe_update(
            data_updates,
            "prefetch_factor",
            _prefetch_factor_default(num_workers, accelerator="cpu"),
        )

    elif resolved_profile == "local-cuda":
        num_workers = _local_cuda_worker_default(environment)
        compile_model, compile_mode = _compile_defaults_for_profile(
            resolved_profile,
            environment,
        )

        # Local CUDA is the profile where throughput-oriented defaults are most
        # justified: pinned memory, deeper prefetching, TF32, and optional
        # compile all exist to reduce the chance that the GPU waits on the host.

        # Run training on CUDA.
        maybe_update(train_updates, "accelerator", "gpu")

        # Use one GPU by default for this profile.
        maybe_update(train_updates, "devices", 1)

        # Prefer BF16 on Ampere+ when safe; otherwise use FP16 mixed precision.
        maybe_update(train_updates, "precision", _cuda_precision_default(environment))

        # Hint PyTorch toward higher internal matmul precision heuristics.
        maybe_update(train_updates, "matmul_precision", "high")

        # Allow TF32 on supported NVIDIA GPUs for faster matmul / convolution paths.
        maybe_update(train_updates, "allow_tf32", True)

        # Let cuDNN benchmark kernels when deterministic mode is not required.
        maybe_update(
            train_updates,
            "cudnn_benchmark",
            not train_config.deterministic,
        )

        # Enable or disable `torch.compile` according to profile policy.
        maybe_update(train_updates, "compile_model", compile_model)

        # Choose the compile strategy when compilation is enabled.
        maybe_update(train_updates, "compile_mode", compile_mode)

        # Number of multiprocessing DataLoader workers.
        maybe_update(data_updates, "num_workers", num_workers)

        # Enable pinned host memory to accelerate host-to-GPU batch transfer.
        maybe_update(data_updates, "pin_memory", True)

        # Keep workers alive across epochs to reduce worker startup overhead.
        maybe_update(
            data_updates,
            "persistent_workers",
            _persistent_workers_enabled(num_workers),
        )

        # Use deeper prefetching to reduce the chance of starving the GPU.
        maybe_update(
            data_updates,
            "prefetch_factor",
            _prefetch_factor_default(num_workers, accelerator="gpu"),
        )

    elif resolved_profile == "colab-cpu":
        # Colab CPU sessions are usually smaller, more volatile notebook
        # environments. Keeping workers at zero and disabling heavier extras
        # tends to be a safer "works out of the box" starting point.

        maybe_update(train_updates, "accelerator", "cpu")
        maybe_update(train_updates, "devices", 1)
        maybe_update(train_updates, "precision", _cpu_precision_default(environment))
        maybe_update(train_updates, "matmul_precision", "high")

        # Zero workers reduces multiprocessing friction in notebook runtimes.
        maybe_update(data_updates, "num_workers", 0)

        # No GPU transfer path here, so pinned memory is unnecessary.
        maybe_update(data_updates, "pin_memory", False)

        # With zero workers, persistent workers have no meaning.
        maybe_update(data_updates, "persistent_workers", False)

        # Prefetching is also irrelevant with zero workers.
        maybe_update(data_updates, "prefetch_factor", None)

        # Disable heavier model visualization extras by default in Colab CPU.
        maybe_update(observability_updates, "enable_torchview", False)

    elif resolved_profile == "colab-cuda":
        # Colab CUDA gets some GPU-friendly defaults, but remains more cautious
        # than a tuned local workstation because notebook kernels and hosted
        # VMs can be more sensitive to aggressive worker counts.

        maybe_update(train_updates, "accelerator", "gpu")
        maybe_update(train_updates, "devices", 1)
        maybe_update(train_updates, "precision", _cuda_precision_default(environment))
        maybe_update(train_updates, "matmul_precision", "high")
        maybe_update(train_updates, "allow_tf32", True)

        # Benchmark cuDNN kernels only when deterministic mode is not required.
        maybe_update(
            train_updates,
            "cudnn_benchmark",
            not train_config.deterministic,
        )

        # Modest worker count for notebook-hosted GPU sessions.
        maybe_update(data_updates, "num_workers", 2)

        # Pinned memory helps CPU-to-GPU transfer.
        maybe_update(data_updates, "pin_memory", True)

        # Reuse workers across epochs when multiprocessing is enabled.
        maybe_update(data_updates, "persistent_workers", True)

        # Mildly deeper prefetching to keep the GPU fed.
        maybe_update(
            data_updates,
            "prefetch_factor",
            _prefetch_factor_default(2, accelerator="gpu"),
        )

        # Disable heavier visualization extras by default in Colab.
        maybe_update(observability_updates, "enable_torchview", False)

    elif resolved_profile == "slurm-cpu":
        num_workers = _slurm_worker_default(environment)

        # Scheduler-backed CPU jobs are where it makes the most sense to trust
        # the allocation shape and disable chatty interactive UX by default.

        maybe_update(train_updates, "accelerator", "cpu")
        maybe_update(train_updates, "devices", 1)
        maybe_update(train_updates, "precision", _cpu_precision_default(environment))

        # Hide Lightning's interactive progress bar in batch-job logs.
        maybe_update(train_updates, "enable_progress_bar", False)

        # Favor higher internal matmul precision heuristics when supported.
        maybe_update(train_updates, "matmul_precision", "high")

        maybe_update(
            train_updates,
            "intraop_threads",
            _cpu_intraop_threads_default(environment),
        )
        maybe_update(
            train_updates,
            "interop_threads",
            _cpu_interop_threads_default(environment),
        )

        maybe_update(data_updates, "num_workers", num_workers)
        maybe_update(data_updates, "pin_memory", False)

        maybe_update(
            data_updates,
            "persistent_workers",
            _persistent_workers_enabled(num_workers),
        )

        # Queue a modest number of prefetched batches per worker.
        maybe_update(
            data_updates,
            "prefetch_factor",
            _prefetch_factor_default(num_workers, accelerator="cpu"),
        )

        # Disable richer terminal progress UI for non-interactive scheduler logs.
        maybe_update(observability_updates, "enable_rich_progress_bar", False)

    elif resolved_profile == "slurm-cuda":
        num_workers = _slurm_worker_default(environment)
        compile_model, compile_mode = _compile_defaults_for_profile(
            resolved_profile,
            environment,
        )

        # Slurm CUDA keeps the throughput-oriented GPU defaults but also bakes
        # in batch-job ergonomics such as muted progress output.

        # Run training on CUDA.
        maybe_update(train_updates, "accelerator", "gpu")

        # Use one GPU by default for this profile.
        maybe_update(train_updates, "devices", 1)

        # Prefer BF16 only on safe Ampere+ hardware; otherwise use FP16 mixed precision.
        maybe_update(train_updates, "precision", _cuda_precision_default(environment))

        # Disable interactive progress output in scheduler logs.
        maybe_update(train_updates, "enable_progress_bar", False)

        maybe_update(train_updates, "matmul_precision", "high")

        # Allow TF32 on supported NVIDIA hardware.
        maybe_update(train_updates, "allow_tf32", True)

        # Let cuDNN benchmark kernels only when determinism is not required.
        maybe_update(
            train_updates,
            "cudnn_benchmark",
            not train_config.deterministic,
        )

        # Enable or disable `torch.compile` according to profile policy.
        maybe_update(train_updates, "compile_model", compile_model)

        # Choose the compile strategy when compilation is enabled.
        maybe_update(train_updates, "compile_mode", compile_mode)

        maybe_update(data_updates, "num_workers", num_workers)

        # Pinned memory improves host-to-GPU transfer in batch jobs too.
        maybe_update(data_updates, "pin_memory", True)

        maybe_update(
            data_updates,
            "persistent_workers",
            _persistent_workers_enabled(num_workers),
        )

        # Use deeper prefetching to reduce the chance of starving the GPU.
        maybe_update(
            data_updates,
            "prefetch_factor",
            _prefetch_factor_default(num_workers, accelerator="gpu"),
        )

        # Rich terminal progress UI is usually noisy in Slurm logs.
        maybe_update(observability_updates, "enable_rich_progress_bar", False)

    elif resolved_profile == "apple-silicon":
        num_workers = _apple_silicon_worker_default(environment)

        # Apple Silicon is intentionally treated as its own profile rather than
        # "CPU with different accelerator name". MPS has different mixed-
        # precision, memory, and loader-behavior tradeoffs, so it deserves a
        # separate default table.

        maybe_update(train_updates, "accelerator", "mps")
        maybe_update(train_updates, "devices", 1)

        # Stay in FP32 by default because MPS precision tradeoffs differ from CUDA.
        maybe_update(train_updates, "precision", 32)

        maybe_update(train_updates, "matmul_precision", "high")

        maybe_update(
            train_updates,
            "intraop_threads",
            _cpu_intraop_threads_default(environment),
        )
        maybe_update(
            train_updates,
            "interop_threads",
            _cpu_interop_threads_default(environment),
        )

        # Upper memory watermark ratio for MPS allocator behavior.
        maybe_update(train_updates, "mps_high_watermark_ratio", 1.3)

        # Lower memory watermark ratio for MPS allocator behavior.
        maybe_update(train_updates, "mps_low_watermark_ratio", 1.0)

        # Allow unsupported ops to fall back when the MPS backend cannot handle them.
        maybe_update(train_updates, "enable_mps_fallback", True)

        maybe_update(data_updates, "num_workers", num_workers)

        # Pinned memory is primarily for CUDA transfer, so keep it off on MPS.
        maybe_update(data_updates, "pin_memory", False)

        maybe_update(
            data_updates,
            "persistent_workers",
            _persistent_workers_enabled(num_workers),
        )

        maybe_update(
            data_updates,
            "prefetch_factor",
            _prefetch_factor_default(num_workers, accelerator="mps"),
        )

        # Device stats can be noisier or less useful on MPS in this repo.
        maybe_update(observability_updates, "enable_device_stats", False)

    resolved_train_config = (
        replace(train_config, **train_updates) if train_updates else train_config
    )
    resolved_data_config = (
        replace(data_config, **data_updates) if data_updates else data_config
    )
    resolved_observability_config = (
        replace(observability_config, **observability_updates)
        if observability_updates
        else observability_config
    )

    return DeviceProfileResolution(
        requested_profile=requested_profile,
        resolved_profile=resolved_profile,
        train_config=resolved_train_config,
        data_config=resolved_data_config,
        observability_config=resolved_observability_config,
        applied_defaults=applied_defaults,
    )