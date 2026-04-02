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
# This keeps explicit special environments from being mistaken for generic
# local runs.
def infer_device_profile(
    requested_profile: str,
    environment: RuntimeEnvironment,
) -> str:
    """
    Resolve `auto` into one concrete device profile for the current environment.
    """

    if requested_profile != "auto":
        return requested_profile

    if environment.is_colab:
        return "colab-cuda" if environment.cuda_available else "colab-cpu"
    if environment.is_slurm:
        return "slurm-cuda" if environment.cuda_available else "slurm-cpu"
    if (
        environment.accelerator_type == "mps"
        or (environment.mps_available and environment.is_apple_silicon)
    ):
        return "apple-silicon"
    if environment.accelerator_type == "cuda" or environment.cuda_available:
        return "local-cuda"
    return "local-cpu"


def _slurm_worker_default(environment: RuntimeEnvironment) -> int:
    """Choose a conservative DataLoader worker default for scheduler-managed jobs."""
    # Slurm allocations often represent the most trustworthy CPU budget signal
    # we have for cluster jobs because they reflect scheduler intent rather
    # than raw host capacity. We still leave one CPU free when possible so the
    # process is less likely to consume every allocated core with DataLoader
    # workers alone.
    if environment.slurm_cpus_per_task is None:
        return 4
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
    # higher priority, while CPU/MPS paths stay more conservative.
    if num_workers <= 0:
        return None
    if accelerator == "gpu":
        return 4
    return 2


def _cuda_precision_default(environment: RuntimeEnvironment) -> str:
    """Choose the preferred mixed-precision policy for CUDA-capable runs."""
    # BF16 is preferred when the CUDA stack reports support because it usually
    # gives the mixed-precision benefits we want with fewer numeric edge cases
    # than FP16. The fallback to `16-mixed` keeps older or narrower GPUs fast.
    return "bf16-mixed" if environment.cuda_supports_bf16 else "16-mixed"


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


def _compile_defaults_for_profile(profile: str) -> tuple[bool, str | None]:
    """Return whether `torch.compile` should be enabled by default for the given profile."""
    # Compilation is treated as a profile-sensitive throughput experiment, not
    # a universal truth. CPU and CUDA local runs get the most useful defaults,
    # while the other profiles stay more conservative unless the caller opts in.
    if profile in {"local-cuda", "slurm-cuda"}:
        return True, "default"
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
        compile_model, compile_mode = _compile_defaults_for_profile(resolved_profile)
        # Local CPU defaults lean conservative on memory movement and slightly
        # more aggressive on thread tuning. The goal is to help a workstation
        # or laptop make decent use of host compute without pretending it is a
        # GPU-first throughput environment.
        maybe_update(train_updates, "accelerator", "cpu")
        maybe_update(train_updates, "devices", 1)
        maybe_update(train_updates, "precision", _cpu_precision_default(environment))
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
        maybe_update(train_updates, "compile_model", compile_model)
        maybe_update(train_updates, "compile_mode", compile_mode)
        maybe_update(data_updates, "num_workers", num_workers)
        maybe_update(data_updates, "pin_memory", False)
        maybe_update(
            data_updates,
            "persistent_workers",
            _persistent_workers_enabled(num_workers),
        )
        maybe_update(
            data_updates,
            "prefetch_factor",
            _prefetch_factor_default(num_workers, accelerator="cpu"),
        )
    elif resolved_profile == "local-cuda":
        num_workers = _local_cuda_worker_default(environment)
        compile_model, compile_mode = _compile_defaults_for_profile(resolved_profile)
        # Local CUDA is the profile where throughput-oriented defaults are most
        # justified: pinned memory, deeper prefetching, TF32, and optional
        # compile all exist to reduce the chance that the GPU waits on the host.
        maybe_update(train_updates, "accelerator", "gpu")
        maybe_update(train_updates, "devices", 1)
        maybe_update(train_updates, "precision", _cuda_precision_default(environment))
        maybe_update(train_updates, "matmul_precision", "high")
        maybe_update(train_updates, "allow_tf32", True)
        maybe_update(train_updates, "cudnn_benchmark", not train_config.deterministic)
        maybe_update(train_updates, "compile_model", compile_model)
        maybe_update(train_updates, "compile_mode", compile_mode)
        maybe_update(data_updates, "num_workers", num_workers)
        maybe_update(data_updates, "pin_memory", True)
        maybe_update(
            data_updates,
            "persistent_workers",
            _persistent_workers_enabled(num_workers),
        )
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
        maybe_update(data_updates, "num_workers", 0)
        maybe_update(data_updates, "pin_memory", False)
        maybe_update(data_updates, "persistent_workers", False)
        maybe_update(data_updates, "prefetch_factor", None)
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
        maybe_update(train_updates, "cudnn_benchmark", not train_config.deterministic)
        maybe_update(data_updates, "num_workers", 2)
        maybe_update(data_updates, "pin_memory", True)
        maybe_update(data_updates, "persistent_workers", True)
        maybe_update(
            data_updates,
            "prefetch_factor",
            _prefetch_factor_default(2, accelerator="gpu"),
        )
        maybe_update(observability_updates, "enable_torchview", False)
    elif resolved_profile == "slurm-cpu":
        num_workers = _slurm_worker_default(environment)
        # Scheduler-backed CPU jobs are where it makes the most sense to trust
        # the allocation shape and disable chatty interactive UX by default.
        maybe_update(train_updates, "accelerator", "cpu")
        maybe_update(train_updates, "devices", 1)
        maybe_update(train_updates, "precision", _cpu_precision_default(environment))
        maybe_update(train_updates, "enable_progress_bar", False)
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
        maybe_update(
            data_updates,
            "prefetch_factor",
            _prefetch_factor_default(num_workers, accelerator="cpu"),
        )
        maybe_update(observability_updates, "enable_rich_progress_bar", False)
    elif resolved_profile == "slurm-cuda":
        num_workers = _slurm_worker_default(environment)
        compile_model, compile_mode = _compile_defaults_for_profile(resolved_profile)
        # Slurm CUDA keeps the throughput-oriented GPU defaults but also bakes
        # in batch-job ergonomics such as muted progress output.
        maybe_update(train_updates, "accelerator", "gpu")
        maybe_update(train_updates, "devices", 1)
        maybe_update(train_updates, "precision", _cuda_precision_default(environment))
        maybe_update(train_updates, "enable_progress_bar", False)
        maybe_update(train_updates, "matmul_precision", "high")
        maybe_update(train_updates, "allow_tf32", True)
        maybe_update(train_updates, "cudnn_benchmark", not train_config.deterministic)
        maybe_update(train_updates, "compile_model", compile_model)
        maybe_update(train_updates, "compile_mode", compile_mode)
        maybe_update(data_updates, "num_workers", num_workers)
        maybe_update(data_updates, "pin_memory", True)
        maybe_update(
            data_updates,
            "persistent_workers",
            _persistent_workers_enabled(num_workers),
        )
        maybe_update(
            data_updates,
            "prefetch_factor",
            _prefetch_factor_default(num_workers, accelerator="gpu"),
        )
        maybe_update(observability_updates, "enable_rich_progress_bar", False)
    elif resolved_profile == "apple-silicon":
        num_workers = _apple_silicon_worker_default(environment)
        # Apple Silicon is intentionally treated as its own profile rather than
        # "CPU with different accelerator name". MPS has different mixed-
        # precision, memory, and loader-behavior tradeoffs, so it deserves a
        # separate default table.
        maybe_update(train_updates, "accelerator", "mps")
        maybe_update(train_updates, "devices", 1)
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
        maybe_update(train_updates, "mps_high_watermark_ratio", 1.3)
        maybe_update(train_updates, "mps_low_watermark_ratio", 1.0)
        maybe_update(train_updates, "enable_mps_fallback", True)
        maybe_update(data_updates, "num_workers", num_workers)
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
