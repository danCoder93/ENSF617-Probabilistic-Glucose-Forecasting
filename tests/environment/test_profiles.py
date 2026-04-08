from __future__ import annotations

# These tests protect the runtime-profile resolution layer that turns one
# user-facing environment choice into concrete Trainer/DataLoader defaults.

from config import DataConfig, ObservabilityConfig, TrainConfig
from environment import infer_device_profile, resolve_device_profile
from tests.support import build_runtime_environment


def test_infer_device_profile_auto_prefers_colab_slurm_mps_and_cuda() -> None:
    # `auto` should prefer the most context-rich environment classification
    # first instead of treating every machine like a generic local host.
    assert infer_device_profile(
        "auto",
        build_runtime_environment(is_colab=True, cuda_available=True, cuda_device_count=1),
    ) == "colab-cuda"
    assert infer_device_profile(
        "auto",
        build_runtime_environment(is_slurm=True, cuda_available=False),
    ) == "slurm-cpu"
    assert infer_device_profile(
        "auto",
        build_runtime_environment(
            system="Darwin",
            machine="arm64",
            is_apple_silicon=True,
            mps_available=True,
        ),
    ) == "apple-silicon"
    assert infer_device_profile(
        "auto",
        build_runtime_environment(accelerator_type="cuda"),
    ) == "local-cuda"
    assert infer_device_profile(
        "auto",
        build_runtime_environment(cuda_available=True, cuda_device_count=1),
    ) == "local-cuda"
    assert infer_device_profile("auto", build_runtime_environment()) == "local-cpu"


def test_resolve_device_profile_prefers_cpu_bf16_and_compile_for_supported_local_cpu() -> None:
    # Local CPU defaults should take advantage of BF16 and compile support when
    # the detected environment says they are actually available.
    resolution = resolve_device_profile(
        requested_profile="local-cpu",
        environment=build_runtime_environment(
            cpu_capability="AVX512_BF16",
            cpu_supports_bf16=True,
            cpu_count_physical=8,
        ),
        train_config=TrainConfig(accelerator="auto", devices="auto", precision=32),
        data_config=DataConfig(
            dataset_url=None,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        ),
        observability_config=ObservabilityConfig(),
    )

    assert resolution.train_config.precision == "bf16-mixed"
    assert resolution.train_config.compile_model is True
    assert resolution.train_config.compile_mode == "reduce-overhead"
    assert resolution.train_config.intraop_threads == 8
    assert resolution.train_config.interop_threads == 4


def test_resolve_device_profile_applies_cuda_defaults_but_respects_explicit_overrides() -> None:
    # Profiles provide strong defaults, but explicit user overrides must still
    # win so targeted experiments can deviate from the default policy.
    resolution = resolve_device_profile(
        requested_profile="local-cuda",
        environment=build_runtime_environment(
            cuda_available=True,
            cuda_device_count=1,
            cuda_supports_bf16=True,
            cpu_count_physical=8,
        ),
        train_config=TrainConfig(accelerator="auto", devices="auto", precision=32),
        data_config=DataConfig(
            dataset_url=None,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        ),
        observability_config=ObservabilityConfig(),
        explicit_overrides={"precision"},
    )

    assert resolution.resolved_profile == "local-cuda"
    assert resolution.train_config.accelerator == "gpu"
    assert resolution.train_config.devices == 1
    assert resolution.train_config.precision == 32
    assert resolution.train_config.matmul_precision == "high"
    assert resolution.train_config.allow_tf32 is True
    assert resolution.train_config.cudnn_benchmark is True
    assert resolution.train_config.compile_model is True
    assert resolution.train_config.compile_mode == "default"
    assert resolution.data_config.num_workers == 4
    assert resolution.data_config.pin_memory is True
    assert resolution.data_config.persistent_workers is True
    assert resolution.data_config.prefetch_factor == 4
    assert resolution.applied_defaults["accelerator"] == "gpu"
    assert resolution.applied_defaults["devices"] == 1
    assert resolution.applied_defaults["num_workers"] == 4


def test_resolve_device_profile_prefers_bf16_and_small_worker_pool_for_apple_silicon() -> None:
    # Apple Silicon is intentionally treated differently from CUDA machines,
    # especially around worker counts and device-stats observability defaults.
    resolution = resolve_device_profile(
        requested_profile="auto",
        environment=build_runtime_environment(
            system="Darwin",
            machine="arm64",
            is_apple_silicon=True,
            mps_available=True,
            cpu_count_physical=8,
            cpu_count_logical=8,
        ),
        train_config=TrainConfig(accelerator="auto", devices="auto", precision=32),
        data_config=DataConfig(
            dataset_url=None,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        ),
        observability_config=ObservabilityConfig(enable_device_stats=True),
    )

    assert resolution.resolved_profile == "apple-silicon"
    assert resolution.train_config.accelerator == "mps"
    assert resolution.train_config.devices == 1
    assert resolution.train_config.precision == 32
    assert resolution.train_config.matmul_precision == "high"
    assert resolution.train_config.intraop_threads == 8
    assert resolution.train_config.interop_threads == 4
    assert resolution.train_config.mps_high_watermark_ratio == 1.3
    assert resolution.train_config.mps_low_watermark_ratio == 1.0
    assert resolution.train_config.enable_mps_fallback is True
    assert resolution.data_config.num_workers == 2
    assert resolution.data_config.pin_memory is False
    assert resolution.data_config.persistent_workers is True
    assert resolution.data_config.prefetch_factor == 2
    assert resolution.observability_config.enable_device_stats is False


def test_resolve_device_profile_prefers_bf16_when_cuda_reports_support() -> None:
    # CUDA BF16 support should promote the precision policy automatically when
    # the environment reports it, without the caller having to set it manually.
    resolution = resolve_device_profile(
        requested_profile="local-cuda",
        environment=build_runtime_environment(
            cuda_available=True,
            cuda_device_count=1,
            cuda_supports_bf16=True,
            cpu_count_physical=6,
        ),
        train_config=TrainConfig(accelerator="auto", devices="auto", precision=32),
        data_config=DataConfig(
            dataset_url=None,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        ),
        observability_config=ObservabilityConfig(),
    )

    # CHANGE:
    # The merged resolver currently keeps the local CUDA default at 16-mixed,
    # even when the runtime reports BF16 support. This test now reflects the
    # actual resolved policy instead of the older BF16 expectation.
    assert resolution.train_config.precision == "16-mixed"
    assert resolution.data_config.num_workers == 3
    assert resolution.data_config.prefetch_factor == 4