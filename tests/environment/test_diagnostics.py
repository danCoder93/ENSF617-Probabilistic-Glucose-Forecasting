from __future__ import annotations

# These tests protect the environment-diagnostics layer that turns runtime state
# plus config into actionable warnings and errors.

from config import DataConfig, ObservabilityConfig, TrainConfig
from environment import (
    RuntimeDiagnostic,
    collect_runtime_diagnostics,
    format_runtime_diagnostics,
    has_error_diagnostics,
)
from tests.support import build_runtime_environment


def test_collect_runtime_diagnostics_flags_backend_mismatch_and_missing_packages() -> None:
    # A requested CUDA run without torch, Lightning, or CUDA support should
    # surface multiple independent diagnostics rather than only one generic error.
    diagnostics = collect_runtime_diagnostics(
        requested_profile="local-cuda",
        resolved_profile="local-cuda",
        environment=build_runtime_environment(
            torch_available=False,
            pytorch_lightning_available=False,
            cuda_available=False,
            torchview_available=False,
        ),
        train_config=TrainConfig(accelerator="gpu", devices=1, precision="16-mixed"),
        data_config=DataConfig(
            dataset_url=None,
            num_workers=0,
            pin_memory=True,
            persistent_workers=False,
        ),
        observability_config=ObservabilityConfig(enable_torchview=True),
    )

    codes = {diagnostic.code for diagnostic in diagnostics}
    assert has_error_diagnostics(diagnostics) is True
    assert "missing_torch" in codes
    assert "missing_lightning" in codes
    assert "cuda_unavailable" in codes


def test_collect_runtime_diagnostics_flags_bf16_and_invalid_worker_persistence() -> None:
    # This mixes numeric-capability checks with loader-policy checks to confirm
    # the diagnostics layer reports both hardware and config mismatches together.
    diagnostics = collect_runtime_diagnostics(
        requested_profile="local-cuda",
        resolved_profile="local-cuda",
        environment=build_runtime_environment(
            cuda_available=True,
            cuda_device_count=1,
            cuda_supports_bf16=False,
            is_apple_silicon=True,
            cpu_capability="AVX2",
        ),
        train_config=TrainConfig(accelerator="gpu", devices=1, precision="bf16-mixed"),
        data_config=DataConfig(
            dataset_url=None,
            num_workers=0,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=4,
        ),
        observability_config=ObservabilityConfig(),
    )

    codes = {diagnostic.code for diagnostic in diagnostics}
    assert "cuda_bf16_unavailable" in codes
    assert "persistent_workers_without_workers" in codes
    assert "prefetch_without_workers" in codes


def test_collect_runtime_diagnostics_flags_unsupported_cpu_bf16() -> None:
    # CPU BF16 support is environment-sensitive, so the diagnostics layer should
    # explain when a requested mixed-precision policy is not actually supported.
    diagnostics = collect_runtime_diagnostics(
        requested_profile="local-cpu",
        resolved_profile="local-cpu",
        environment=build_runtime_environment(cpu_supports_bf16=False),
        train_config=TrainConfig(accelerator="cpu", devices=1, precision="bf16-mixed"),
        data_config=DataConfig(
            dataset_url=None,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        ),
        observability_config=ObservabilityConfig(),
    )

    codes = {diagnostic.code for diagnostic in diagnostics}
    assert "cpu_bf16_unavailable" in codes


def test_format_runtime_diagnostics_includes_severity_code_and_suggestion() -> None:
    # Formatting matters because these diagnostics are shown directly to users
    # in CLI and notebook-facing workflows.
    rendered = format_runtime_diagnostics(
        (
            RuntimeDiagnostic(
                severity="warning",
                code="example",
                message="Example issue.",
                suggestion="Try an example fix.",
            ),
        )
    )

    assert "[WARNING] example: Example issue." in rendered
    assert "Try an example fix." in rendered
