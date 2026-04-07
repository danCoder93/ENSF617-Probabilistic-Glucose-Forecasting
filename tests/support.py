"""
AI-assisted implementation note:
This helper module was drafted with AI assistance and then reviewed/adapted for
this project. It provides typed helpers for the refactored AZT1D data tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from config import Config, DataConfig, TCNConfig, TFTConfig
from environment import RuntimeEnvironment


# These Protocols describe the callable fixtures provided by `conftest.py`.
# Giving them names has two benefits:
# 1. Pylance can understand fixture return shapes inside test functions.
# 2. The tests read more like documentation because the fixture roles are named
#    after what they do rather than appearing as untyped callables.
class WriteProcessedCsv(Protocol):
    """
    Callable fixture contract for building small processed data fixtures.

    Context:
    many tests need the same kind of synthetic processed CSV but with slightly
    different subject counts, sequence lengths, or gap patterns. Naming the
    callable contract here makes those tests easier to type-check and easier to
    read as "test data builders" rather than opaque fixtures.
    """

    def __call__(
        self,
        *,
        filename: str = "processed.csv",
        subject_ids: tuple[str, ...] = ("subject_a",),
        steps_per_subject: int = 12,
        gap_after_step: int | None = None,
    ) -> Path:
        """Write one processed CSV fixture and return its filesystem path."""
        ...


class BuildDataConfig(Protocol):
    """
    Callable fixture contract for producing temp-directory-backed DataConfig objects.

    Context:
    tests often want to override only one or two DataConfig fields while
    inheriting the same safe temp-directory defaults from `conftest.py`.
    """

    def __call__(self, processed_csv_path: Path, **overrides: Any) -> DataConfig:
        """Build a `DataConfig` rooted entirely in pytest-managed temp folders."""
        ...


def build_minimal_data_config(**overrides: Any) -> DataConfig:
    """
    Build the smallest valid `DataConfig` for focused unit tests.

    Context:
    many tests do not care about real filesystem layout or dataloader tuning.
    Starting from these intentionally conservative defaults keeps those tests
    short while still constructing the production dataclass.
    """
    defaults: dict[str, Any] = {
        "dataset_url": None,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    }
    defaults.update(overrides)
    return DataConfig(**defaults)


def build_base_config(data_config: DataConfig) -> Config:
    """
    Pair a prepared `DataConfig` with lightweight default model configs.

    Context:
    test modules that are not about architecture tuning usually just need a
    coherent top-level `Config` object whose model branches are small enough to
    initialize quickly.
    """
    return Config(
        data=data_config,
        tft=TFTConfig(),
        tcn=TCNConfig(num_inputs=1, num_channels=(4,), dilations=(1,)),
    )


def build_runtime_environment(**overrides: Any) -> RuntimeEnvironment:
    """
    Build a deterministic runtime-environment snapshot for environment tests.

    Context:
    the environment package reasons about many hardware and dependency flags at
    once. Using one canonical synthetic environment here keeps tests explicit
    about which capability they are overriding instead of rebuilding the full
    structure inline every time.
    """
    base = RuntimeEnvironment(
        platform="test-platform",
        system="Linux",
        release="test-release",
        machine="x86_64",
        is_apple_silicon=False,
        python_version="3.12.0",
        cpu_count_logical=8,
        cpu_count_physical=4,
        system_memory_gb=16.0,
        cpu_capability="AVX2",
        cpu_supports_bf16=False,
        is_colab=False,
        is_slurm=False,
        torch_available=True,
        pytorch_lightning_available=True,
        tensorboard_available=True,
        torchview_available=True,
        torch_version="2.0.0",
        accelerator_api_available=False,
        accelerator_available=False,
        accelerator_type=None,
        accelerator_device_count=0,
        cuda_available=False,
        cuda_device_count=0,
        cuda_device_name=None,
        cuda_capability=None,
        cuda_supports_bf16=False,
        cuda_visible_devices=None,
        mps_built=False,
        mps_available=False,
        slurm_job_id=None,
        slurm_cpus_per_task=None,
        slurm_gpus=None,
        slurm_detected_by_lightning=False,
    )
    payload = dict(base.__dict__)
    payload.update(overrides)
    return RuntimeEnvironment(**payload)
