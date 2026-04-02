from __future__ import annotations

# AI-assisted maintenance note:
# This module detects the active execution environment and backend surfaces.
# It is intentionally separate from profile resolution and diagnostics so the
# repo has one clear place that answers the question:
# "what environment are we actually running in?"
#
# Responsibility boundary:
# - inspect Python modules, backend APIs, and environment variables
# - normalize the observed runtime into one `RuntimeEnvironment` object
#
# What does *not* live here:
# - policy such as "which profile should we choose?"
# - validation such as "is this config compatible with the environment?"
# - user-facing error classification
#
# Important disclaimer:
# environment detection is best-effort by design. A failed backend probe should
# degrade to "less information available" rather than crash the CLI or
# notebook before diagnostics can even be reported.

import importlib.util
import os
import platform
import sys
from typing import Any, Mapping

from environment.types import RuntimeEnvironment


# ============================================================================
# Small Detection Helpers
# ============================================================================
# These helpers are intentionally tiny and local to the detection layer. They
# keep the main detection function readable without turning basic stdlib probes
# into package-level public API.
def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_apple_silicon(system: str, machine: str) -> bool:
    return system == "Darwin" and machine in {"arm64", "arm64e"}


def _optional_bool_probe(namespace: Any, attribute_name: str) -> bool | None:
    probe = getattr(namespace, attribute_name, None)
    if not callable(probe):
        return None
    try:
        return bool(probe())
    except Exception:
        return None


def _cpu_supports_bf16(
    cpu_capability: str | None,
    *,
    torch_module: Any | None = None,
) -> bool:
    if torch_module is not None:
        cpu_namespace = getattr(torch_module, "cpu", None)
        c_namespace = getattr(torch_module, "_C", None)
        for namespace, attribute_name in (
            (cpu_namespace, "_is_avx512_bf16_supported"),
            (cpu_namespace, "_is_amx_tile_supported"),
            (c_namespace, "_cpu_supports_avx512_bf16"),
            (c_namespace, "_cpu_supports_amx_tile"),
        ):
            if namespace is None:
                continue
            probed = _optional_bool_probe(namespace, attribute_name)
            if probed is not None:
                return probed

    if cpu_capability is None:
        return False
    normalized = cpu_capability.upper()
    return "BF16" in normalized or "AMX" in normalized


def detect_runtime_environment(
    environ: Mapping[str, str] | None = None,
) -> RuntimeEnvironment:
    """
    Detect the current runtime environment and available backend surfaces.

    Context:
    this is the package entrypoint for runtime probing. It intentionally
    returns a plain dataclass summary rather than raw backend objects so the
    rest of the codebase can reason about runtime state without depending on
    torch or Lightning internals directly.
    """

    env = os.environ if environ is None else environ
    system = platform.system()
    machine = platform.machine()

    torch_available = _module_available("torch")
    pytorch_lightning_available = _module_available("pytorch_lightning")
    tensorboard_available = _module_available("tensorboard")
    torchview_available = _module_available("torchview")
    is_apple_silicon = _is_apple_silicon(system, machine)
    cpu_count_logical = os.cpu_count()
    cpu_count_physical: int | None = None
    system_memory_gb: float | None = None
    cpu_capability: str | None = None
    cpu_supports_bf16 = False

    torch_version: str | None = None
    accelerator_api_available = False
    accelerator_available = False
    accelerator_type: str | None = None
    accelerator_device_count = 0
    cuda_available = False
    cuda_device_count = 0
    cuda_device_name: str | None = None
    cuda_capability: str | None = None
    cuda_supports_bf16 = False
    mps_built = False
    mps_available = False
    slurm_detected_by_lightning = False

    if _module_available("psutil"):
        try:
            import psutil

            cpu_count_physical = psutil.cpu_count(logical=False)
            if cpu_count_logical is None:
                cpu_count_logical = psutil.cpu_count(logical=True)
            system_memory_gb = round(
                psutil.virtual_memory().total / (1024.0**3),
                2,
            )
        except Exception:
            pass

    # PyTorch probing is kept behind the import check so diagnostics-only flows
    # remain usable in environments where the training stack is not yet
    # installed.
    if torch_available:
        try:
            import torch

            torch_version = getattr(torch, "__version__", None)
            cpu_backend = getattr(getattr(torch, "backends", None), "cpu", None)
            if cpu_backend is not None and hasattr(cpu_backend, "get_cpu_capability"):
                cpu_capability = str(cpu_backend.get_cpu_capability())
                cpu_supports_bf16 = _cpu_supports_bf16(
                    cpu_capability,
                    torch_module=torch,
                )
            accelerator_module = getattr(torch, "accelerator", None)
            if accelerator_module is not None and all(
                hasattr(accelerator_module, attribute_name)
                for attribute_name in (
                    "is_available",
                    "device_count",
                    "current_accelerator",
                )
            ):
                accelerator_api_available = True
                # Prefer the generic accelerator API when it exists so newer
                # torch versions can describe the active backend through one
                # common surface instead of forcing this repository to branch on
                # every backend family directly.
                accelerator_available = bool(accelerator_module.is_available())
                accelerator_device_count = int(accelerator_module.device_count())
                current_accelerator = accelerator_module.current_accelerator(
                    check_available=True
                )
                if current_accelerator is not None:
                    accelerator_type = str(getattr(current_accelerator, "type", None))

            cuda_available = bool(torch.cuda.is_available())
            cuda_device_count = int(torch.cuda.device_count())
            if cuda_available and cuda_device_count > 0:
                current_device = int(torch.cuda.current_device())
                cuda_device_name = str(
                    torch.cuda.get_device_name(current_device)
                )
                capability = torch.cuda.get_device_capability(current_device)
                cuda_capability = f"{capability[0]}.{capability[1]}"
                cuda_supports_bf16 = bool(
                    getattr(torch.cuda, "is_bf16_supported", lambda: False)()
                )

            mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
            if mps_backend is not None:
                mps_built = bool(getattr(mps_backend, "is_built", lambda: False)())
                mps_available = bool(
                    getattr(mps_backend, "is_available", lambda: False)()
                )

            # Backend-specific probes remain useful because they expose richer
            # detail than the generic accelerator API alone, especially for
            # CUDA device naming and MPS build-vs-available distinctions.
            if accelerator_type == "cuda":
                cuda_available = accelerator_available
                cuda_device_count = accelerator_device_count
            if accelerator_type == "mps":
                mps_available = accelerator_available
        except Exception:
            # Detection should never hard-fail the runtime path. If one probe
            # breaks because of a partial install or backend quirk, the
            # diagnostics layer can still report the missing information later.
            pass

    # Lightning's SLURMEnvironment is treated as an optional additional signal.
    # The repository still falls back to raw `SLURM_*` environment variables so
    # this code stays compatible with partial Lightning installs and simpler
    # local tests.
    if pytorch_lightning_available:
        for import_path in (
            "lightning.pytorch.plugins.environments",
            "pytorch_lightning.plugins.environments",
        ):
            try:
                module = __import__(import_path, fromlist=["SLURMEnvironment"])
                slurm_environment = getattr(module, "SLURMEnvironment", None)
                if slurm_environment is not None:
                    slurm_detected_by_lightning = bool(slurm_environment.detect())
                    break
            except Exception:
                continue

    return RuntimeEnvironment(
        platform=platform.platform(),
        system=system,
        release=platform.release(),
        machine=machine,
        is_apple_silicon=is_apple_silicon,
        python_version=platform.python_version(),
        cpu_count_logical=cpu_count_logical,
        cpu_count_physical=cpu_count_physical,
        system_memory_gb=system_memory_gb,
        cpu_capability=cpu_capability,
        cpu_supports_bf16=cpu_supports_bf16,
        is_colab=(
            "google.colab" in sys.modules
            or "COLAB_GPU" in env
            or "COLAB_RELEASE_TAG" in env
        ),
        is_slurm=(
            "SLURM_JOB_ID" in env
            or "SLURM_JOBID" in env
            or slurm_detected_by_lightning
        ),
        torch_available=torch_available,
        pytorch_lightning_available=pytorch_lightning_available,
        tensorboard_available=tensorboard_available,
        torchview_available=torchview_available,
        torch_version=torch_version,
        accelerator_api_available=accelerator_api_available,
        accelerator_available=accelerator_available,
        accelerator_type=accelerator_type,
        accelerator_device_count=accelerator_device_count,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        cuda_capability=cuda_capability,
        cuda_supports_bf16=cuda_supports_bf16,
        cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"),
        mps_built=mps_built,
        mps_available=mps_available,
        slurm_job_id=env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID"),
        slurm_cpus_per_task=_optional_int(env.get("SLURM_CPUS_PER_TASK")),
        slurm_gpus=env.get("SLURM_GPUS"),
        slurm_detected_by_lightning=slurm_detected_by_lightning,
    )
