# Environment Runtime Profiles And Diagnostics Summary

AI-assisted documentation note:
This summary was drafted with AI assistance and then reviewed/adapted for this
project. It documents the pass that introduced environment-aware runtime
profiles, preflight diagnostics, and the dedicated `src/environment/` package.

This document summarizes the work that added:

- a reusable runtime-environment detection layer
- explicit device-profile selection for local, Colab, Slurm, CUDA, CPU, and
  Apple Silicon workflows
- preflight diagnostics and failure analysis for environment-sensitive runs
- notebook/script alignment around one shared environment-aware workflow
- a source-layout refinement that moved this logic out of `src/config/` into
  `src/environment/`

Later follow-up note:
the environment layer was later expanded beyond profile selection and
diagnostics. It now also includes a dedicated runtime tuning module, backend-
aware compile defaults, CPU BF16 detection, Apple Silicon MPS allocator
overrides, a short benchmark-only workflow, and several robustness fixes around
compile fallback and direct workflow usage outside the CLI.

## Goals

- Make the repository easier to run across different execution environments
  without rewriting core model or data code.
- Keep environment-specific runtime policy explicit and inspectable rather than
  burying it in ad hoc CLI branches.
- Detect likely environment/setup problems early and report them in a more
  actionable way than a raw stack trace alone.
- Preserve the repository's existing responsibility split by keeping model and
  data semantics separate from runtime-environment concerns.
- Keep `main.py` and `main.ipynb` aligned on one shared runtime flow.

## Files Added or Updated

Files added:

- `src/environment/__init__.py`
- `src/environment/types.py`
- `src/environment/detection.py`
- `src/environment/profiles.py`
- `src/environment/diagnostics.py`
- `src/environment/tuning.py`
- `docs/history/environment_runtime_profiles_summary.md`

Files updated:

- `defaults.py`
- `main.py`
- `main.ipynb`
- `README.md`
- `src/config/__init__.py`
- `src/config/data.py`
- `src/config/runtime.py`
- `src/train.py`
- `tests/test_config.py`
- `tests/test_main.py`
- `tests/test_train.py`

Files removed:

- `src/config/environment.py`

## Main Architectural Addition

The repository now has a dedicated runtime-environment layer:

1. `src/environment/detection.py`
   detects where the run is happening and which accelerators/backends are
   available
2. `src/environment/profiles.py`
   resolves high-level device profiles into concrete runtime defaults
3. `src/environment/diagnostics.py`
   validates the requested setup and explains likely environment failures
4. `src/environment/tuning.py`
   applies low-level backend tuning once policy has already been chosen
5. `main.py` and `main.ipynb`
   call that shared layer before constructing the DataModule or trainer

This preserves an important ownership boundary:

- `src/config/` still owns typed config contracts
- `src/environment/` now owns runtime-environment interpretation and
  diagnostics plus backend tuning
- `src/train.py` still owns Lightning orchestration

## `src/environment/`

The environment package was introduced because the old placement under
`src/config/` was no longer the right abstraction boundary.

The code had grown beyond pure config shaping. It was now responsible for:

- backend detection
- Colab and Slurm interpretation
- auto-profile selection
- compatibility validation
- failure explanation

Those are operational runtime concerns, not configuration definitions.

Moving the layer into `src/environment/` makes the architecture clearer:

- config describes intended contracts
- environment logic interprets the current machine/runtime
- tuning logic applies the chosen runtime policy to backend surfaces
- training code consumes the resolved result

### Detection strategy

The new detection path uses:

- PyTorch's generic accelerator API when available
- CUDA and MPS backend-specific checks as fallbacks or detail sources
- Lightning's `SLURMEnvironment.detect()` when available
- environment-variable checks for cluster/notebook context

This combination gives the repo a more robust picture of:

- backend availability
- device count
- notebook versus local execution
- cluster versus non-cluster execution

Later follow-up detection additions include:

- explicit Apple Silicon detection
- logical and physical CPU counts
- system memory reporting
- CUDA capability strings
- CUDA BF16 support reporting
- coarse CPU capability and CPU BF16 support reporting

## Device Profiles

The pass introduced a high-level `--device-profile` flag.

Supported profiles now include:

- `auto`
- `local-cpu`
- `local-cuda`
- `colab-cpu`
- `colab-cuda`
- `slurm-cpu`
- `slurm-cuda`
- `apple-silicon`

### Why profiles were needed

The repository already had low-level runtime flags such as accelerator,
devices, precision, and loader settings, but it did not have one environment-
aware way to choose sensible defaults for the most common execution contexts.

The new design keeps both layers:

- high-level profiles for repeatable workflows
- low-level overrides for one-off control

The precedence rule is:

- explicit CLI or notebook override
- then profile default
- then baseline repository default

That rule matters because it keeps `auto` useful without making it impossible
to force a specific setup such as Colab CUDA or Apple Silicon.

### Later profile-default expansion

The profile layer later became more throughput-aware rather than only
backend-aware.

Examples of defaults that are now profile-sensitive include:

- `bf16-mixed` versus `16-mixed` selection for CUDA depending on reported BF16
  support
- CPU `bf16-mixed` selection on CPUs that report BF16-capable support
- worker-count heuristics derived from host CPU counts rather than only fixed
  constants
- backend-aware `torch.compile` defaults for local CPU and CUDA profiles
- CUDA `matmul_precision`, TF32, and cuDNN benchmark defaults
- Apple Silicon MPS allocator/fallback defaults
- CPU and Apple Silicon Torch thread-count defaults
- DataLoader `prefetch_factor` defaults that vary by backend and worker count

## Preflight Diagnostics And Failure Analysis

Another major addition was the diagnostics path.

Before this work, environment-related failures were much harder to classify.
The runtime would often fail only after reaching deeper Trainer or backend
construction code.

The repository can now:

- run diagnostics before training starts
- classify common environment-related failures
- record detected runtime metadata in the run summary
- run a diagnostics-only entry flow without launching full training

Examples of issues this layer now tries to surface more clearly include:

- missing `torch` or Lightning dependencies
- requesting GPU acceleration without CUDA availability
- requesting MPS without Apple Silicon backend support
- suspicious worker or pin-memory settings for the current environment
- likely Slurm allocation mismatches
- unsupported CUDA or CPU BF16 requests
- MPS compile-mode caution
- prefetch settings that do not make sense when `num_workers=0`
- deterministic-plus-cuDNN-benchmark mismatches

Important disclaimer:
the diagnostics are best-effort heuristics. They improve error quality, but
they do not claim to be a perfect root-cause engine for every runtime failure.

## `main.py` And `main.ipynb` Follow-Up

The entry surfaces now share an expanded runtime flow:

1. build baseline configs
2. detect the current runtime environment
3. resolve the requested device profile
4. apply explicit overrides
5. run preflight diagnostics
6. continue into the normal training/evaluation workflow

This means the notebook did not gain a separate environment policy path. It
uses the same shared helpers as the script entrypoint.

The top-level workflow now also records additional runtime metadata such as:

- requested and resolved device profile
- applied profile defaults
- detected runtime environment
- preflight diagnostics
- runtime tuning actions that were applied or skipped

Later follow-up work also added:

- a short benchmark-only workflow for comparing environments without running
  the full test/predict/report stack
- benchmark memory reporting for CUDA, MPS, and process RSS where available
- early Apple Silicon environment-variable bootstrapping so MPS allocator
  overrides take effect before Torch import
- workflow-side profile resolution so notebook or direct Python callers receive
  the same environment defaults as the CLI path

## Runtime Tuning Layer

The later `src/environment/tuning.py` addition formalized a previously missing
last-mile step:

- profile resolution decides the runtime policy
- diagnostics validate that policy against the host
- tuning applies the backend-level knobs that make the policy real

That tuning layer currently handles:

- MPS environment-variable overrides
- float32 matmul precision
- CUDA TF32 controls
- cuDNN benchmark policy
- Torch intra-op and inter-op thread counts
- optional model compilation through `torch.compile(...)`

This matters because environment-aware policy is only partially useful if it
never reaches the backend knobs that actually influence throughput or
compatibility.

## Benchmark-Only Workflow

One later follow-up also added a small environment-focused benchmark mode to
the top-level entrypoint.

That workflow is intentionally narrow:

- run a tiny fit-only loop
- disable most observability extras
- skip held-out test/predict/report generation
- write a compact `benchmark_summary.json`

The goal is not to replace a real profiling suite. The goal is to make it easy
to compare local CPU, CUDA, and Apple Silicon runs using one shared repository
surface.

## Robustness Follow-Up

The final environment-specific follow-up focused on robustness rather than on
adding more knobs.

That pass addressed several important gaps:

- direct `run_training_workflow(...)` callers now receive profile resolution,
  not just the CLI path
- `torch.compile(...)` failures now fall back to eager execution rather than
  aborting the full run before training starts
- benchmark reporting now records requested versus actual train-batch counts
  when the trainer exposes that information
- CPU BF16 detection became more conservative by preferring backend-specific
  Torch probes when available

## Why This Pass Mattered

This work improved two things at once:

- portability across local and hosted execution environments
- debuggability when environment assumptions are wrong

That is especially important for this repository because it is meant to be run
in several different contexts:

- local development
- notebooks and Colab
- CPU-only systems
- CUDA systems
- Apple Silicon laptops
- Slurm-managed cluster jobs

Without a dedicated environment layer, those contexts would continue to leak
into ad hoc conditionals scattered across entrypoint code.

## Lasting Impact

This pass established a new stable architectural idea in the repository:

- environment interpretation is a first-class runtime subsystem

That subsystem now sits alongside the existing major layers:

- config
- data
- models
- training orchestration
- observability
- evaluation

It also reinforced an existing team principle:

- internal package cleanup is worthwhile when it preserves the outer entry
  surface while making responsibilities clearer inside

`main.py` and `main.ipynb` remain the public entry surfaces. The new
`src/environment/` package exists to keep those entry surfaces thin while
still making environment-aware behavior explicit.
