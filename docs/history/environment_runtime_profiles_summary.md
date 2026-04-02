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
- `docs/history/environment_runtime_profiles_summary.md`

Files updated:

- `main.py`
- `main.ipynb`
- `README.md`
- `src/config/__init__.py`
- `tests/test_config.py`
- `tests/test_main.py`

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
4. `main.py` and `main.ipynb`
   call that shared layer before constructing the DataModule or trainer

This preserves an important ownership boundary:

- `src/config/` still owns typed config contracts
- `src/environment/` now owns runtime-environment interpretation and
  diagnostics
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
