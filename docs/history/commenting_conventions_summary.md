# Commenting And Documentation Conventions Summary

AI-assisted documentation note:
This summary was drafted with AI assistance and then reviewed/adapted for this
project. It documents the in-code commenting and docstring conventions used in
the recently updated implementation files and the later source-wide cleanup
pass.

## Goal

The goal of this documentation pass was not to add comments for their own sake.

The goal was to make the implementation easier to understand for future work by
ensuring that:

- important architectural decisions are explained
- runtime responsibilities are clear
- observability behavior is understandable without reverse-engineering every
  call site
- comments stay consistent across the files most recently changed

## Files Covered By This Convention Pass

- `src/config/`
- `defaults.py`
- `src/observability/`
- `src/train.py`
- `src/models/fused_model.py`
- `src/models/tft.py`
- `src/models/tcn.py`
- `src/models/grn.py`
- `src/models/nn_head.py`
- `src/data/`
- `src/utils/tft_utils.py`
- `main.py`

## Comment Style Standard

The current convention is:

- file-level maintenance notes and disclaimers use `#` comments
- class and dataclass descriptions use `"""..."""` docstrings
- function descriptions use `"""..."""` docstrings
- local implementation rationale stays in nearby `#` comments
- field-level explanations stay close to the fields they describe

This split was chosen because each comment type serves a different purpose.

In shorthand, the rule is now:

- top comments in a file: `#`
- class comments: `"""..."""`
- function comments: `"""..."""`
- inline and detailed walkthrough comments: `#`

## File-Level Comments

File-level comments are used for:

- maintenance notes
- AI-assisted implementation disclosures
- scope disclaimers
- high-level design intent

These remain as `#` comments because they behave more like module prefaces than
API documentation.

Examples include:

- repository-level design notes in `defaults.py`
- observability-scope notes in `src/observability/`
- entrypoint disclaimers in `main.py`

## Class And Function Docstrings

Class-level and function-level descriptions were standardized into proper
docstrings.

The preferred structure is now:

- a short opening description
- `Purpose:`
- `Context:`
- optionally another focused heading such as:
  - `Responsibility boundary:`
  - `Architecture overview:`
  - `Operational role:`
  - `Important disclaimer:`
  - `Important note:`

### Why this structure was chosen

The codebase benefits from comments that are more detailed than one-line
docstrings, but banner-style comment blocks above every class make the files
harder to scan and less consistent with Python tooling.

Using docstrings for class/function-level explanations gives:

- cleaner API-style structure
- better discoverability in editors
- enough room for detailed context
- a clear distinction from line-by-line implementation notes

## Inline Comments

Inline `#` comments are still used heavily where they add value, especially
for:

- explaining why a helper exists
- clarifying design tradeoffs
- documenting Lightning-specific behavior
- noting best-effort or dependency-sensitive behavior
- recording why a fallback exists

These comments are intentionally kept close to the code they explain so the
reader does not have to jump elsewhere to understand a critical design choice.

## Wording Convention

During review, the phrase `Why this exists` was replaced with `Context` because
it reads more naturally while still communicating the same information.

The headings now tend to use:

- `Purpose`
- `Context`
- `Operational role`
- `Architecture overview`
- `Responsibility boundary`
- `Important note`
- `Important disclaimer`

This gives the comments a more natural tone without making them vague.

## What The Comments Now Emphasize

The updated comments and docstrings intentionally focus on:

- responsibility boundaries between model, data, trainer, and entrypoint code
- the difference between semantic model config and runtime observability policy
- how TensorBoard, torchview, telemetry, prediction export, and reports fit
  together
- why the training wrapper prepares and binds runtime config before model
  construction
- why the fused model owns optimizer/loss behavior while the trainer wrapper
  owns orchestration
- how tensor-heavy paths move grouped feature streams, latent branch features,
  and quantile outputs through the model stack

## What The Comments Intentionally Avoid

The comments try to avoid:

- repeating the code in plain English when the code is already obvious
- vague filler comments like "set variable" or "do processing"
- overly short docstrings that add no real context
- giant prose blocks that obscure the implementation

The aim is detailed but purposeful documentation, not just volume.

## Practical Outcome

After this pass, the main changed implementation files now have a more uniform
documentation style:

- detailed enough for onboarding and later maintenance
- structured enough to scan quickly
- explicit about design intent and runtime behavior
- consistent across config, observability, training, model, and entrypoint
  surfaces

The later cleanup pass also standardized the source tree structurally:

- file-level notes in `src/` now use `#` comments rather than module docstrings
- every top-level class in `src/` now has a docstring
- every top-level function in `src/` now has a docstring
- dense `#` walkthrough comments are still used freely in the complex
  tensor-heavy files

## Important Nuance

The convention is not meant to suppress detail.

In this project, dense inline `#` comments are considered valuable when they
explain:

- tensor shapes
- grouped feature semantics
- broadcasting behavior
- fusion boundaries
- Trainer lifecycle behavior
- observability callback flow

That is why files like:

- `src/models/tft.py`
- `src/models/fused_model.py`
- `src/train.py`
- `src/observability/`

remain more heavily annotated than the simpler modules.

This should make future work easier when the project returns to:

- deeper probabilistic diagnostics
- fail-fast debugging callbacks
- Colab packaging
- further report generation
- additional model instrumentation
