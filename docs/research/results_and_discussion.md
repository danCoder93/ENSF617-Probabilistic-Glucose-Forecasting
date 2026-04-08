# Results And Discussion

Role: Canonical results-and-discussion section for the research companion.
Audience: Researchers and collaborators translating current artifact evidence
into manuscript-ready findings later.
Owns: Results framing, interpretation guidance, and current evidence paths.
Related docs: [`materials_and_methods.md`](materials_and_methods.md),
[`conclusions.md`](conclusions.md),
[`../artifact_diagnosis.md`](../artifact_diagnosis.md),
[`../inspiration/paper_material_notes.md`](../inspiration/paper_material_notes.md).

## Status

This section is intentionally placeholder-friendly. The repository already
produces meaningful artifact evidence, but the final paper-style results and
discussion narrative should be human-authored.

## Current Evidence Sources

Today, the strongest repository-native evidence lives in:

- `run_summary.json`
- `report_index.json`
- `metrics_summary.json`
- grouped evaluation tables
- exported prediction artifacts
- the case-study interpretation in [`../artifact_diagnosis.md`](../artifact_diagnosis.md)

## Discussion Framing

The discussion section should eventually connect:

- headline quantitative results
- horizon-wise and subject-wise behavior
- uncertainty interpretation
- failure modes and limitations
- implications for future model or data changes

## Interim Reading Path

Until the final paper narrative is written, use:

1. [`../artifact_diagnosis.md`](../artifact_diagnosis.md)
2. [`../inspiration/paper_material_notes.md`](../inspiration/paper_material_notes.md)
3. the current artifact outputs under `artifacts/main_run/`
