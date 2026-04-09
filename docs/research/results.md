# Results

All values reported below come from the saved run under `artifacts/main_run/`, time stamped April 7, 2026 in [`../../artifacts/main_run/run_summary.json`](../../artifacts/main_run/run_summary.json). The shared evaluation report covers `28,234` test windows and `338,808` forecast rows across `24` subjects. Subject `17` is present in the cleaned data summary but does not appear in the saved prediction table, so the grouped evaluation summaries cover `24` subjects rather than the full cohort of `25`.

## 5.1 Reported Run Configuration

The saved run used the same data framing described earlier: a `70 / 15 / 15` within subject chronological split, an encoder length of `168`, a prediction length of `12`, and the quantile set `(0.1, 0.5, 0.9)`. The recorded batch size was `128`, the optimizer was `Adam` with learning rate `0.001`, and `max_epochs` was set to `1`. The results below should therefore be read as the performance of the recorded baseline run, not as the result of a longer tuning study.

## 5.2 Overall Held Out Performance

Across all forecast rows, the fused model achieved an MAE of `20.17 mg/dL` and an RMSE of `27.21 mg/dL`. The overall bias was `+12.98 mg/dL`, which shows a clear tendency to overpredict glucose on average. On the probabilistic side, the model achieved an overall pinball loss of `6.67`, a mean interval width of `58.87 mg/dL`, and empirical coverage of `0.761` for the nominal central `80%` interval. In aggregate, the model therefore produced intervals that were reasonably broad but still somewhat undercovered.

Table 1 provides a compact summary of the main results from the saved run.

| Overall Accuracy (MAE / RMSE) | Probabilistic Quality | Bias | Temporal Degradation | Clinical Regime Performance | Subject Variation | Summary Interpretation |
|---|---|---|---|---|---|---|
| `20.17 / 27.21 mg/dL` | Pinball loss `6.67`; empirical coverage `0.761` against nominal `0.80`; mean interval width `58.87 mg/dL` | `+12.98 mg/dL` | MAE rises from `12.57 mg/dL` at horizon `0` to `28.35 mg/dL` at horizon `11`; coverage falls from `0.964` to `0.585` | Best in `70 to 180 mg/dL`; performance weakens above `180 mg/dL` and deteriorates sharply below `70 mg/dL`, where MAE reaches `39.05 mg/dL` and coverage falls to `0.202` | Lowest subject MAE: Subject `13` at `13.86 mg/dL`; highest: Subject `5` at `40.91 mg/dL`, though that subject contributes only `96` forecast rows | Useful short range baseline with meaningful structure, but undercoverage, positive bias, and poor low glucose behavior limit stronger claims |

## 5.3 Probabilistic Forecast Quality

The quantile forecasts are most informative when read together with the global coverage and width values. The central interval was wide enough to span almost `59 mg/dL` on average, yet coverage still fell below the nominal `0.80` target. This gap indicates that the uncertainty bands were informative, but not fully calibrated, over the full test partition.

## 5.4 Performance Across Forecast Horizon

Error grew steadily across the `12` forecast steps. At horizon `0`, MAE was `12.57 mg/dL` and RMSE was `16.66 mg/dL`. By horizon `11`, which corresponds to the full `60` minute forecast limit, MAE rose to `28.35 mg/dL` and RMSE rose to `36.06 mg/dL`. Empirical coverage declined from `0.964` at horizon `0` to `0.585` at horizon `11`. Mean interval width increased only modestly across the same span, from `58.11 mg/dL` to `60.20 mg/dL`.

Detailed horizon tables are available in [`../../artifacts/main_run/reports/artifacts/shared_report/tables/by_horizon.csv`](../../artifacts/main_run/reports/artifacts/shared_report/tables/by_horizon.csv).

## 5.5 Performance Across Glucose Range And Subject

The model performed best in the clinically common `70 to 180 mg/dL` range, where MAE was `19.35 mg/dL` and RMSE was `26.37 mg/dL`. Performance degraded above `180 mg/dL`, where MAE rose to `22.22 mg/dL`, and it deteriorated substantially below `70 mg/dL`, where MAE reached `39.05 mg/dL` and empirical coverage fell to `0.202`. The low glucose range also showed a bias of `+39.04 mg/dL`, indicating strong overprediction in the regime where errors are most clinically concerning.

Subject level performance also varied meaningfully. The lowest MAE was observed for Subject `13` at `13.86 mg/dL`. The highest MAE was observed for Subject `5` at `40.91 mg/dL`, although that subject contributed only `96` forecast rows and should therefore be interpreted cautiously. The full subject table is available in [`../../artifacts/main_run/reports/artifacts/shared_report/tables/by_subject.csv`](../../artifacts/main_run/reports/artifacts/shared_report/tables/by_subject.csv).

## 5.6 Checkpoint Selection

Because the recorded run used `max_epochs = 1`, the best checkpoint and the final checkpoint both correspond to epoch `00`. This point matters for interpretation: the current report documents a consistent baseline run, but it does not yet test how the architecture behaves under a longer training schedule.
