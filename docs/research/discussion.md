# Discussion

## 6.1 Reading The Quantile Forecast

The three quantile outputs summarize one forecast distribution rather than three separate targets. The `0.5` quantile serves as the central forecast used for MAE and RMSE, while the `0.1` and `0.9` quantiles define the central `80%` interval. If the model is well calibrated, realized glucose values should fall inside that interval about four times out of five.

## 6.2 High Level Interpretation Of The Final Run

At a high level, the saved run shows that the fused TCN and TFT pipeline learns nontrivial structure from glucose, insulin, carbohydrate, device, and calendar signals. The earliest horizons are reasonably accurate, with MAE close to `12 mg/dL`, which suggests that the model is capturing short range temporal behavior rather than producing only coarse averages.

At the same time, the overall numbers do not support an overly strong claim. Global MAE remains `20.17 mg/dL`, overall coverage is `0.761` rather than the nominal `0.80`, and the bias of `+12.98 mg/dL` shows that predictions tend to sit above the truth. The current evidence therefore supports the architecture as a working baseline, not as a mature forecasting system.

## 6.3 Horizon Wise Error Growth And Uncertainty

The horizon level results are the clearest sign of where the current model fails. Error rises steadily from horizon `0` to horizon `11`, which is expected for a one hour forecast task. More revealing is the joint behavior of interval width and coverage. Coverage falls sharply from `0.964` at the first step to `0.585` at the last step, while mean interval width increases only slightly. In other words, the uncertainty bands do not widen enough to keep pace with the growing difficulty of longer horizon prediction.

This pattern suggests that the main issue is not just point error. It is a combination of point error and horizon dependent undercoverage. A better model would need either sharper temporal representations for later steps, more informative future context, or a learning objective that encourages wider and better calibrated intervals when forecast uncertainty increases.

## 6.4 Hypoglycemia As The Main Weakness

The most serious weakness appears below `70 mg/dL`. In that range, MAE reaches `39.05 mg/dL`, bias reaches `+39.04 mg/dL`, and empirical coverage drops to `0.202`. This is not a small deterioration at the edge of the distribution. It is a systematic failure to track low glucose states and to express adequate uncertainty when those states occur.

From a clinical perspective, this is the least acceptable failure mode in the current report. Overpredicting glucose during hypoglycemia can hide risk at the moment when early warning matters most. Any next stage of model development should treat low glucose performance as a primary target rather than as a secondary metric.

## 6.5 What Is Strong About The Current Design

Even with the limitations above, several aspects of the current design are worth keeping. The data contract is clear about which variables are static, known in advance, or observed only in history. That separation reduces leakage risk and makes the forecasting problem easier to reason about. The late fusion architecture is also methodologically coherent: the TCN branches and the TFT branch meet only after each has formed its own latent view of the sequence.

The reporting stack is another strength. Because predictions, grouped metrics, and run summaries are saved explicitly, the present weaknesses are visible in enough detail to guide further work instead of being hidden behind one average test loss value.

## 6.6 What Still Limits The Current Claims

The strongest claims are limited by the evaluation setting and by the training budget of the saved run. The split is within subject rather than subject held out, and subject identity is part of the default feature contract, so the results do not establish generalization to unseen individuals. In addition, the reported run used only one epoch. That makes the current numbers informative as an execution baseline, but not as an upper bound on what the architecture can do.

The hybrid design also remains a hypothesis rather than a proven advantage. It is easy to motivate why TCN and TFT should complement one another. It is harder to show that the fused model improves on strong TCN only, TFT only, or persistence baselines. That comparison remains necessary before the added complexity can be defended empirically.
