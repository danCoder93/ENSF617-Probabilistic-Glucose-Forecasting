# Limitations And Future Work

The present report documents a functioning probabilistic glucose forecasting pipeline, but the saved experiment should be interpreted as a baseline rather than a finished study. The main limitations of the current evidence are listed below together with the next steps that would most directly strengthen the project.

## 7.1 Scope Of The Present Experimental Setting

The recorded run used `max_epochs = 1`, so the reported metrics reflect an initial trained baseline rather than a carefully optimized model. This point is important because some of the observed weaknesses may come from limited fitting time rather than from fundamental flaws in the architecture itself. A stronger study would need a controlled training sweep with repeated runs, longer training, and explicit checkpoint comparison.

## 7.2 Hyperparameter Tuning And Regularization

Because the training budget was minimal, hyperparameter tuning is the clearest immediate opportunity. Future experiments should vary learning rate, batch size, dropout, weight decay, and stopping policy in a systematic way. The existing artifact pipeline already records enough information to support that kind of study; what is missing is the breadth of experiments, not the ability to inspect them afterward.

## 7.3 Horizon Wise And Range Wise Performance Limits

The current model degrades in two places that matter most: late forecast steps and low glucose states. Coverage falls steadily across the one hour horizon, which indicates that uncertainty estimates do not scale properly with forecast difficulty. Performance below `70 mg/dL` is especially concerning because the model shows both high error and strong positive bias. Future work should therefore focus on low glucose sensitivity and horizon dependent calibration instead of chasing a lower average MAE alone.

## 7.4 Generalization Beyond Known Subjects

The evaluation protocol is within subject and chronological, which is a reasonable design for personalized forecasting. However, it does not answer how well the model transfers to unseen individuals, and that question becomes more important because subject identity is included as a static feature. A subject held out benchmark is therefore a necessary complement to the current split, even if personalized forecasting remains the primary use case.

## 7.5 Deferred Analyses And Additional Future Directions

Several concrete extensions would improve the scientific value of the report. The most important are baseline comparison against simpler forecasters, ablation against TCN only and TFT only variants, richer calibration analysis, and more informative features such as time since meal, time since bolus, insulin on board, and carbohydrate on board approximations. These additions would make it easier to determine whether gains come from the fused architecture itself or from the broader data and reporting pipeline around it.
