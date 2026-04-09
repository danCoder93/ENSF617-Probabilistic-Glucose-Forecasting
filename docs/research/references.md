# References

The numbered citations used throughout this report are listed below. Local copies of the cited papers are preserved under `docs/publications/`.

## Core Literature

The following bibliography provides the numbered IEEE style citations used in the report.

[1] B. Lim, S. O. Arik, N. Loeff, and T. Pfister, "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting," *International Journal of Forecasting*, vol. 37, no. 4, pp. 1748-1764, 2021, doi: 10.1016/j.ijforecast.2021.03.012. Local copy: [`../publications/TFT.pdf`](../publications/TFT.pdf).

[2] S. Bai, J. Z. Kolter, and V. Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," *arXiv preprint* arXiv:1803.01271, 2018, doi: 10.48550/arXiv.1803.01271. Local copy: [`../publications/TCN.pdf`](../publications/TCN.pdf).

[3] M. A. Rafi, G. N. Rodrigues, M. N. H. Mir, M. S. M. Bhuiyan, M. F. Mridha, M. R. Islam, and Y. Watanobe, "A Hybrid Temporal Convolutional Network and Transformer Model for Accurate and Scalable Sales Forecasting," *IEEE Open Journal of the Computer Society*, vol. 6, pp. 380-391, 2025, doi: 10.1109/OJCS.2025.3538579. Local copy: [`../publications/Hyb_TCN_TFT_Sales.pdf`](../publications/Hyb_TCN_TFT_Sales.pdf).

[4] L. Mi, Y. Han, L. Long, H. Chen, and C. S. Cai, "A Physics-Informed Temporal Convolutional Network-Temporal Fusion Transformer Hybrid Model for Probabilistic Wind Speed Predictions With Quantile Regression," *Energy*, vol. 326, art. no. 136302, 2025, doi: 10.1016/j.energy.2025.136302. Local copy: [`../publications/TCN_TFT_Wind.pdf`](../publications/TCN_TFT_Wind.pdf).

[5] A. B. Alsultani, O. S. Alkhafaf, J. G. Chase, and B. Benyo, "A Multi-Feature Quantile Regression Approach for Insulin Sensitivity Prediction," in *2025 IEEE 19th International Symposium on Applied Computational Intelligence and Informatics (SACI)*, pp. 359-364, 2025, doi: 10.1109/SACI66288.2025.11030158. Local copy: [`../publications/Quantile_Regression_Insulin_Prediction.pdf`](../publications/Quantile_Regression_Insulin_Prediction.pdf).

[6] O. S. Alkhafaf, J. G. Chase, and B. Benyo, "Evaluation of Insulin Sensitivity Temporal Prediction by Using Quantile Regression Combined With Neural Network Model," *International Journal of Medical Informatics*, vol. 202, art. no. 105964, 2025, doi: 10.1016/j.ijmedinf.2025.105964. Local copy: [`../publications/Temporal_Quantile_Insulin_Prediction.pdf`](../publications/Temporal_Quantile_Insulin_Prediction.pdf).

[7] S. Khamesian, A. Arefeen, B. M. Thompson, M. A. Grando, and H. Ghasemzadeh, "AZT1D: A Real-World Dataset for Type 1 Diabetes," *arXiv preprint* arXiv:2506.14789, 2025, doi: 10.48550/arXiv.2506.14789. Local copy: [`../publications/azt1d-dataset.pdf`](../publications/azt1d-dataset.pdf).

## Repository Sources

Implementation claims in the report were checked against the following source files:

- `README.md`
- `defaults.py`
- `src/config/data.py`
- `src/config/model.py`
- `src/config/runtime.py`
- `src/data/preprocessor.py`
- `src/data/transforms.py`
- `src/data/schema.py`
- `src/data/indexing.py`
- `src/data/dataset.py`
- `src/data/datamodule.py`
- `src/models/tcn.py`
- `src/models/tft.py`
- `src/models/grn.py`
- `src/models/nn_head.py`
- `src/models/fused_model.py`
- `src/evaluation/metrics.py`
- `src/train.py`
- `src/workflows/training.py`

These files provide the implementation basis for the claims made in the dataset, methodology, and results sections.
