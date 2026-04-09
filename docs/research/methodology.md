
It is an adaptation of the hybrid temporal convolution and transformer based fusion networks for time series forecasting [sales and wind speed paper] integrated with a gated feature fusion network [tft paper, https://www.emergentmind.com/topics/gated-residual-fusion] and a quantile regession problem [A Multi-Feature Quantile Regression Approach for Insulin Sensitivity Prediction, Evaluation of insulin sensitivity temporal prediction by using quantile regression combined with neural network model]

The method incorporates the strengths of temporal convolution neural network and temporal fusion transformer. The dilations and causation aspects of TCN encodes features of short-terms patterns by extracting local temporal dependencies [hybrid tcn-tft sales paper, tcn paper]. similarly, the TFT model using attention head and gating mechanisms encode features of long-term patterns by extracting global temporal dependencies from multivariate time series [tft wind speed paper, tft paper]. A Gated residual network (GRN)[TFT paper, https://www.emergentmind.com/topics/gated-residual-fusion] is then repurposed as a feature fusion block to transforms concatenated encoded latent features from TCN and TFT into a non-linear, learnable and gated fused feature. Finally, a neural network head is then used for quantile regresion based on the fused latent input feature [quantile regression insulin papers]. 

Torch-TCN library codebase is used as starting point for TCN and the decoder is stripped off to output learned latent fetures. NVIDIA deep learning excercise libary is used as starting point for TFT and quantile output prediction is stripped off to output latent decoded features. Grn is extracted from TFT and is generalized to be used within TFT and Fused Model.

[TCN, TFT, GRN, Fused model diagrams go here]

The data for glucose represents a non-gaussian distributions, which is the reason an uncertainity-aware future prediction solution is required to provide confidence interval and boundary decisions

pinball function is used as a loss function for our model because it is a simple function that penalizes the quantiles and pushes them to their appropriate quantile levels. The central training objective is pinball loss, also called quantile loss. The
loss is:

\[
\mathcal{L}_q(y, \hat{y}_q) =
\begin{cases}
q(y - \hat{y}_q), & y \ge \hat{y}_q \\
(1 - q)(\hat{y}_q - y), & y < \hat{y}_q
\end{cases}
\]

This asymmetry is what makes each output channel settle into its role. When
\(q = 0.1\), overprediction is penalized much more heavily than underprediction,
so the model is pushed toward a lower conditional cutoff. When \(q = 0.9\), the
asymmetry reverses and the channel is pushed toward an upper cutoff. When
\(q = 0.5\), the asymmetry disappears and the channel behaves like a median
estimator.

Training flow:

Original Input Features arrive as static covariates, future known inputs, past observed inputs, past observed inputs are passed to TCN of different kernel sizes [3, 5, 7] to increase the receptive field for each individual TCN, all the original input features are then passed to TFT, the latent features from TCNs, TFT of tensor shape [B, [future horizon], [hiddent size]] are then concatenated [b, [future horizon], [hidden size * 4]] passed through the GRN to be converted into a [B, future horizon, hidden size] before passing to nn head of output layer size len(quantiles). pinball loss is used as a loss function and adam is used a optimizer.

TCN branch (kernel size k = 3/5/7)
| input_channels | hidden_channels | kernel_sizes | dilations | activation | dropout | normalization | output_size | prediction_length |
|---------------|----------------|--------------|-----------|------------|---------|---------------|-------------|------------------|
| 6 | [64, 64, 128] | k | [1, 2, 4] | ReLU | 0.1 | LayerNorm | 1 | 12 |

TFT
| hidden_size | num_heads | dropout | attention_dropout | layer_norm_eps | encoder_length | total_sequence_length | prediction_length | num_static_vars | num_historic_vars | num_future_vars | target_dim | quantiles |
|------------|----------|--------|------------------|----------------|----------------|-----------------------|------------------|-----------------|-------------------|----------------|------------|-----------|
| 128 | 4 | 0.1 | 0.0 | 1e-3 | 168 | 180 | 12 | 1 | 13 | 5 | 1 | [0.1, 0.5, 0.9] |

GRN

| hidden_size | num_heads | dropout | attention_dropout | layer_norm_eps | encoder_length | total_sequence_length | prediction_length | num_static_vars | num_historic_vars | num_future_vars | target_dim | quantiles |
|------------|----------|--------|------------------|----------------|----------------|-----------------------|------------------|-----------------|-------------------|----------------|------------|-----------|
| 128 | 4 | 0.1 | 0.0 | 1e-3 | 168 | 180 | 12 | 1 | 13 | 5 | 1 | [0.1, 0.5, 0.9] |

NN Head 
| feature_head_input_dim | feature_head_hidden_dim | feature_head_output_dim | feature_head_activation | feature_head_dropout | output_head_input_dim | output_head_output_dim |
|------------------------|--------------------------|--------------------------|--------------------------|----------------------|------------------------|------------------------|
| 128 | 128 | 1536 | ReLU | 0.1 | 128 | 1 |


training

train size : 70%, val size: 15%, test size: 15%
epoch : 20, batch size: 64


metrics MAE, RMSE metrics are still valid for prediction error and they are being calculated on 0.5 quantile as baseline.