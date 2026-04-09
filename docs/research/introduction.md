# Introduction

## 2.1 Problem Context And Motivation

Forecasting blood glucose has clear practical value. Better short term estimates can support meal and insulin decisions, improve alerts, and increase the usefulness of automated insulin delivery systems. Although the present study focuses on type 1 diabetes, the broader question is how to predict glycemic trajectories from recent physiological and behavioral signals rather than from isolated readings.

## 2.2 Why Glucose Forecasting Is Difficult

Glucose prediction is difficult because the target is temporal, person dependent, and strongly shaped by delayed events. Future glucose depends on recent history, not just the current reading, so the task is naturally a multivariate time series problem. Meals, insulin delivery, activity, and device state changes can alter the trajectory in nonlinear ways, and the same intervention may have different effects across subjects. A useful model therefore needs both temporal memory and a way to represent uncertainty.

## 2.3 Problem Formulation

This report frames glucose prediction as a probabilistic multi horizon forecasting problem rather than as one step point regression. Given a historical window of glucose and related covariates, the model predicts a 12 step future glucose trajectory together with lower, median, and upper quantiles. This formulation is motivated by two practical needs. First, a useful forecasting system should estimate more than the next reading. Second, glucose dynamics are uncertain enough that interval information is often as important as the center prediction.

## 2.4 Rationale For Model Direction

This framing motivated a fused architecture that combines a Temporal Fusion Transformer, which is designed for structured multi horizon forecasting [1], with Temporal Convolutional Networks, which provide efficient causal temporal encoding through dilated convolutions [2]. The aim is to let the TCN branches capture local temporal structure while the TFT branch handles richer grouped covariates and decoder context. The resulting design is a late fusion model in which both families contribute to the final forecast.

## 2.5 Report Structure

The remainder of the report follows a standard paper structure. The dataset section describes source data, preprocessing, and sample construction. The methodology section presents the fused model, the probabilistic objective, and the experimental setup used for the reported run. The results and discussion sections then summarize empirical findings and interpret the main strengths and weaknesses of the current system.
