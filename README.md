# Predictive Modeling and Spatiotemporal Analysis of Tuberculosis in Argentina

## Overview

This repository contains the code, datasets, and analyses used in the study **"Predictive Modeling and Spatiotemporal Analysis of TB in Argentina: Advancing Control Efforts Through Machine Learning"**.  
The project integrates **machine learning**, **time-series forecasting**, and **spatial epidemiology** to understand tuberculosis (TB) dynamics across Argentina, aiming to improve **surveillance**, **prediction**, and **targeted public health interventions**.

### Key Objectives
- Identify **spatiotemporal clusters** of high TB incidence using *Moran’s I* and *LISA*.
- Model **weekly TB incidence** through ARIMA, SARIMAX, and LSTM models.
- Predict **treatment outcomes** using machine learning classifiers (HGB, XGBoost, Random Forest, Logistic Regression).
- Quantify the influence of **socioeconomic, demographic, and environmental** factors on TB transmission and treatment success.

---

## Repository Structure

### Folders
- **Bases/** – Raw and processed epidemiological, socioeconomic, and climatic datasets (2010–2022).  
- **Models/** – Trained models for incidence forecasting and treatment outcome prediction.  
- **Resultados/** – Model predictions, metrics, and validation results.  
- **Documentation/** – Figures, tables, and reports used in the manuscript.  
- **utils/** – Helper scripts (data preprocessing, visualization, and evaluation).  
- **Additional/** – Ancillary scripts for data extraction and preprocessing.

### Notebooks
1. **Cleaning.ipynb** – Data cleaning and preprocessing.  
2. **Analysis.ipynb** – Exploratory data analysis (EDA).  
3. **LISA.ipynb** – Spatial autocorrelation and cluster detection (Moran’s I, LISA).  
4. **Modelling.ipynb** – Machine learning models for treatment outcome prediction.  
5. **Time_series.ipynb** – Time-series forecasting using SARIMAX and LSTM models.

---

## Methodology Summary

| Task | Approach | Tools |
|------|-----------|-------|
| Treatment Outcome Prediction | Histogram-Based Gradient Boosting (HGB), XGBoost, Random Forest, Logistic Regression | `scikit-learn`, `xgboost`, `imbalanced-learn` |
| Time-Series Forecasting | ARIMA, SARIMAX, LSTM | `statsmodels`, `pmdarima`, `tensorflow/keras` |
| Spatial Analysis | Moran’s I, Local Indicators of Spatial Association (LISA) | `PySAL`, `GeoPandas`, `Matplotlib` |

---

## Main Findings

- **Post-COVID increase in TB incidence**: weekly cases rose significantly (t = 4.75, p = 2.10×10⁻⁶).  
- **Two major hotspots** detected: northern provinces (Formosa–Jujuy–Salta) and Buenos Aires metropolitan area (CABA–La Matanza).  
- **HGB model** achieved the best treatment-outcome prediction (ROC–AUC = 0.86).  
- **LSTM models** outperformed SARIMAX for temporal forecasting (up to **77% RMSE reduction** in Buenos Aires).  
- **Top predictors**: HIV status, treatment duration, age, resistance type, and sex.

---

## Evaluation Metrics

| Model Type | Metrics |
|-------------|----------|
| **Treatment Prediction** | ROC–AUC, Precision, Recall, Log Loss |
| **Time-Series Forecasting** | MAE, MSE, RMSE, AIC, BIC |

---

## Software Environment

- **Python 3.11**  
- Main libraries:  
  `scikit-learn`, `xgboost`, `statsmodels`, `pmdarima`, `tensorflow`, `imbalanced-learn`,  
  `geopandas`, `PySAL`, `matplotlib`

---

## How to Clone

```bash
git https://github.com/Nainho1703/tb-predictive-modeling-argentina
cd tb-predictive-modeling-argentina