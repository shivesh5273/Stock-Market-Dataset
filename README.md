---
title: "Stock Market Time Series Modeling – End-to-End ML Pipeline"
author: "Your Name"
date: "`r Sys.Date()`"
output:
  html_document:
    toc: true
    toc_depth: 3
    number_sections: true
---

## Project Overview

This project implements a **complete, disciplined machine learning workflow** for stock market time-series modeling using tabular features.  
The goal is **next-day price forecasting (regression)** and **direction prediction (classification)** while strictly avoiding data leakage.

The project emphasizes:
- Time-aware validation
- Feature engineering for financial data
- Model benchmarking and selection
- Reproducibility and portfolio-readiness

> ⚠️ This project is intended as a **research / portfolio benchmark**, not a production trading system.

---

## Dataset Description

| Aspect | Description |
|------|------------|
| Type | Time-series (chronological) |
| Frequency | Daily |
| Target (Regression) | Next-day closing price |
| Target (Classification) | Up / Down movement |
| Size | ~60 trading days (small dataset) |
| Split Strategy | Chronological (Train → Validation → Test) |

Key constraints:
- No shuffling
- No future leakage
- All feature transformations fit **only on training data**

---

## Feature Engineering

Leakage-safe feature construction was used throughout.

### Engineered Features
- Lagged prices (previous days)
- Rolling statistics (mean, std)
- Trend-based features
- Scaled numeric inputs (pipeline-based)

### Preprocessing Strategy
- `Pipeline` + `ColumnTransformer`
- Scaling fitted **only on train**
- Reused consistently across all models

---

## Model Zoo (Benchmarks)

We evaluated a broad set of models to establish strong baselines.

### Regression Models
- Ridge Regression
- Support Vector Regression (SVR)
- Random Forest Regressor

### Classification Models
- Logistic Regression
- Linear SVM
- Tree-based classifiers

---

## Time Series Cross-Validation

Traditional k-fold CV was **not used**.

Instead:
- `TimeSeriesSplit` was applied
- Validation windows respected temporal order
- Hyperparameters tuned via `RandomizedSearchCV`

This ensures realistic performance estimation for time-series data.

---

## Dimensionality Reduction (PCA & LDA)

| Method | Type | Result |
|------|-----|-------|
| PCA + Logistic Regression | Unsupervised | Moderate improvement |
| LDA + Logistic Regression | Supervised | Performance degradation |

**Conclusion:**  
PCA was mildly helpful, while LDA struggled due to:
- Small sample size
- Class distribution instability

---

## Ensemble Models

Two ensemble strategies were explored:

| Ensemble Type | Description |
|-------------|------------|
| Soft Voting | Combines probabilistic outputs |
| Stacking | Meta-learner on top of base models |

Ensembles improved **stability**, but gains were limited by dataset size.

---

## Final Regression Model Selection

The best model was selected based on **validation RMSE**, then retrained on **Train + Validation** and evaluated on the **Test set**.

### Test Performance Comparison

| Model | MAE | RMSE | R² |
|------|-----|------|----|
| Ridge (Tuned) | 108.01 | 124.05 | -0.62 |
| Random Forest (Tuned) | 107.31 | 126.36 | -0.68 |
| SVR (Tuned) | 114.54 | 132.26 | -0.84 |

**Selected Model:** `Ridge Regression (Tuned)`

> Negative R² is expected in small, noisy financial datasets.

---

## Neural Networks (Exploratory)

Neural networks were added **for completeness**, not performance claims.

### Architectures Tested
- PyTorch MLP (2 hidden layers)
- Keras MLP (Early stopping)

### Results Summary

| Model | MAE | RMSE | R² |
|------|-----|------|----|
| PyTorch MLP | ~119 | ~135 | ~ -0.90 |
| Keras MLP | ~92 | ~119 | ~ -0.50 |

**Observation:**  
Neural networks did not consistently outperform linear models due to:
- Extremely small dataset
- High variance
- Overfitting risk

---

## Model Persistence & Reproducibility

The following artifacts were saved using `joblib`:

| Artifact | File |
|-------|------|
| Final Regression Model | `final_reg_model.joblib` |
| Final Classification Ensemble | `final_clf_model.joblib` |
| Feature Columns | `feature_cols.joblib` |

This ensures:
- Exact reproducibility
- Deployment-readiness
- No retraining required for inference

---

## Key Learnings

- Time-series ML requires **discipline over complexity**
- Leakage prevention matters more than model choice
- Simple models often outperform complex ones on small financial datasets
- Validation strategy is more important than raw accuracy

---

## Limitations

- Very small dataset (~60 days)
- Simulated / simplified market behavior
- No transaction costs or trading logic
- No walk-forward backtesting

---

## Future Improvements

- Use real market data (Yahoo Finance / Alpha Vantage)
- Add technical indicators (RSI, MACD, Bollinger Bands)
- Implement walk-forward backtesting
- Add prediction intervals (uncertainty estimation)
- Explore sequence models with larger datasets (LSTM / Transformers)

---

## Project Structure

```text
.
├── sample.ipynb
├── outputs_models/
│   ├── final_reg_model.joblib
│   ├── final_clf_model.joblib
│   └── feature_cols.joblib
├── exports/
│   ├── sample.html
│   └── sample.pdf
└── README.Rmd
