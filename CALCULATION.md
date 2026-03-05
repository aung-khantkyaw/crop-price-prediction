# CALCULATION - Crop Price Prediction

This document explains how each core method is used step by step and how model metrics are calculated in this project.

## 1) End-to-End Method Usage (Step by Step)

### Step 1: Load and parse dataset

Method: `load_dataset_points(csv_path)`

What it does:

1. Reads CSV rows.
2. Detects date, price, change, and percent columns using aliases.
3. Parses date values with supported formats (`parse_date_value`).
4. Parses numeric values by removing non-numeric symbols (`parse_price_value`).
5. Sorts rows by date.
6. Applies rice-specific outlier handling (`_sanitize_dataset_points`).
7. Adds `price_diff` using previous day price (`_attach_price_diff`).

Output:

- `TrainingPoint` list in format:
  - `(date, price, change, percent, price_diff)`

---

### Step 2: Split train and test sets

Method: `_split_train_test_points(points)`

What it does:

1. Requires at least 3 valid rows.
2. Uses 80/20 split.
3. Keeps first 80% as training data and last 20% as testing data.

Output:

- `train_points`, `test_points`

---

### Step 3: Build training features

Methods:

- `_build_training_features_for_columns(...)`
- `_apply_rolling_window_standard_scaler(...)`

What it does:

1. Creates time-based features:
   - `day_index`, `day_of_week`, `day_of_month`, `month`, `week_of_year`
2. Creates lag and statistical features:
   - `lag_1`, `lag_1_change`, `lag_1_percent`, `lag_7`
   - `ma_7`, `ma_30`, `volatility_7`, `price_momentum`
3. Builds target values (usually `price`; CatBoost uses `price_diff` in final training mode).
4. Applies rolling-window standardization (window size = 30 rows).

Output:

- `features`, `targets`, `origin_date`, `last_date`

---

### Step 4: Train selected model

Method: `train_selected_model_from_points(points, model_name)`

Supported models and training methods:

- XGBoost Regressor → `_train_xgboost(...)`
- LightGBM Regressor → `_train_lightgbm(...)`
- CatBoostRegressor → `_train_catboost(...)`
- SARIMA + ElasticNet → `_train_sarima_elasticnet(...)`

Important behavior:

- For **CatBoost**, evaluation is done using predicted `price_diff`, then converted back to price:
  - `predicted_price_t = max(previous_actual_price + predicted_diff_t, 0)`
- For other models, prediction is direct price prediction.

---

### Step 5: Evaluate model on test set

Method: `calculate_regression_metrics(actual_values, predicted_values)`

What it does:

1. Computes error arrays.
2. Calculates MAE, RMSE, R², MAPE.
3. Derives Accuracy % from MAPE.

Output metric keys:

- `mae`
- `rmse`
- `r2`
- `mape_percent`
- `accuracy_percent`

---

### Step 6: Save model and metadata

Method: `save_model_and_metadata(...)`

What it saves:

1. Model artifact:
   - XGBoost: `.ubj`
   - LightGBM: `.txt`
   - CatBoost: `.cbm`
   - SARIMA + ElasticNet: `.pkl`
2. Metadata JSON (`.meta.json`) containing:
   - model and dataset info
   - date range and train/test size
   - feature columns and target mode
   - last observed values
   - recent prices/feature rows
   - metrics

---

### Step 7: Predict next 30 days from saved model

Method: `predict_next_days_from_model(metadata, model_dir, days=30, start_date=None)`

What it does:

1. Loads model from metadata.
2. Rebuilds future feature vectors day by day (`_build_future_feature_vector`).
3. Applies rolling scaling using recent feature history.
4. Produces recursive predictions:
   - Non-SARIMA hybrid: direct per-step prediction.
   - SARIMA + ElasticNet: average of ElasticNet and SARIMA forecasts.
5. Returns rows with date and predicted price.

---

## 2) Metric Calculation Formulas

Assume:

- Actual values: $y_i$
- Predicted values: $\hat{y}_i$
- Number of samples: $n$

### 2.1 MAE (Mean Absolute Error)

$$
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

Code mapping:

- `absolute_errors = [abs(actual - predicted) ...]`
- `mae = sum(absolute_errors) / count`

### 2.2 RMSE (Root Mean Squared Error)

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

Code mapping:

- `squared_errors = [(actual - predicted) ** 2 ...]`
- `mse = sum(squared_errors) / count`
- `rmse = mse ** 0.5`

### 2.3 R-square (Coefficient of Determination)

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
$$

where

$$
\bar{y}=\frac{1}{n}\sum_{i=1}^{n} y_i
$$

Code mapping:

- `total_variance = sum((actual - mean_actual) ** 2 for actual in actual_values)`
- `residual_variance = sum(squared_errors)`
- `r2 = 1 - (residual_variance / total_variance)`
- Special case: if `total_variance == 0`, code sets `r2 = 0.0`

### 2.4 MAPE (Mean Absolute Percentage Error)

$$
\mathrm{MAPE}(\%) = \frac{100}{m}\sum_{i=1}^{m}\left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

Notes:

- The code excludes samples where $y_i = 0$.
- So $m$ is the number of non-zero actual values.

Code mapping:

- `mape_values = [abs((actual - predicted)/actual) for ... if actual != 0]`
- `mape = (sum(mape_values) / len(mape_values)) * 100`

### 2.5 Accuracy (%) used in this project

$$
\mathrm{Accuracy}(\%) = \max(0, 100 - \mathrm{MAPE}(\%))
$$

Code mapping:

- `accuracy_percent = max(0.0, 100.0 - mape)` (if MAPE exists)

---

## 3) Metadata Fields and How They Are Produced

Metadata is created in `save_model_and_metadata(...)` using `training_info` from `train_selected_model_from_points(...)`.

### Core metadata fields

- `model_name`: saved model name
- `model_display_name`: selected algorithm label
- `model_type`: normalized type key
- `model_file`: artifact filename
- `dataset_file`: source CSV filename
- `created_at`: UTC ISO timestamp
- `origin_date`: first date in training timeline
- `last_date`: last date in dataset used for final fit
- `train_size`: number of training rows (80%)
- `test_size`: number of test rows (20%)
- `feature_columns`: feature list used
- `target_mode`: `price` or `price_diff`

### Last-observed context fields

- `last_observed_price`
- `last_observed_change`
- `last_observed_percent`
- `last_observed_price_diff`
- `recent_prices` (last up to 30 prices)
- `recent_feature_rows_raw` (recent raw features, rolling window context)

### Metrics block

`metrics` contains rounded values:

- `accuracy_percent` (4 decimals)
- `r2` (6 decimals)
- `mae` (4 decimals)
- `rmse` (4 decimals)
- `mape_percent` (4 decimals)

---

## 4) Quick Example (Manual Metric Check)

Given:

- Actual: `[100, 110, 105]`
- Predicted: `[98, 112, 103]`

Errors:

- Absolute errors: `[2, 2, 2]` → MAE = $(2+2+2)/3 = 2.0$
- Squared errors: `[4, 4, 4]` → MSE = $12/3 = 4$ → RMSE = $2.0$
- MAPE:
  - $|2/100|=0.02$
  - $|2/110|\approx0.01818$
  - $|2/105|\approx0.01905$
  - Average $\approx 0.01908$ → MAPE $\approx 1.908\%$
- Accuracy (%) = $100 - 1.908 = 98.092\%$

---

## 5) How to Use This in the App

1. Open `Traing Model` page.
2. Select dataset and algorithm.
3. Click `Train Model`.
4. Read metrics shown (Accuracy, R², MAE, RMSE, MAPE).
5. Open `Model` page to inspect metadata and next 30-day predictions.
6. Open `Compare Model` page to compare metrics across algorithms on the same dataset.

---

## 6) Detailed Explanation of Each Model

### 6.1 XGBoost Regressor

Training method: `_train_xgboost(features, targets)`

How it works:

- Uses gradient boosting of decision trees.
- Builds trees sequentially; each new tree focuses on previous errors.
- In this project, it uses:
  - `n_estimators=500`
  - `learning_rate=0.05`
  - `max_depth=6`
  - objective: squared error

Why it is useful for this project:

- Handles nonlinear relationships between engineered time features and price.
- Usually strong baseline performance on tabular datasets.

Practical strengths:

- Good accuracy when data pattern is nonlinear.
- Robust to mixed feature scales after rolling normalization.

Possible limitations:

- Can overfit if data is very small or noisy.
- Training can be slower than LightGBM on larger datasets.

---

### 6.2 LightGBM Regressor

Training method: `_train_lightgbm(features, targets)`

How it works:

- Also a gradient-boosted tree model, optimized for speed and memory.
- In this project, it uses:
  - `n_estimators=300`
  - `learning_rate=0.03`
  - `num_leaves=31`

Why it is useful for this project:

- Fast training and inference for interactive model iteration.
- Works well with numeric engineered features.

Practical strengths:

- Efficient for repeated training experiments.
- Often achieves competitive RMSE/MAE with lower runtime.

Possible limitations:

- May need parameter tuning to outperform CatBoost/XGBoost consistently.

---

### 6.3 CatBoostRegressor

Training method: `_train_catboost(features, targets)`

How it works in this project:

- Model is trained with robust loss (`Huber`) and early stopping.
- Uses:
  - `iterations=2000`
  - `learning_rate=0.03`
  - `depth=6`
  - `loss_function="Huber:delta=180"`
- Special behavior in your pipeline:
  - Final target mode is `price_diff` (daily difference), not direct price.
  - Predicted difference is reconstructed to price by
    - `predicted_price_t = max(previous_price + predicted_diff_t, 0)`

Why it is useful for this project:

- Predicting difference can stabilize training when absolute prices trend over time.
- Huber loss helps reduce the influence of outliers.

Practical strengths:

- Good robustness when market jumps/noise appear.
- Early stopping helps avoid unnecessary overfitting.

Possible limitations:

- If reconstructed error accumulates over many steps, long-horizon forecast may drift.

---

### 6.4 SARIMA + ElasticNet (Hybrid)

Training method: `_train_sarima_elasticnet(features, targets)`

How it works:

1. Trains an `ElasticNet` regression model on engineered features.
2. Fits a SARIMA model on the target series (`SARIMAX`).
3. Combines both predictions using simple averaging:
   $$
   \hat{y}^{(hybrid)}_t = \frac{\hat{y}^{(elastic)}_t + \hat{y}^{(sarima)}_t}{2}
   $$

Why it is useful for this project:

- ElasticNet captures feature-driven patterns.
- SARIMA captures classical time-series structure (autoregressive/moving-average behavior).
- Hybrid averaging can reduce single-model bias.

Practical strengths:

- More interpretable time-series component than pure boosting models.
- Can perform well when trend/serial structure is strong.

Possible limitations:

- More complex pipeline than single-model methods.
- Sensitive to time-series assumptions and order selection quality.

---

## 7) Theoretical Comparison of 4 Models

This section compares the four algorithms conceptually (model design, assumptions, and expected behavior), without relying on experiment numbers.

### 7.1 Comparison dimensions

Theoretical comparison is based on:

- learning principle,
- ability to model nonlinearity,
- robustness to noise/outliers,
- data-size sensitivity,
- interpretability,
- computational cost,
- suitability for recursive multi-step forecasting.

### 7.2 Model-by-model theoretical view

#### XGBoost Regressor

- **Core idea:** additive tree boosting with residual correction at each stage.
- **Nonlinearity:** strong; captures complex interactions among lag/time features.
- **Bias-variance profile:** low bias, potentially higher variance if overfit is not controlled.
- **Outlier handling:** moderate; depends on objective/loss and tree splits.
- **Complexity:** medium to high training cost; medium inference cost.
- **Interpretability:** moderate (feature importance available, but global behavior is still complex).
- **Expected behavior in crop prices:** strong when relationship between engineered features and price is nonlinear.

#### LightGBM Regressor

- **Core idea:** gradient-boosted trees with histogram-based and leaf-wise growth for efficiency.
- **Nonlinearity:** strong, similar family to XGBoost.
- **Bias-variance profile:** often low bias; may overfit faster because leaf-wise growth is aggressive.
- **Outlier handling:** moderate.
- **Complexity:** generally faster training and lower memory use than XGBoost.
- **Interpretability:** moderate (feature importance is available).
- **Expected behavior in crop prices:** good practical balance of speed and predictive power for tabular time features.

#### CatBoostRegressor (with `price_diff` target)

- **Core idea:** ordered boosting; here trained with Huber loss and `price_diff` target reconstruction.
- **Nonlinearity:** strong.
- **Bias-variance profile:** balanced; ordered boosting is designed to reduce prediction shift.
- **Outlier handling:** relatively strong in this project due to Huber loss.
- **Complexity:** medium to high training cost.
- **Interpretability:** moderate.
- **Expected behavior in crop prices:** predicting day-to-day difference can improve stability in trending series; however, recursive reconstruction can accumulate error over long horizons.

#### SARIMA + ElasticNet (Hybrid)

- **Core idea:** combine statistical time-series model (SARIMA) and regularized linear feature model (ElasticNet), then average predictions.
- **Nonlinearity:** limited compared with tree boosting (ElasticNet is linear; SARIMA is parametric time-series).
- **Bias-variance profile:** usually higher bias but potentially lower variance and better structure control.
- **Outlier handling:** moderate to weak unless preprocessing is strong.
- **Complexity:** conceptually more complex pipeline, but interpretable components.
- **Interpretability:** highest among the four (clear coefficients and time-series parameters).
- **Expected behavior in crop prices:** useful when trend/autocorrelation structure is stable and explainability is important.

### 7.3 Theoretical comparison matrix

| Model               | Core Learning Principle                                                   | Nonlinear Capacity | Training Efficiency | Noise/Outlier Robustness         | Interpretability | Best Theoretical Fit                                                   |
| ------------------- | ------------------------------------------------------------------------- | ------------------ | ------------------- | -------------------------------- | ---------------- | ---------------------------------------------------------------------- |
| XGBoost Regressor   | Sequential gradient-boosted trees with regularization                     | High               | Medium              | Medium                           | Medium           | Complex nonlinear interactions among lag, trend, and calendar features |
| LightGBM Regressor  | Histogram-based, leaf-wise gradient-boosted trees                         | High               | High                | Medium                           | Medium           | Fast experimentation and scalable boosting on tabular features         |
| CatBoostRegressor   | Ordered boosting with robust loss behavior                                | High               | Medium              | Medium-High                      | Medium           | Noisy market data, especially with difference-based target strategy    |
| SARIMA + ElasticNet | Hybrid of statistical time-series model and regularized linear regression | Low-Medium         | Medium              | Medium (preprocessing-dependent) | High             | Structured autocorrelation + interpretable feature effects             |

Interpretation guideline:

- If priority is nonlinear predictive power, boosted tree models are generally favored.
- If priority is speed for repeated training cycles, LightGBM is theoretically strong.
- If priority is robustness to abrupt market noise, CatBoost with robust loss is attractive.
- If priority is interpretability and explicit time-series structure, SARIMA + ElasticNet is preferred.

### 7.4 Theoretical ranking logic (without experiment values)

If priority is **highest nonlinear predictive capacity**:

1. XGBoost / LightGBM / CatBoost (same family level, tune-dependent)
2. SARIMA + ElasticNet

If priority is **training speed and iterative experimentation**:

1. LightGBM
2. XGBoost
3. CatBoost
4. SARIMA + ElasticNet (hybrid workflow complexity)

If priority is **explainability and classical time-series reasoning**:

1. SARIMA + ElasticNet
2. XGBoost / LightGBM / CatBoost

If priority is **robustness under noisy market movement** (with this project setup):

1. CatBoost (Huber + `price_diff` pipeline)
2. XGBoost / LightGBM
3. SARIMA + ElasticNet (depends strongly on preprocessing quality)

### 7.5 Practical theoretical conclusion

- There is no universally best algorithm by theory alone.
- Tree-boosting models are generally preferred for nonlinear tabular forecasting.
- Hybrid SARIMA + ElasticNet is preferred when interpretability and explicit time-series structure are prioritized.
- Final model choice should combine this theoretical view with empirical metrics (Accuracy, R², MAE, RMSE, MAPE) on the target dataset.
