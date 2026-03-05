# Crop Price Prediction System Documentation

## ABSTRACT

The Crop Price Prediction (CPP) system is a practical data-driven application designed to forecast short-term agricultural commodity prices using historical market data. The project combines web scraping, preprocessing, machine learning model training, and interactive visualization within a single Streamlit application. Data is collected from Agrosight market listings, transformed into structured datasets, and used to train forecasting models such as XGBoost, LightGBM, CatBoost, and a hybrid SARIMA + ElasticNet approach. The system outputs model evaluation metrics and next 30-day price forecasts to support early decision-making for market participants.

## INTRODUCTION

Agricultural commodity prices are volatile due to seasonality, demand-supply imbalance, transportation constraints, and market sentiment. This volatility creates uncertainty for farmers, traders, and consumers. Manual monitoring of daily market prices is time-consuming and often reactive.

The CPP project addresses this challenge by building an end-to-end forecasting workflow:

- Collect price records from Agrosight pages.
- Store the records in reusable CSV/JSON datasets.
- Clean and normalize time-series data.
- Train and evaluate multiple predictive models.
- Generate and visualize future price trends.

The application is implemented as an interactive Streamlit interface so non-technical users can operate the workflow from data collection to prediction without writing code.

## OBJECTIVE

The main objectives of this project are:

- Build an automated pipeline to scrape crop-price data from Agrosight.
- Create a consistent preprocessing process for multilingual and irregular real-world market data.
- Train and compare multiple forecasting approaches on the same dataset.
- Provide reliable 30-day ahead price predictions.
- Present metrics and forecast visualizations in a user-friendly dashboard.
- Maintain reproducibility by saving model artifacts and metadata.

## REQUIREMENTS

### Software Requirements

- Python 3.10+ (recommended)
- Streamlit
- requests
- beautifulsoup4
- lxml
- xgboost
- lightgbm
- catboost
- scikit-learn
- statsmodels
- pandas
- altair

### Hardware Requirements

- Minimum: Dual-core CPU, 4 GB RAM, 2 GB free disk space
- Recommended: Quad-core CPU, 8 GB+ RAM for faster training and visualization
- Internet connection required for web scraping

### Project Structure Requirements

- `dataset/csv/` for source and processed CSV files
- `dataset/json/` for JSON exports
- `Model/` for trained model artifacts and metadata files
- `history.json` for scrape activity logs

## THEORY BACKGROUND

### 1) Time-Series Forecasting in Agriculture

Time-series forecasting estimates future values based on historical observations ordered by time. For crop prices, recurring patterns (seasonality), short-term persistence (autocorrelation), and sudden shocks (outliers) are common. A robust pipeline should therefore combine temporal features, lag information, and noise handling.

### 2) Feature Engineering Principles Used

The project derives predictive signals from observed price history, including:

- Calendar features: day index, day of week, day of month, month, week of year
- Lag features: previous-day and previous-week prices
- Moving statistics: 7-day and 30-day moving averages
- Volatility and momentum proxies
- Price difference/change-related attributes

These engineered variables help models learn trend, periodicity, and local fluctuations.

### 3) Model Families Implemented

#### 3.1 XGBoost Regressor

XGBoost is a gradient-boosted decision-tree method where trees are added sequentially to correct previous residual errors. In each boosting round, the model minimizes an objective composed of prediction loss and regularization, which helps control overfitting. For crop-price forecasting, XGBoost is effective because it can capture nonlinear relationships between engineered variables (lag values, moving averages, momentum, calendar effects) and future prices.

Key theoretical characteristics:

- High nonlinear modeling capacity for tabular time-derived features.
- Strong bias-variance control through tree depth, learning rate, and regularization.
- Good handling of mixed feature interactions without explicit manual interaction terms.

#### 3.2 LightGBM Regressor

LightGBM is also a gradient-boosting tree method, but it is optimized for speed and memory via histogram-based binning and leaf-wise tree growth. This design often trains faster while maintaining strong predictive performance. In forecasting tasks with repeated model experiments, LightGBM provides efficient iteration while still learning complex feature relationships.

Key theoretical characteristics:

- Efficient training through histogram approximation of continuous features.
- Leaf-wise growth can improve fit quality but may require regularization to prevent overfitting.
- Well suited for medium-to-large tabular datasets with frequent retraining.

#### 3.3 CatBoostRegressor

CatBoost is a boosting algorithm designed for stability and reduced prediction shift through ordered boosting. In this project, CatBoost is configured with Huber loss, which is robust to outliers compared with pure squared-error loss. The pipeline also models `price_diff` (day-to-day price change) and reconstructs absolute price, which can stabilize learning in trending market series.

Key theoretical characteristics:

- Robust optimization behavior under noisy observations using robust loss.
- Strong nonlinear fit while maintaining practical stability.
- Difference-target strategy can reduce trend-scale sensitivity in short-horizon forecasting.

#### 3.4 SARIMA + ElasticNet (Hybrid)

This hybrid combines two modeling views:

- **SARIMA** (Seasonal AutoRegressive Integrated Moving Average) models autocorrelation and differencing-based temporal structure in the raw series.
- **ElasticNet** models the relationship between engineered features and target values with both L1 and L2 regularization.

The project combines both predictions by averaging, so the final estimate benefits from statistical time-series structure (SARIMA) and feature-driven regression signals (ElasticNet).

Key theoretical characteristics:

- Higher interpretability than pure tree-boosting methods.
- Useful when serial dynamics are strong and regularization is needed for feature effects.
- Hybridization can reduce single-model bias by blending complementary assumptions.

Using multiple model families reduces over-reliance on one hypothesis and supports comparative evaluation across nonlinear, statistical, and hybrid forecasting paradigms.

### 4) Theoretical Comparison Matrix (XGBoost, LightGBM, CatBoost, SARIMA + ElasticNet)

The matrix below summarizes expected theoretical behavior of each model family in the context of crop-price forecasting with engineered tabular time-series features.

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

### 5) Evaluation Metrics

The application reports common regression metrics:

- R² (coefficient of determination)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Accuracy (project-defined summary metric in metadata/output)

Together, these metrics provide complementary perspectives on forecast quality and error behavior.

### 6) Data Preprocessing Rationale

Real-world scraped tables can include inconsistent date formats, missing values, and non-numeric symbols. The pipeline includes:

- Date parsing/normalization across multiple formats
- Numeric extraction from formatted price strings
- Missing date filling using previous valid observations
- Outlier mitigation (notably for rice datasets) using robust median/MAD-based checks

This improves training stability and forecast reliability.

## CONCLUSION

The Crop Price Prediction project demonstrates a complete and practical forecasting system for agricultural market prices. It integrates data acquisition, cleaning, model training, evaluation, and forecasting into one deployable Streamlit application. By supporting multiple algorithms and visual comparison, the system helps users choose better-performing models for specific datasets and provides actionable short-term price expectations.

Future enhancements may include external factors (weather, fuel, logistics), model retraining automation, confidence interval estimation, and scheduled data refresh for continuous operation.

## REFERENCES

1. Agrosight Market Data Portal: https://agrosightinfo.com/
2. Streamlit Documentation: https://docs.streamlit.io/
3. XGBoost Documentation: https://xgboost.readthedocs.io/
4. LightGBM Documentation: https://lightgbm.readthedocs.io/
5. CatBoost Documentation: https://catboost.ai/docs/
6. scikit-learn Documentation: https://scikit-learn.org/stable/
7. statsmodels Documentation: https://www.statsmodels.org/stable/
