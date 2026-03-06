# CPP - Crop Price Prediction

CPP is a Streamlit application for scraping crop price data from Agrosight, preprocessing datasets, training forecasting models, and comparing model outputs.

## Data Source

- https://agrosightinfo.com/

## Features

- Scrape crop price tables from Agrosight URLs.
- Save scraped results to both CSV and JSON.
- Track scraping activity in `history.json`.
- Preprocess datasets and apply consistency fixes.
- Visualize prices with line charts and monthly boxplots.
- Train multiple forecasting algorithms.
- Train models using a `price_diff` target and reconstruct price outputs.
- Predict the next 30 days of prices.
- Compare metrics and forecast curves across models.

## Project Structure

- `streamlit_app.py`: main Streamlit UI and page flow.
- `agrosight_scraper.py`: scraping logic and URL/output helpers.
- `training_model.py`: data parsing, feature engineering, training, forecasting, and model I/O.
- `dataset/csv/`: scraped or preprocessed CSV datasets.
- `dataset/json/`: scraped JSON datasets.
- `Model/`: trained model artifacts and metadata files.
- `history.json`: scrape execution history.
- `requirement.txt`: Python dependencies.
- `SYSTEM_OVERVIEW.md`: end-to-end project architecture and workflow.
- `PROJECT_DOCUMENTATION.md`: formal project documentation (abstract, introduction, objective, requirements, theory, conclusion, references).

## Installation

1. Create a virtual environment (recommended).
2. Install dependencies:

```bash
py -m pip install -r requirement.txt
```

## Run

```bash
streamlit run streamlit_app.py
```

If `streamlit` is not on `PATH`:

```bash
py -m streamlit run streamlit_app.py
```

## App Pages

### Home

- Lists trained models and per-dataset forecast summary.
- Shows a "today forecast" if today is after the dataset end date.
- Shows a "next day forecast" if dataset already includes today.

### Scrap Dataset

- Inputs: URL, max page, optional output prefix.
- Outputs:
  - CSV in `dataset/csv/`
  - JSON in `dataset/json/`
- Logs success/failure in `history.json`.

### Dataset

- Loads selected CSV and applies preprocessing rules:
  - Replace `2025-05-04` price with `2025-05-03` if target is zero/null.
  - Fill missing dates in the full observed range using previous day values.
  - Reorder serial column to sequential values when needed.
  - Normalize change/percent columns (`-` to numeric defaults where needed).
  - Normalize date to datetime string format and price to float.
- Shows preprocessing notes and allows saving back to CSV.
- Supports date-range filtering.
- Displays:
  - interval-based line chart
  - monthly price boxplot

### Traing Model

- Select dataset and algorithm.
- Displays parsed row counts and training feature preview.
- Trains one of:
  - XGBoost Regressor
  - LightGBM Regressor
  - CatBoostRegressor
  - SARIMA + ElasticNet
- Uses `price_diff` as training target for all algorithms and reconstructs predicted prices.
- Saves:
  - model artifact in `Model/`
  - metadata in `Model/*.meta.json`
- Saves model-specific `training_policy` in metadata (feature set, split, target mode, and tuning budget).
- Shows metrics: Accuracy, R2, MAE, RMSE, MAPE.
- Shows training "Actual vs Predicted" chart.

Evaluation metric formulas (actual $y_i$, predicted $\hat{y}_i$, sample size $n$, mean $\bar{y}=\frac{1}{n}\sum_{i=1}^{n}y_i$):

- $R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar{y})^2}$
- $\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|$
- $\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}$
- $\mathrm{MAPE}(\%) = \frac{100}{n'}\sum_{i\in I}\left|\frac{y_i-\hat{y}_i}{y_i}\right|$, where $I=\{i\mid y_i\neq 0\}$ and $n'=|I|$
- $\mathrm{Accuracy}(\%) = \max(0,\;100-\mathrm{MAPE}(\%))$

### Model

- Select model artifact (`.ubj`, `.txt`, `.cbm`, `.pkl`).
- Auto-load matching metadata.
- Shows metadata and metrics.
- Shows model training policy from metadata.
- Shows training fit chart for the original dataset.
- Shows "Next 30 Days Price Prediction".

Forecast date behavior:

- Starts from today when today is later than the model's last dataset date.
- Otherwise starts from the next day after the model's last dataset date.

### Compare Model

- Select one dataset and multiple models.
- Shows side-by-side metric table.
- Shows per-model training policy and consistency status.
- Generates and overlays 30-day forecasts in one chart.
- Uses the same forecast start-date behavior as the Model page.

## Model Artifacts

- XGBoost Regressor -> `.ubj`
- LightGBM Regressor -> `.txt`
- CatBoostRegressor -> `.cbm`
- SARIMA + ElasticNet -> `.pkl`
- Metadata -> `<model_name>.meta.json`

## Notes

- Install all dependencies from `requirement.txt` before training/predicting.
- Keep artifact and matching metadata together in `Model/`.
- Legacy metadata fallback is supported when loading models.
- See `SYSTEM_OVERVIEW.md` for a full step-by-step architecture walkthrough with a figure.
