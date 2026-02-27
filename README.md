# CPP - Crop Price Prediction

CPP is a Streamlit-based application for scraping crop price data from Agrosight, managing datasets, training forecasting models, and comparing model predictions.

## Data Source

All datasets are collected from:

- https://agrosightinfo.com/

---

## Features

- Scrape crop price table data from Agrosight URLs
- Save outputs to CSV and JSON
- Keep activity logs in `history.json`
- Read and visualize datasets
- Train multiple forecasting algorithms
- Predict next 30 days prices
- Compare models trained on the same dataset

---

## Project Structure

- `streamlit_app.py` – Main Streamlit web UI
- `agrosight_scraper.py` – Scraping utilities
- `training_model.py` – Model training and prediction logic
- `dataset/csv/` – Scraped CSV datasets
- `dataset/json/` – Scraped JSON datasets
- `Model/` – Trained model artifacts + metadata
- `history.json` – Scraping activity history
- `requirement.txt` – Python dependencies

---

## Installation

1. Create and activate virtual environment (optional but recommended).
2. Install dependencies:

```bash
py -m pip install -r requirement.txt
```

---

## Run Application

```bash
streamlit run streamlit_app.py
```

If `streamlit` command is not found, use:

```bash
py -m streamlit run streamlit_app.py
```

---

## App Pages

### Home

Project overview, workflow explanation, model options, and comparison flow.

### Scrap Dataset

- Input:
  - URL
  - Max Page
  - Output Prefix (optional)
- Output:
  - CSV file in `dataset/csv`
  - JSON file in `dataset/json`
- Logs action to `history.json`

### Dataset

- Select CSV file
- Preview dataset table
- Filter by date range
- Line graph behavior:
  - Uses date (`နေ့စွဲ`) and price (`စျေးနှုန်း (မြန်မာကျပ်)`)
  - Default interval is monthly (every 1st day)
  - Y-axis range is always:
  - Start = `min_price - 1000`
  - End = `max_price + 1000`

### Traing Model

Train model from selected dataset and algorithm.

Supported algorithms:

- XGBoost Regressor
- LightGBM Regressor
- CatBoostRegressor
- SARIMA + ElasticNet

Training output includes:

- Model artifact in `Model/`
- Metadata file (`.meta.json`) in `Model/`
- Metrics:
  - Accuracy (%), R², MAE, RMSE, MAPE

### Model

- Select model artifact file (`.ubj`, `.txt`, `.cbm`, `.pkl`)
- Auto-load matching metadata (`.meta.json`)
- Show model metadata + metrics
- Predict next 30 days prices
- Show prediction table and line graph

### Compare Model

- Select one dataset
- Select multiple models trained on that dataset
- Compare metrics side-by-side
- Compare next 30-day predictions in one chart
- Chart labels use algorithm names

### History

Shows scraping history from `history.json`.

---

## Model Artifacts

Artifact extension by algorithm:

- XGBoost Regressor → `.ubj`
- LightGBM Regressor → `.txt`
- CatBoostRegressor → `.cbm`
- SARIMA + ElasticNet → `.pkl`

Each model has a corresponding metadata file:

- `<model_name>.meta.json`

---

## Notes

- Ensure required libraries are installed from `requirement.txt`.
- If a model fails to load, verify both artifact file and matching `.meta.json` exist in `Model/`.
- For legacy metadata formats, compatibility logic is included in model loading.
