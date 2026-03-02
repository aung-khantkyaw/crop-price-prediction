import csv
import json
import pickle
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import Booster as LGBMBooster
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMBooster = None
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

try:
    from sklearn.linear_model import ElasticNet
except ImportError:
    ElasticNet = None

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    SARIMAX = None

MODEL_OPTIONS = [
    "XGBoost Regressor",
    "LightGBM Regressor",
    "CatBoostRegressor",
    "SARIMA + ElasticNet",
]

MODEL_NAME_TO_TYPE = {
    "XGBoost Regressor": "xgboost_regressor",
    "LightGBM Regressor": "lightgbm_regressor",
    "CatBoostRegressor": "catboost_regressor",
    "SARIMA + ElasticNet": "sarima_elasticnet",
}

LEGACY_MODEL_TYPE_MAP = {
    "xgboost": "xgboost_regressor",
    "lightgbm": "lightgbm_regressor",
    "catboost": "catboost_regressor",
    "sarima+elasticnet": "sarima_elasticnet",
    "sarima_elastic_net": "sarima_elasticnet",
}

DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d")
X_COLUMN = "နေ့စွဲ"
Y_COLUMN = "စျေးနှုန်း (မြန်မာကျပ်)"
CHANGE_COLUMN = "အတက်/အကျ"
PERCENT_COLUMN = "%"
FEATURE_COLUMNS = ["day_index", "change_value", "percent_value"]
TrainingPoint = tuple[date, float, float, float]


def get_training_model_options() -> list[str]:
    return MODEL_OPTIONS.copy()


def normalize_model_type(model_type: str) -> str:
    normalized = str(model_type).strip().lower()
    return LEGACY_MODEL_TYPE_MAP.get(normalized, normalized)


def is_training_model_available(model_name: str) -> tuple[bool, str | None]:
    model_type = MODEL_NAME_TO_TYPE.get(model_name)
    if model_type == "xgboost_regressor":
        return XGBRegressor is not None, "xgboost is not installed."
    if model_type == "lightgbm_regressor":
        return LGBMRegressor is not None, "lightgbm is not installed."
    if model_type == "catboost_regressor":
        return CatBoostRegressor is not None, "catboost is not installed."
    if model_type == "sarima_elasticnet":
        if ElasticNet is None:
            return False, "scikit-learn is not installed."
        if SARIMAX is None:
            return False, "statsmodels is not installed."
        return True, None
    return False, "Unknown model type."


def is_xgboost_available() -> bool:
    return XGBRegressor is not None


def parse_date_value(raw_date: str) -> date | None:
    value = str(raw_date).strip()
    if not value:
        return None

    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def parse_price_value(raw_value: str) -> float | None:
    value = str(raw_value).strip()
    cleaned = re.sub(r"[^0-9.\-]", "", value)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def load_dataset_points(csv_path: Path) -> list[TrainingPoint]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    points: list[TrainingPoint] = []
    for row in rows:
        parsed_date = parse_date_value(str(row.get(X_COLUMN, "")))
        parsed_price = parse_price_value(str(row.get(Y_COLUMN, "")))
        parsed_change = parse_price_value(str(row.get(CHANGE_COLUMN, "")))
        parsed_percent = parse_price_value(str(row.get(PERCENT_COLUMN, "")))
        if parsed_date is None or parsed_price is None:
            continue
        change_value = float(parsed_change) if parsed_change is not None else 0.0
        percent_value = float(parsed_percent) if parsed_percent is not None else 0.0
        points.append((parsed_date, float(parsed_price), change_value, percent_value))

    points.sort(key=lambda item: item[0])
    return points


def build_training_features(points: list[TrainingPoint]) -> tuple[list[list[float]], list[float], date, date]:
    if len(points) < 2:
        raise ValueError("Need at least 2 valid rows to train model.")

    origin_date = points[0][0]
    last_date = points[-1][0]
    features: list[list[float]] = []
    targets: list[float] = []

    for point_date, price, change_value, percent_value in points:
        day_index = float((point_date - origin_date).days)
        features.append([day_index, float(change_value), float(percent_value)])
        targets.append(float(price))

    return features, targets, origin_date, last_date


def calculate_regression_metrics(actual_values: list[float], predicted_values: list[float]) -> dict:
    if not actual_values or len(actual_values) != len(predicted_values):
        raise ValueError("Invalid values for metric calculation.")

    count = len(actual_values)
    absolute_errors = [abs(actual - predicted) for actual, predicted in zip(actual_values, predicted_values)]
    squared_errors = [(actual - predicted) ** 2 for actual, predicted in zip(actual_values, predicted_values)]

    mae = sum(absolute_errors) / count
    mse = sum(squared_errors) / count
    rmse = mse ** 0.5

    mean_actual = sum(actual_values) / count
    total_variance = sum((actual - mean_actual) ** 2 for actual in actual_values)
    residual_variance = sum(squared_errors)
    if total_variance == 0:
        r2 = 0.0
    else:
        r2 = 1 - (residual_variance / total_variance)

    mape_values = [
        abs((actual - predicted) / actual)
        for actual, predicted in zip(actual_values, predicted_values)
        if actual != 0
    ]
    mape = (sum(mape_values) / len(mape_values)) * 100 if mape_values else None
    accuracy_percent = max(0.0, 100.0 - mape) if mape is not None else None

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape_percent": mape,
        "accuracy_percent": accuracy_percent,
    }


def _to_float_list(values: Any) -> list[float]:
    output: list[float] = []
    for value in values:
        try:
            output.append(float(value))
        except (TypeError, ValueError):
            output.append(0.0)
    return output


def _train_xgboost(features: list[list[float]], targets: list[float]) -> tuple[Any, list[float]]:
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed.")

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(features, targets)
    predictions = _to_float_list(model.predict(features))
    return model, predictions


def _train_lightgbm(features: list[list[float]], targets: list[float]) -> tuple[Any, list[float]]:
    if LGBMRegressor is None:
        raise ImportError("lightgbm is not installed.")

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42,
    )
    model.fit(features, targets)
    predictions = _to_float_list(model.predict(features))
    return model, predictions


def _train_catboost(features: list[list[float]], targets: list[float]) -> tuple[Any, list[float]]:
    if CatBoostRegressor is None:
        raise ImportError("catboost is not installed.")

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
    )
    model.fit(features, targets)
    predictions = _to_float_list(model.predict(features))
    return model, predictions


def _train_sarima_elasticnet(features: list[list[float]], targets: list[float]) -> tuple[dict, list[float]]:
    if ElasticNet is None:
        raise ImportError("scikit-learn is not installed.")
    if SARIMAX is None:
        raise ImportError("statsmodels is not installed.")

    elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000)
    elastic_model.fit(features, targets)
    elastic_predictions = _to_float_list(elastic_model.predict(features))

    sarima_result = SARIMAX(
        targets,
        order=(1, 1, 1),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    sarima_predictions = _to_float_list(sarima_result.get_prediction(start=0, end=len(targets) - 1).predicted_mean)

    combined_predictions = [
        (elastic_pred + sarima_pred) / 2.0
        for elastic_pred, sarima_pred in zip(elastic_predictions, sarima_predictions)
    ]

    return {"elastic_model": elastic_model, "sarima_result": sarima_result}, combined_predictions


def create_xgb_regressor() -> Any:
    if XGBRegressor is None:
        raise ImportError("XGBoost is not installed.")

    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="reg:squarederror",
        random_state=42,
    )


def train_xgboost_from_points(points: list[TrainingPoint]) -> tuple[Any, dict]:
    features, targets, origin_date, last_date = build_training_features(points)
    model = create_xgb_regressor()
    model.fit(features, targets)

    train_predictions = [float(value) for value in model.predict(features)]
    metrics = calculate_regression_metrics(targets, train_predictions)

    training_info = {
        "origin_date": origin_date,
        "last_date": last_date,
        "train_size": len(points),
        "feature_columns": FEATURE_COLUMNS,
        "last_observed_price": float(points[-1][1]),
        "last_observed_change": float(points[-1][2]),
        "last_observed_percent": float(points[-1][3]),
        "metrics": {
            "accuracy_percent": round(metrics["accuracy_percent"], 4)
            if metrics["accuracy_percent"] is not None
            else None,
            "r2": round(metrics["r2"], 6),
            "mae": round(metrics["mae"], 4),
            "rmse": round(metrics["rmse"], 4),
            "mape_percent": round(metrics["mape_percent"], 4)
            if metrics["mape_percent"] is not None
            else None,
        },
    }
    return model, training_info


def train_selected_model_from_points(points: list[TrainingPoint], model_name: str) -> tuple[Any, dict]:
    features, targets, origin_date, last_date = build_training_features(points)
    model_type = MODEL_NAME_TO_TYPE.get(model_name)
    if model_type is None:
        raise ValueError("Unsupported model type.")

    if model_type == "xgboost_regressor":
        model, train_predictions = _train_xgboost(features, targets)
    elif model_type == "lightgbm_regressor":
        model, train_predictions = _train_lightgbm(features, targets)
    elif model_type == "catboost_regressor":
        model, train_predictions = _train_catboost(features, targets)
    elif model_type == "sarima_elasticnet":
        model, train_predictions = _train_sarima_elasticnet(features, targets)
    else:
        raise ValueError("Unsupported model type.")

    metrics = calculate_regression_metrics(targets, train_predictions)
    training_info = {
        "model_name": model_name,
        "model_type": model_type,
        "origin_date": origin_date,
        "last_date": last_date,
        "train_size": len(points),
        "feature_columns": FEATURE_COLUMNS,
        "last_observed_price": float(points[-1][1]),
        "last_observed_change": float(points[-1][2]),
        "last_observed_percent": float(points[-1][3]),
        "metrics": {
            "accuracy_percent": round(metrics["accuracy_percent"], 4)
            if metrics["accuracy_percent"] is not None
            else None,
            "r2": round(metrics["r2"], 6),
            "mae": round(metrics["mae"], 4),
            "rmse": round(metrics["rmse"], 4),
            "mape_percent": round(metrics["mape_percent"], 4)
            if metrics["mape_percent"] is not None
            else None,
        },
    }
    return model, training_info


def _artifact_file_name(model_name: str, model_type: str) -> str:
    if model_type == "xgboost_regressor":
        return f"{model_name}.ubj"
    if model_type == "lightgbm_regressor":
        return f"{model_name}.txt"
    if model_type == "catboost_regressor":
        return f"{model_name}.cbm"
    if model_type == "sarima_elasticnet":
        return f"{model_name}.pkl"
    raise ValueError("Unsupported model type.")


def _save_model_artifact(model: Any, model_type: str, artifact_path: Path) -> None:
    if model_type == "xgboost_regressor":
        model.save_model(str(artifact_path))
        return
    if model_type == "lightgbm_regressor":
        model.booster_.save_model(str(artifact_path))
        return
    if model_type == "catboost_regressor":
        model.save_model(str(artifact_path))
        return
    if model_type == "sarima_elasticnet":
        with artifact_path.open("wb") as f:
            pickle.dump(model, f)
        return
    raise ValueError("Unsupported model type.")


def save_model_and_metadata(
    model: Any,
    model_dir: Path,
    model_name: str,
    dataset_file: str,
    training_info: dict,
) -> tuple[Path, Path, dict]:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_type = str(training_info.get("model_type", "xgboost_regressor"))
    model_path = model_dir / _artifact_file_name(model_name, model_type)
    metadata_path = model_dir / f"{model_name}.meta.json"

    _save_model_artifact(model, model_type, model_path)

    model_payload = {
        "model_name": model_name,
        "model_display_name": str(training_info.get("model_name", model_name)),
        "model_type": model_type,
        "model_file": model_path.name,
        "dataset_file": dataset_file,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "origin_date": training_info["origin_date"].isoformat(),
        "last_date": training_info["last_date"].isoformat(),
        "train_size": training_info["train_size"],
        "feature_columns": training_info.get("feature_columns", ["day_index"]),
        "last_observed_price": training_info.get("last_observed_price"),
        "last_observed_change": training_info.get("last_observed_change"),
        "last_observed_percent": training_info.get("last_observed_percent"),
        "metrics": training_info["metrics"],
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(model_payload, f, ensure_ascii=False, indent=2)

    return model_path, metadata_path, model_payload


def load_model_metadata(metadata_path: Path) -> dict:
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_xgboost_model(model_path: Path) -> Any:
    model = create_xgb_regressor()
    model.load_model(str(model_path))
    return model


def load_model_from_metadata(metadata: dict, model_dir: Path) -> Any:
    model_type = normalize_model_type(str(metadata.get("model_type", "xgboost_regressor")))
    model_file = model_dir / str(metadata["model_file"])

    if model_type == "xgboost_regressor":
        return load_xgboost_model(model_file)
    if model_type == "lightgbm_regressor":
        if LGBMBooster is None:
            raise ImportError("lightgbm is not installed.")
        return LGBMBooster(model_file=str(model_file))
    if model_type == "catboost_regressor":
        if CatBoostRegressor is None:
            raise ImportError("catboost is not installed.")
        model = CatBoostRegressor(verbose=False)
        model.load_model(str(model_file))
        return model
    if model_type == "sarima_elasticnet":
        with model_file.open("rb") as f:
            return pickle.load(f)
    raise ValueError("Unsupported model type.")


def _predict_for_features(model: Any, model_type: str, feature_values: list[float]) -> float:
    if model_type in {"xgboost_regressor", "catboost_regressor"}:
        return float(model.predict([feature_values])[0])
    if model_type == "lightgbm_regressor":
        return float(model.predict([feature_values])[0])
    if model_type == "sarima_elasticnet":
        raise ValueError("Use SARIMA + ElasticNet batch forecast method.")
    raise ValueError("Unsupported model type.")


def _calculate_percent_change(change_value: float, previous_price: float) -> float:
    if previous_price == 0:
        return 0.0
    return (change_value / previous_price) * 100.0


def _build_future_feature_vector(
    day_index: float,
    feature_columns: list[str],
    previous_change: float,
    previous_percent: float,
) -> list[float]:
    if len(feature_columns) <= 1:
        return [day_index]
    return [day_index, float(previous_change), float(previous_percent)]


def get_training_fit_rows(points: list[TrainingPoint], model: Any, model_type: str) -> list[dict]:
    if not points:
        return []

    normalized_model_type = normalize_model_type(model_type)
    features, targets, _, _ = build_training_features(points)

    if normalized_model_type in {"xgboost_regressor", "lightgbm_regressor", "catboost_regressor"}:
        predicted_values = _to_float_list(model.predict(features))
    elif normalized_model_type == "sarima_elasticnet":
        elastic_predictions = _to_float_list(model["elastic_model"].predict(features))
        sarima_predictions = _to_float_list(model["sarima_result"].get_prediction(start=0, end=len(targets) - 1).predicted_mean)
        predicted_values = [
            (elastic_pred + sarima_pred) / 2.0
            for elastic_pred, sarima_pred in zip(elastic_predictions, sarima_predictions)
        ]
    else:
        raise ValueError("Unsupported model type.")

    fit_rows: list[dict] = []
    for point, actual_value, predicted_value in zip(points, targets, predicted_values):
        fit_rows.append(
            {
                X_COLUMN: point[0].isoformat(),
                "Actual": round(float(actual_value), 2),
                "Predicted": round(float(predicted_value), 2),
            }
        )
    return fit_rows


def predict_next_days(model: Any, origin_date: date, last_date: date, days: int = 30) -> list[dict]:
    predictions = []
    for day_offset in range(1, days + 1):
        predict_date = last_date + timedelta(days=day_offset)
        x_value = float((predict_date - origin_date).days)
        predicted_price = float(model.predict([[x_value]])[0])
        predictions.append(
            {
                X_COLUMN: predict_date.isoformat(),
                Y_COLUMN: round(max(predicted_price, 0.0), 2),
            }
        )
    return predictions


def predict_next_days_from_model(metadata: dict, model_dir: Path, days: int = 30) -> list[dict]:
    origin_date = date.fromisoformat(str(metadata["origin_date"]))
    last_date = date.fromisoformat(str(metadata["last_date"]))
    model_type = normalize_model_type(str(metadata.get("model_type", "xgboost_regressor")))
    model = load_model_from_metadata(metadata, model_dir)
    feature_columns = [str(value) for value in metadata.get("feature_columns", ["day_index"])]
    previous_price = float(metadata.get("last_observed_price", 0.0) or 0.0)
    previous_change = float(metadata.get("last_observed_change", 0.0) or 0.0)
    previous_percent = float(metadata.get("last_observed_percent", 0.0) or 0.0)

    if model_type not in {"sarima_elasticnet"}:
        predictions = []
        for day_offset in range(1, days + 1):
            predict_date = last_date + timedelta(days=day_offset)
            day_index = float((predict_date - origin_date).days)
            feature_values = _build_future_feature_vector(
                day_index=day_index,
                feature_columns=feature_columns,
                previous_change=previous_change,
                previous_percent=previous_percent,
            )
            predicted_price = _predict_for_features(model, model_type, feature_values)
            predicted_price = max(float(predicted_price), 0.0)
            predicted_change = predicted_price - previous_price
            predicted_percent = _calculate_percent_change(predicted_change, previous_price)
            previous_price = predicted_price
            previous_change = predicted_change
            previous_percent = predicted_percent
            predictions.append(
                {
                    X_COLUMN: predict_date.isoformat(),
                    Y_COLUMN: round(predicted_price, 2),
                }
            )
        return predictions

    elastic_model = model["elastic_model"]
    sarima_result = model["sarima_result"]
    elastic_predictions: list[float] = []
    for day_offset in range(1, days + 1):
        predict_date = last_date + timedelta(days=day_offset)
        day_index = float((predict_date - origin_date).days)
        feature_values = _build_future_feature_vector(
            day_index=day_index,
            feature_columns=feature_columns,
            previous_change=previous_change,
            previous_percent=previous_percent,
        )
        elastic_pred = float(elastic_model.predict([feature_values])[0])
        elastic_predictions.append(elastic_pred)

        estimated_price = max(elastic_pred, 0.0)
        estimated_change = estimated_price - previous_price
        estimated_percent = _calculate_percent_change(estimated_change, previous_price)
        previous_price = estimated_price
        previous_change = estimated_change
        previous_percent = estimated_percent

    sarima_forecast = _to_float_list(sarima_result.get_forecast(steps=days).predicted_mean)
    predictions = []
    for idx in range(days):
        predict_date = last_date + timedelta(days=idx + 1)
        predicted_price = (elastic_predictions[idx] + sarima_forecast[idx]) / 2.0
        predictions.append(
            {
                X_COLUMN: predict_date.isoformat(),
                Y_COLUMN: round(max(predicted_price, 0.0), 2),
            }
        )
    return predictions
