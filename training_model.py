import csv
import json
import pickle
import re
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import Booster as LGBMBooster
    from lightgbm import LGBMRegressor
except Exception:
    LGBMBooster = None
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    from sklearn.linear_model import ElasticNet
except Exception:
    ElasticNet = None

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
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

DATE_FORMATS = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)
X_COLUMN = "နေ့စွဲ"
Y_COLUMN = "စျေးနှုန်း (မြန်မာကျပ်)"
CHANGE_COLUMN = "အတက်/အကျ"
PERCENT_COLUMN = "%"
PRICE_DIFF_COLUMN = "price_diff"
DATE_COLUMN_ALIASES = (X_COLUMN, "နေစွဲ", "ရက်စွဲ", "date", "datetime")
PRICE_COLUMN_ALIASES = (Y_COLUMN, "ဈေးနှုန်း (မြန်မာကျပ်)", "ဈေးနှုန်း", "စျေးနှုန်း", "price")
CHANGE_COLUMN_ALIASES = (CHANGE_COLUMN, "အပြောင်းအလဲ", "change")
PERCENT_COLUMN_ALIASES = (PERCENT_COLUMN, "ရာခိုင်နှုန်း", "percent", "percentage")
FEATURE_COLUMNS = [
    "day_index",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "lag_1",
    "lag_1_change",
    "lag_1_percent",
    "lag_7",
    "ma_7",
    "ma_30",
    "volatility_7",
    "price_momentum",
]
ROLLING_SCALER_WINDOW = 30
VALIDATION_RATIO = 0.2
MIN_ROWS_FOR_VALIDATION = 10
XGB_ESTIMATORS = 1200
XGB_LEARNING_RATE = 0.03
LIGHTGBM_ESTIMATORS = 1200
LIGHTGBM_LEARNING_RATE = 0.03
CATBOOST_ITERATIONS = 2000
CATBOOST_LEARNING_RATE = 0.03
CATBOOST_EARLY_STOPPING_ROUNDS = 100
SARIMA_ORDER_TRIALS = 12
TrainingPoint = tuple[date, float, float, float, float]


def get_training_policy(model_type: str | None = None, target_mode: str = "price") -> dict:
    normalized_model_type = normalize_model_type(model_type or "")
    policy = {
        "feature_set": FEATURE_COLUMNS.copy(),
        "validation_ratio": VALIDATION_RATIO,
        "min_rows_for_validation": MIN_ROWS_FOR_VALIDATION,
        "target_mode": target_mode,
        "model_type": normalized_model_type or "auto",
    }

    if normalized_model_type == "xgboost_regressor":
        policy["tuning_budget"] = {
            "estimators": XGB_ESTIMATORS,
            "learning_rate": XGB_LEARNING_RATE,
            "max_depth": 6,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        }
        return policy

    if normalized_model_type == "lightgbm_regressor":
        policy["tuning_budget"] = {
            "estimators": LIGHTGBM_ESTIMATORS,
            "learning_rate": LIGHTGBM_LEARNING_RATE,
            "num_leaves": 31,
            "max_depth": -1,
        }
        return policy

    if normalized_model_type == "catboost_regressor":
        policy["tuning_budget"] = {
            "iterations": CATBOOST_ITERATIONS,
            "learning_rate": CATBOOST_LEARNING_RATE,
            "depth": 6,
            "loss_function": "Huber:delta=180",
            "early_stopping_rounds": CATBOOST_EARLY_STOPPING_ROUNDS,
        }
        return policy

    if normalized_model_type == "sarima_elasticnet":
        policy["tuning_budget"] = {
            "sarima_order_trials": SARIMA_ORDER_TRIALS,
            "elasticnet_alpha": 0.1,
            "elasticnet_l1_ratio": 0.5,
        }
        return policy

    policy["tuning_budget"] = {
        "xgb_estimators": XGB_ESTIMATORS,
        "lightgbm_estimators": LIGHTGBM_ESTIMATORS,
        "catboost_iterations": CATBOOST_ITERATIONS,
        "sarima_order_trials": SARIMA_ORDER_TRIALS,
    }
    return policy


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

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        pass

    compact_value = value.replace("T", " ").replace("Z", "").strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(compact_value, fmt).date()
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


def _normalize_column_key(value: str) -> str:
    normalized = str(value).replace("\ufeff", "")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.strip().lower()


def _resolve_row_value(row: dict, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in row:
            return str(row.get(candidate, ""))

    normalized_candidates = {_normalize_column_key(candidate) for candidate in candidates}
    for key, value in row.items():
        if _normalize_column_key(str(key)) in normalized_candidates:
            return str(value)

    return ""


def _detect_best_column(rows: list[dict], parser: Any, minimum_success: int = 2) -> str | None:
    if not rows:
        return None

    column_names: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in column_names:
                column_names.append(key)

    if not column_names:
        return None

    sample_rows = rows[: min(len(rows), 300)]
    best_column: str | None = None
    best_score = 0

    for column_name in column_names:
        score = 0
        for row in sample_rows:
            parsed = parser(str(row.get(column_name, "")))
            if parsed is not None:
                score += 1
        if score > best_score:
            best_score = score
            best_column = column_name

    if best_score < minimum_success:
        return None
    return best_column


def _is_rice_dataset(csv_path: Path) -> bool:
    file_name = csv_path.name.lower()
    return "rice" in file_name or "wartan" in file_name


def _sanitize_dataset_points(points: list[tuple[date, float, float, float]], csv_path: Path) -> list[tuple[date, float, float, float]]:
    cleaned: list[tuple[date, float, float, float]] = [point for point in points if float(point[1]) > 0]
    if not cleaned or not _is_rice_dataset(csv_path):
        return cleaned

    prices = [float(price) for _, price, _, _ in cleaned]
    if len(prices) < 10:
        return cleaned

    adjusted: list[tuple[date, float, float, float]] = []
    for index, point in enumerate(cleaned):
        point_date, price, change_value, percent_value = point
        if index < 7:
            adjusted.append((point_date, float(price), float(change_value), float(percent_value)))
            continue

        window = prices[max(0, index - 30):index]
        if len(window) < 5:
            adjusted.append((point_date, float(price), float(change_value), float(percent_value)))
            continue

        median_price = statistics.median(window)
        absolute_deviation = [abs(value - median_price) for value in window]
        mad = statistics.median(absolute_deviation) if absolute_deviation else 0.0
        robust_scale = max(mad * 1.4826, 1.0)
        robust_z = abs((float(price) - median_price) / robust_scale)
        previous_price = prices[index - 1]
        jump_ratio = abs(float(price) - previous_price) / max(abs(previous_price), 1.0)

        if robust_z > 6.0 and jump_ratio > 0.35:
            adjusted_price = float(median_price)
            adjusted.append((point_date, adjusted_price, float(change_value), float(percent_value)))
        else:
            adjusted.append((point_date, float(price), float(change_value), float(percent_value)))

    return adjusted


def _attach_price_diff(points: list[tuple[date, float, float, float]]) -> list[TrainingPoint]:
    output: list[TrainingPoint] = []
    previous_price: float | None = None
    for point_date, price, change_value, percent_value in points:
        if previous_price is None:
            price_diff = 0.0
        else:
            price_diff = float(price) - float(previous_price)
        output.append((point_date, float(price), float(change_value), float(percent_value), float(price_diff)))
        previous_price = float(price)
    return output


def load_dataset_points(csv_path: Path) -> list[TrainingPoint]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    def build_points(
        date_column: str,
        price_column: str,
        change_column: str,
        percent_column: str,
    ) -> list[tuple[date, float, float, float]]:
        output: list[tuple[date, float, float, float]] = []
        for row in rows:
            parsed_date = parse_date_value(str(row.get(date_column, ""))) if date_column else None
            parsed_price = parse_price_value(str(row.get(price_column, ""))) if price_column else None
            parsed_change = parse_price_value(str(row.get(change_column, ""))) if change_column else 0.0
            parsed_percent = parse_price_value(str(row.get(percent_column, ""))) if percent_column else 0.0
            if parsed_date is None or parsed_price is None:
                continue
            change_value = float(parsed_change) if parsed_change is not None else 0.0
            percent_value = float(parsed_percent) if parsed_percent is not None else 0.0
            output.append((parsed_date, float(parsed_price), change_value, percent_value))

        output.sort(key=lambda item: item[0])
        return output

    date_column = ""
    price_column = ""
    change_column = ""
    percent_column = ""
    if rows:
        date_column = next(
            (
                key
                for key in rows[0].keys()
                if _normalize_column_key(key)
                in {_normalize_column_key(candidate) for candidate in DATE_COLUMN_ALIASES}
            ),
            "",
        )
        price_column = next(
            (
                key
                for key in rows[0].keys()
                if _normalize_column_key(key)
                in {_normalize_column_key(candidate) for candidate in PRICE_COLUMN_ALIASES}
            ),
            "",
        )
        change_column = next(
            (
                key
                for key in rows[0].keys()
                if _normalize_column_key(key)
                in {_normalize_column_key(candidate) for candidate in CHANGE_COLUMN_ALIASES}
            ),
            "",
        )
        percent_column = next(
            (
                key
                for key in rows[0].keys()
                if _normalize_column_key(key)
                in {_normalize_column_key(candidate) for candidate in PERCENT_COLUMN_ALIASES}
            ),
            "",
        )

    points = build_points(date_column, price_column, change_column, percent_column)

    if len(points) < 2 and rows:
        fallback_date_column = _detect_best_column(rows, parse_date_value)
        fallback_price_column = _detect_best_column(rows, parse_price_value)
        fallback_change_column = _detect_best_column(rows, parse_price_value, minimum_success=1) or change_column
        fallback_percent_column = _detect_best_column(rows, parse_price_value, minimum_success=1) or percent_column

        if fallback_date_column and fallback_price_column:
            points = build_points(
                fallback_date_column,
                fallback_price_column,
                fallback_change_column,
                fallback_percent_column,
            )

    sanitized_points = _sanitize_dataset_points(points, csv_path)
    return _attach_price_diff(sanitized_points)


def _apply_rolling_window_standard_scaler(feature_rows: list[list[float]], window_size: int = ROLLING_SCALER_WINDOW) -> list[list[float]]:
    if not feature_rows:
        return []

    scaled_rows: list[list[float]] = []
    feature_count = len(feature_rows[0])
    for index, row in enumerate(feature_rows):
        history = feature_rows[max(0, index - window_size):index]
        if not history:
            scaled_rows.append([float(value) for value in row])
            continue

        scaled_row: list[float] = []
        for feature_index in range(feature_count):
            history_values = [float(history_row[feature_index]) for history_row in history]
            mean_value = sum(history_values) / len(history_values)
            variance = sum((value - mean_value) ** 2 for value in history_values) / len(history_values)
            std_value = variance ** 0.5
            current_value = float(row[feature_index])
            if std_value <= 1e-9:
                scaled_row.append(current_value - mean_value)
            else:
                scaled_row.append((current_value - mean_value) / std_value)
        scaled_rows.append(scaled_row)

    return scaled_rows


def _scale_feature_vector_with_history(
    feature_values: list[float],
    history_rows: list[list[float]],
    window_size: int = ROLLING_SCALER_WINDOW,
) -> list[float]:
    if not history_rows:
        return [float(value) for value in feature_values]

    history = history_rows[-window_size:]
    scaled_values: list[float] = []
    for feature_index, current_value in enumerate(feature_values):
        history_values = [float(row[feature_index]) for row in history if feature_index < len(row)]
        if not history_values:
            scaled_values.append(float(current_value))
            continue
        mean_value = sum(history_values) / len(history_values)
        variance = sum((value - mean_value) ** 2 for value in history_values) / len(history_values)
        std_value = variance ** 0.5
        if std_value <= 1e-9:
            scaled_values.append(float(current_value) - mean_value)
        else:
            scaled_values.append((float(current_value) - mean_value) / std_value)

    return scaled_values


def _build_training_features_for_columns(
    points: list[TrainingPoint],
    feature_columns: list[str],
    origin_date_override: date | None = None,
) -> tuple[list[list[float]], list[float], date, date]:
    if len(points) < 2:
        raise ValueError("Need at least 2 valid rows to train model.")

    origin_date = origin_date_override or points[0][0]
    last_date = points[-1][0]
    raw_features: list[list[float]] = []
    targets: list[float] = []

    prices = [float(price) for _, price, _, _, _ in points]

    for index, (point_date, price, _, _, _) in enumerate(points):
        day_index = float((point_date - origin_date).days)
        day_of_week = float(point_date.weekday())
        day_of_month = float(point_date.day)
        month = float(point_date.month)
        week_of_year = float(point_date.isocalendar()[1])

        lag_1 = prices[index - 1] if index >= 1 else prices[index]
        lag_1_change = (prices[index - 1] - prices[index - 2]) if index >= 2 else 0.0
        previous_previous_price = prices[index - 2] if index >= 2 else 0.0
        lag_1_percent = ((lag_1_change / previous_previous_price) * 100.0) if previous_previous_price != 0 else 0.0
        lag_7 = prices[index - 7] if index >= 7 else lag_1

        ma_7_window = prices[max(0, index - 6): index + 1]
        ma_30_window = prices[max(0, index - 29): index + 1]
        ma_7 = sum(ma_7_window) / len(ma_7_window)
        ma_30 = sum(ma_30_window) / len(ma_30_window)
        volatility_window = prices[max(0, index - 7): index]
        if not volatility_window:
            volatility_window = [lag_1]
        volatility_7 = statistics.pstdev(volatility_window) if len(volatility_window) > 1 else 0.0
        price_momentum = (float(lag_1) / float(lag_7)) if float(lag_7) != 0 else 0.0

        feature_map = {
            "day_index": day_index,
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,
            "month": month,
            "week_of_year": week_of_year,
            "lag_1": float(lag_1),
            "lag_1_change": float(lag_1_change),
            "lag_1_percent": float(lag_1_percent),
            "lag_7": float(lag_7),
            "ma_7": float(ma_7),
            "ma_30": float(ma_30),
            "volatility_7": float(volatility_7),
            "price_momentum": float(price_momentum),
        }
        raw_features.append([float(feature_map.get(column, 0.0)) for column in feature_columns])
        targets.append(float(price))

    scaled_features = _apply_rolling_window_standard_scaler(raw_features)
    return scaled_features, targets, origin_date, last_date


def build_training_features(points: list[TrainingPoint]) -> tuple[list[list[float]], list[float], date, date]:
    return _build_training_features_for_columns(points, FEATURE_COLUMNS)


def _build_train_test_features(
    train_points: list[TrainingPoint],
    test_points: list[TrainingPoint],
) -> tuple[list[list[float]], list[float], list[list[float]], list[float], date, date]:
    if not train_points:
        raise ValueError("Training points are empty.")

    shared_origin_date = train_points[0][0]
    train_features, train_targets, _, train_last_date = _build_training_features_for_columns(
        train_points,
        FEATURE_COLUMNS,
        origin_date_override=shared_origin_date,
    )
    test_features, test_targets, _, _ = _build_training_features_for_columns(
        test_points,
        FEATURE_COLUMNS,
        origin_date_override=shared_origin_date,
    )
    return train_features, train_targets, test_features, test_targets, shared_origin_date, train_last_date


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


def _split_train_test_points(points: list[TrainingPoint]) -> tuple[list[TrainingPoint], list[TrainingPoint]]:
    if len(points) < 3:
        raise ValueError("Need at least 3 valid rows to keep last 20% as testing data.")

    split_index = int(len(points) * 0.8)
    split_index = max(2, split_index)
    split_index = min(split_index, len(points) - 1)

    train_points = points[:split_index]
    test_points = points[split_index:]
    return train_points, test_points


def _predict_for_dataset(model: Any, model_type: str, features: list[list[float]]) -> list[float]:
    normalized_model_type = normalize_model_type(model_type)

    if normalized_model_type in {"xgboost_regressor", "lightgbm_regressor", "catboost_regressor"}:
        return _to_float_list(model.predict(features))

    if normalized_model_type == "sarima_elasticnet":
        elastic_predictions = _to_float_list(model["elastic_model"].predict(features))
        sarima_forecast = _to_float_list(model["sarima_result"].get_forecast(steps=len(features)).predicted_mean)
        return [
            (elastic_pred + sarima_pred) / 2.0
            for elastic_pred, sarima_pred in zip(elastic_predictions, sarima_forecast)
        ]

    raise ValueError("Unsupported model type.")


def _to_float_list(values: Any) -> list[float]:
    output: list[float] = []
    for value in values:
        try:
            output.append(float(value))
        except (TypeError, ValueError):
            output.append(0.0)
    return output


def _split_train_validation(
    features: list[list[float]],
    targets: list[float],
) -> tuple[list[list[float]], list[float], list[list[float]] | None, list[float] | None]:
    if len(features) != len(targets):
        raise ValueError("Features and targets must have the same length.")

    if len(features) < MIN_ROWS_FOR_VALIDATION:
        return features, targets, None, None

    validation_count = max(1, int(len(features) * VALIDATION_RATIO))
    validation_count = min(validation_count, len(features) - 1)

    train_features = features[:-validation_count]
    train_targets = targets[:-validation_count]
    valid_features = features[-validation_count:]
    valid_targets = targets[-validation_count:]
    return train_features, train_targets, valid_features, valid_targets


def _train_xgboost(features: list[list[float]], targets: list[float]) -> tuple[Any, list[float]]:
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed.")

    train_features, train_targets, valid_features, valid_targets = _split_train_validation(features, targets)
    eval_set: list[tuple[list[list[float]], list[float]]] = [(train_features, train_targets)]
    if valid_features is not None and valid_targets is not None:
        eval_set.append((valid_features, valid_targets))

    model = XGBRegressor(
        n_estimators=XGB_ESTIMATORS,
        learning_rate=XGB_LEARNING_RATE,
        max_depth=6,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="reg:squarederror",
        random_state=42,
    )
    fit_kwargs: dict[str, Any] = {"eval_set": eval_set, "verbose": False}
    model.fit(train_features, train_targets, **fit_kwargs)
    predictions = _to_float_list(model.predict(features))
    return model, predictions


def _train_lightgbm(features: list[list[float]], targets: list[float]) -> tuple[Any, list[float]]:
    if LGBMRegressor is None:
        raise ImportError("lightgbm is not installed.")

    try:
        import numpy as np
    except Exception as exc:
        raise ImportError("numpy is required for lightgbm training.") from exc

    train_features, train_targets, valid_features, valid_targets = _split_train_validation(features, targets)

    train_features_array = np.asarray(train_features, dtype=float)
    train_targets_array = np.asarray(train_targets, dtype=float)

    eval_set = None
    if valid_features is not None and valid_targets is not None:
        valid_features_array = np.asarray(valid_features, dtype=float)
        valid_targets_array = np.asarray(valid_targets, dtype=float)
        eval_set = [(valid_features_array, valid_targets_array)]

    model = LGBMRegressor(
        n_estimators=LIGHTGBM_ESTIMATORS,
        learning_rate=LIGHTGBM_LEARNING_RATE,
        num_leaves=31,
        max_depth=-1,
        random_state=42,
    )
    fit_kwargs: dict[str, Any] = {"eval_metric": "l2"}
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set
    model.fit(train_features_array, train_targets_array, **fit_kwargs)
    predictions = _to_float_list(model.predict(features))
    return model, predictions


def _train_catboost(features: list[list[float]], targets: list[float]) -> tuple[Any, list[float]]:
    if CatBoostRegressor is None:
        raise ImportError("catboost is not installed.")

    train_features, train_targets, valid_features, valid_targets = _split_train_validation(features, targets)

    model = CatBoostRegressor(
        iterations=CATBOOST_ITERATIONS,
        learning_rate=CATBOOST_LEARNING_RATE,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="Huber:delta=180",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False,
        early_stopping_rounds=CATBOOST_EARLY_STOPPING_ROUNDS,
    )
    if valid_features is not None and valid_targets is not None:
        model.fit(
            train_features,
            train_targets,
            eval_set=(valid_features, valid_targets),
            use_best_model=True,
        )
    else:
        model.fit(train_features, train_targets)
    predictions = _to_float_list(model.predict(features))
    return model, predictions


def _select_sarima_order(targets: list[float]) -> tuple[int, int, int]:
    candidate_orders = [
        (1, 1, 1),
        (2, 1, 1),
        (1, 1, 2),
        (2, 1, 2),
        (3, 1, 1),
        (0, 1, 1),
        (1, 0, 1),
    ]
    candidate_orders = candidate_orders[: max(1, SARIMA_ORDER_TRIALS)]
    best_order = (1, 1, 1)
    best_aic = float("inf")

    if SARIMAX is None:
        return best_order

    for order in candidate_orders:
        try:
            fitted = SARIMAX(
                targets,
                order=order,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            current_aic = float(getattr(fitted, "aic", float("inf")))
            if current_aic < best_aic:
                best_aic = current_aic
                best_order = order
        except Exception:
            continue

    return best_order


def _train_sarima_elasticnet(features: list[list[float]], targets: list[float]) -> tuple[dict, list[float]]:
    if ElasticNet is None:
        raise ImportError("scikit-learn is not installed.")
    if SARIMAX is None:
        raise ImportError("statsmodels is not installed.")

    train_features, train_targets, _, _ = _split_train_validation(features, targets)

    elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000)
    elastic_model.fit(train_features, train_targets)
    elastic_predictions = _to_float_list(elastic_model.predict(features))

    sarima_order = _select_sarima_order(train_targets)
    sarima_result = SARIMAX(
        train_targets,
        order=sarima_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    sarima_predictions = _to_float_list(sarima_result.get_forecast(steps=len(targets)).predicted_mean)

    combined_predictions = [
        (elastic_pred + sarima_pred) / 2.0
        for elastic_pred, sarima_pred in zip(elastic_predictions, sarima_predictions)
    ]

    return {"elastic_model": elastic_model, "sarima_result": sarima_result}, combined_predictions


def create_xgb_regressor() -> Any:
    if XGBRegressor is None:
        raise ImportError("XGBoost is not installed.")

    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="reg:squarederror",
        random_state=42,
    )


def train_xgboost_from_points(points: list[TrainingPoint]) -> tuple[Any, dict]:
    train_points, test_points = _split_train_test_points(points)
    train_features, train_targets, test_features, test_targets, origin_date, last_date = _build_train_test_features(
        train_points,
        test_points,
    )

    model = create_xgb_regressor()
    model.fit(train_features, train_targets)

    test_predictions = [float(value) for value in model.predict(test_features)]
    metrics = calculate_regression_metrics(test_targets, test_predictions)

    training_info = {
        "origin_date": origin_date,
        "last_date": last_date,
        "train_size": len(train_points),
        "test_size": len(test_points),
        "feature_columns": FEATURE_COLUMNS,
        "target_mode": "price",
        "training_policy": get_training_policy(model_type="xgboost_regressor", target_mode="price"),
        "last_observed_price": float(train_points[-1][1]),
        "last_observed_change": float(train_points[-1][2]),
        "last_observed_percent": float(train_points[-1][3]),
        "last_observed_price_diff": float(train_points[-1][4]),
        "recent_prices": [float(point[1]) for point in train_points[-30:]],
        "recent_feature_rows_raw": [list(row) for row in _build_raw_feature_rows(train_points, FEATURE_COLUMNS)[-ROLLING_SCALER_WINDOW:]],
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


def _train_model_by_type(model_type: str, features: list[list[float]], targets: list[float]) -> Any:
    if model_type == "xgboost_regressor":
        model, _ = _train_xgboost(features, targets)
        return model
    if model_type == "lightgbm_regressor":
        model, _ = _train_lightgbm(features, targets)
        return model
    if model_type == "catboost_regressor":
        model, _ = _train_catboost(features, targets)
        return model
    if model_type == "sarima_elasticnet":
        model, _ = _train_sarima_elasticnet(features, targets)
        return model
    raise ValueError("Unsupported model type.")


def train_selected_model_from_points(points: list[TrainingPoint], model_name: str) -> tuple[Any, dict]:
    train_points, test_points = _split_train_test_points(points)
    train_features, train_targets, test_features, test_targets, _, _ = _build_train_test_features(
        train_points,
        test_points,
    )

    model_type = MODEL_NAME_TO_TYPE.get(model_name)
    if model_type is None:
        raise ValueError("Unsupported model type.")

    target_mode = "price_diff"
    train_target_diffs = [float(point[4]) for point in train_points]
    evaluation_model = _train_model_by_type(model_type, train_features, train_target_diffs)
    predicted_diffs = _predict_for_dataset(evaluation_model, model_type, test_features)
    reconstructed_prices: list[float] = []
    for index, predicted_diff in enumerate(predicted_diffs):
        previous_actual_price = float(train_points[-1][1]) if index == 0 else float(test_points[index - 1][1])
        reconstructed_price = max(previous_actual_price + float(predicted_diff), 0.0)
        reconstructed_prices.append(reconstructed_price)
    test_actual_prices = [float(point[1]) for point in test_points]
    metrics = calculate_regression_metrics(test_actual_prices, reconstructed_prices)

    full_features, full_targets, origin_date, last_date = _build_training_features_for_columns(points, FEATURE_COLUMNS)
    full_target_diffs = [float(point[4]) for point in points]
    final_model = _train_model_by_type(model_type, full_features, full_target_diffs)

    training_info = {
        "model_name": model_name,
        "model_type": model_type,
        "origin_date": origin_date,
        "last_date": last_date,
        "train_size": len(train_points),
        "test_size": len(test_points),
        "feature_columns": FEATURE_COLUMNS,
        "target_mode": target_mode,
        "training_policy": get_training_policy(model_type=model_type, target_mode=target_mode),
        "last_observed_price": float(points[-1][1]),
        "last_observed_change": float(points[-1][2]),
        "last_observed_percent": float(points[-1][3]),
        "last_observed_price_diff": float(points[-1][4]),
        "recent_prices": [float(point[1]) for point in points[-30:]],
        "recent_feature_rows_raw": [list(row) for row in _build_raw_feature_rows(points, FEATURE_COLUMNS)[-ROLLING_SCALER_WINDOW:]],
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
    return final_model, training_info


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
        "test_size": training_info.get("test_size"),
        "feature_columns": training_info.get("feature_columns", ["day_index"]),
        "target_mode": training_info.get("target_mode", "price"),
        "training_policy": training_info.get(
            "training_policy",
            get_training_policy(
                model_type=model_type,
                target_mode=str(training_info.get("target_mode", "price")),
            ),
        ),
        "last_observed_price": training_info.get("last_observed_price"),
        "last_observed_change": training_info.get("last_observed_change"),
        "last_observed_percent": training_info.get("last_observed_percent"),
        "last_observed_price_diff": training_info.get("last_observed_price_diff"),
        "recent_prices": training_info.get("recent_prices", []),
        "recent_feature_rows_raw": training_info.get("recent_feature_rows_raw", []),
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


def _feature_columns_from_count(feature_count: int) -> list[str]:
    if feature_count <= 1:
        return ["day_index"]
    if feature_count == 3:
        return ["day_index", "change_value", "percent_value"]
    if feature_count >= len(FEATURE_COLUMNS):
        return FEATURE_COLUMNS.copy()
    return FEATURE_COLUMNS[:feature_count]


def _get_expected_feature_count(model: Any, model_type: str) -> int | None:
    normalized_model_type = normalize_model_type(model_type)

    if normalized_model_type == "xgboost_regressor":
        try:
            expected_count = int(model.get_booster().num_features())
            return expected_count if expected_count > 0 else None
        except Exception:
            return None

    if normalized_model_type == "lightgbm_regressor":
        try:
            if hasattr(model, "num_feature"):
                expected_count = int(model.num_feature())
                return expected_count if expected_count > 0 else None
        except Exception:
            return None
        return None

    if normalized_model_type == "catboost_regressor":
        try:
            if hasattr(model, "feature_count_"):
                expected_count = int(model.feature_count_)
                if expected_count > 0:
                    return expected_count
            if hasattr(model, "get_n_features_in"):
                expected_count = int(model.get_n_features_in())
                if expected_count > 0:
                    return expected_count
            if hasattr(model, "feature_names_"):
                expected_count = len(getattr(model, "feature_names_", []) or [])
                if expected_count > 0:
                    return expected_count
        except Exception:
            return None
        return None

    if normalized_model_type == "sarima_elasticnet":
        try:
            elastic_model = model.get("elastic_model")
            if elastic_model is not None and hasattr(elastic_model, "n_features_in_"):
                expected_count = int(elastic_model.n_features_in_)
                return expected_count if expected_count > 0 else None
        except Exception:
            return None
        return None

    return None


def _resolve_effective_feature_columns(
    model: Any,
    model_type: str,
    feature_columns: list[str] | None,
) -> list[str]:
    provided_columns = [str(column) for column in (feature_columns or []) if str(column).strip()]
    expected_count = _get_expected_feature_count(model, model_type)

    if expected_count is None:
        if provided_columns:
            return provided_columns
        return FEATURE_COLUMNS.copy()

    if provided_columns and len(provided_columns) == expected_count:
        return provided_columns

    return _feature_columns_from_count(expected_count)


def _calculate_percent_change(change_value: float, previous_price: float) -> float:
    if previous_price == 0:
        return 0.0
    return (change_value / previous_price) * 100.0


def _build_future_feature_vector(
    predict_date: date,
    day_index: float,
    feature_columns: list[str],
    previous_change: float,
    previous_percent: float,
    price_history: list[float],
) -> list[float]:
    latest_price = float(price_history[-1]) if price_history else 0.0
    lag_1 = float(price_history[-1]) if len(price_history) >= 1 else latest_price
    lag_1_change = float(price_history[-1] - price_history[-2]) if len(price_history) >= 2 else 0.0
    lag_1_percent = float((lag_1_change / price_history[-2]) * 100.0) if len(price_history) >= 2 and price_history[-2] != 0 else 0.0
    lag_7 = float(price_history[-7]) if len(price_history) >= 7 else lag_1

    ma_7_window = price_history[-7:] if price_history else [latest_price]
    ma_30_window = price_history[-30:] if price_history else [latest_price]
    ma_7 = sum(ma_7_window) / len(ma_7_window)
    ma_30 = sum(ma_30_window) / len(ma_30_window)
    volatility_7 = statistics.pstdev(ma_7_window) if len(ma_7_window) > 1 else 0.0
    price_momentum = (lag_1 / lag_7) if lag_7 != 0 else 0.0

    feature_map: dict[str, float] = {
        "day_index": float(day_index),
        "day_of_week": float(predict_date.weekday()),
        "day_of_month": float(predict_date.day),
        "month": float(predict_date.month),
        "week_of_year": float(predict_date.isocalendar()[1]),
        "change_value": float(previous_change),
        "percent_value": float(previous_percent),
        "lag_1": lag_1,
        "lag_1_change": lag_1_change,
        "lag_1_percent": lag_1_percent,
        "lag_7": lag_7,
        "ma_7": float(ma_7),
        "ma_30": float(ma_30),
        "volatility_7": float(volatility_7),
        "price_momentum": float(price_momentum),
    }
    return [float(feature_map.get(column, 0.0)) for column in feature_columns]


def _build_raw_feature_rows(
    points: list[TrainingPoint],
    feature_columns: list[str],
    origin_date_override: date | None = None,
) -> list[list[float]]:
    if not points:
        return []

    origin_date = origin_date_override or points[0][0]
    prices = [float(price) for _, price, _, _, _ in points]
    raw_rows: list[list[float]] = []

    for index, (point_date, _, _, _, _) in enumerate(points):
        day_index = float((point_date - origin_date).days)
        day_of_week = float(point_date.weekday())
        day_of_month = float(point_date.day)
        month = float(point_date.month)
        week_of_year = float(point_date.isocalendar()[1])

        lag_1 = prices[index - 1] if index >= 1 else prices[index]
        lag_1_change = (prices[index - 1] - prices[index - 2]) if index >= 2 else 0.0
        previous_previous_price = prices[index - 2] if index >= 2 else 0.0
        lag_1_percent = ((lag_1_change / previous_previous_price) * 100.0) if previous_previous_price != 0 else 0.0
        lag_7 = prices[index - 7] if index >= 7 else lag_1

        ma_7_window = prices[max(0, index - 6): index + 1]
        ma_30_window = prices[max(0, index - 29): index + 1]
        ma_7 = sum(ma_7_window) / len(ma_7_window)
        ma_30 = sum(ma_30_window) / len(ma_30_window)
        volatility_window = prices[max(0, index - 7): index]
        if not volatility_window:
            volatility_window = [lag_1]
        volatility_7 = statistics.pstdev(volatility_window) if len(volatility_window) > 1 else 0.0
        price_momentum = (float(lag_1) / float(lag_7)) if float(lag_7) != 0 else 0.0

        feature_map = {
            "day_index": day_index,
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,
            "month": month,
            "week_of_year": week_of_year,
            "lag_1": float(lag_1),
            "lag_1_change": float(lag_1_change),
            "lag_1_percent": float(lag_1_percent),
            "lag_7": float(lag_7),
            "ma_7": float(ma_7),
            "ma_30": float(ma_30),
            "volatility_7": float(volatility_7),
            "price_momentum": float(price_momentum),
        }
        raw_rows.append([float(feature_map.get(column, 0.0)) for column in feature_columns])

    return raw_rows


def plot_learning_curve(metric_payload: dict, model_name: str = "Model") -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise ImportError("matplotlib is required to plot learning curves.") from exc

    if not isinstance(metric_payload, dict) or not metric_payload:
        raise ValueError("Learning curve data is empty.")

    first_series = next(iter(metric_payload.values()))
    if not isinstance(first_series, dict) or not first_series:
        raise ValueError("Learning curve data format is invalid.")

    metric_name = next(iter(first_series.keys()))
    plt.figure(figsize=(8, 4))
    for dataset_name, dataset_metrics in metric_payload.items():
        values = dataset_metrics.get(metric_name, [])
        if not values:
            continue
        plt.plot(range(1, len(values) + 1), values, label=str(dataset_name))

    plt.title(f"{model_name} Learning Curve ({metric_name})")
    plt.xlabel("Iteration")
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_learning_curve_data(model: Any, model_type: str) -> dict:
    normalized_model_type = normalize_model_type(model_type)

    if normalized_model_type == "xgboost_regressor":
        if hasattr(model, "evals_result"):
            try:
                return dict(model.evals_result())
            except Exception:
                return {}
        return {}

    if normalized_model_type == "lightgbm_regressor":
        return dict(getattr(model, "evals_result_", {}) or {})

    if normalized_model_type == "catboost_regressor":
        if hasattr(model, "get_evals_result"):
            try:
                return dict(model.get_evals_result())
            except Exception:
                return {}
        return {}

    return {}


def get_feature_importance_rows(model: Any, model_type: str, feature_columns: list[str]) -> list[tuple[str, float]]:
    normalized_model_type = normalize_model_type(model_type)

    if normalized_model_type in {"xgboost_regressor", "lightgbm_regressor"}:
        raw_importances = getattr(model, "feature_importances_", None)
        raw_scores = list(raw_importances) if raw_importances is not None else []
    elif normalized_model_type == "catboost_regressor":
        try:
            raw_scores = list(model.get_feature_importance(type="FeatureImportance"))
        except Exception:
            raw_scores = []
    else:
        return []

    rows: list[tuple[str, float]] = []
    for index, score in enumerate(raw_scores):
        feature_name = feature_columns[index] if index < len(feature_columns) else f"feature_{index}"
        rows.append((feature_name, float(score)))

    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def get_training_fit_rows(
    points: list[TrainingPoint],
    model: Any,
    model_type: str,
    feature_columns: list[str] | None = None,
    target_mode: str | None = None,
) -> list[dict]:
    if not points:
        return []

    normalized_model_type = normalize_model_type(model_type)
    effective_feature_columns = _resolve_effective_feature_columns(
        model=model,
        model_type=normalized_model_type,
        feature_columns=feature_columns,
    )
    features, targets, _, _ = _build_training_features_for_columns(points, effective_feature_columns)

    effective_target_mode = str(target_mode or "price")

    if normalized_model_type in {"xgboost_regressor", "lightgbm_regressor", "catboost_regressor"}:
        predicted_outputs = _to_float_list(model.predict(features))
    elif normalized_model_type == "sarima_elasticnet":
        elastic_predictions = _to_float_list(model["elastic_model"].predict(features))
        sarima_predictions = _to_float_list(model["sarima_result"].get_prediction(start=0, end=len(targets) - 1).predicted_mean)
        predicted_outputs = [
            (elastic_pred + sarima_pred) / 2.0
            for elastic_pred, sarima_pred in zip(elastic_predictions, sarima_predictions)
        ]
    else:
        raise ValueError("Unsupported model type.")

    if effective_target_mode == "price_diff":
        predicted_values = []
        for index, predicted_diff in enumerate(predicted_outputs):
            previous_actual_price = float(points[index - 1][1]) if index >= 1 else float(points[0][1])
            predicted_values.append(max(previous_actual_price + float(predicted_diff), 0.0))
    else:
        predicted_values = predicted_outputs

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


def predict_next_days(
    model: Any,
    origin_date: date,
    last_date: date,
    days: int = 30,
    start_date: date | None = None,
) -> list[dict]:
    if days <= 0:
        return []

    first_prediction_date = last_date + timedelta(days=1)
    prediction_start_date = first_prediction_date if start_date is None else max(start_date, first_prediction_date)

    predictions = []
    for day_offset in range(days):
        predict_date = prediction_start_date + timedelta(days=day_offset)
        x_value = float((predict_date - origin_date).days)
        predicted_price = float(model.predict([[x_value]])[0])
        predictions.append(
            {
                X_COLUMN: predict_date.isoformat(),
                Y_COLUMN: round(max(predicted_price, 0.0), 2),
            }
        )
    return predictions


def predict_next_days_from_model(
    metadata: dict,
    model_dir: Path,
    days: int = 30,
    start_date: date | None = None,
) -> list[dict]:
    if days <= 0:
        return []

    origin_date = date.fromisoformat(str(metadata["origin_date"]))
    last_date = date.fromisoformat(str(metadata["last_date"]))
    first_prediction_date = last_date + timedelta(days=1)
    prediction_start_date = first_prediction_date if start_date is None else max(start_date, first_prediction_date)
    warmup_days = max((prediction_start_date - first_prediction_date).days, 0)
    total_steps = warmup_days + days

    model_type = normalize_model_type(str(metadata.get("model_type", "xgboost_regressor")))
    model = load_model_from_metadata(metadata, model_dir)
    feature_columns = _resolve_effective_feature_columns(
        model=model,
        model_type=model_type,
        feature_columns=[str(value) for value in metadata.get("feature_columns", [])],
    )
    previous_price = float(metadata.get("last_observed_price", 0.0) or 0.0)
    previous_change = float(metadata.get("last_observed_change", 0.0) or 0.0)
    previous_percent = float(metadata.get("last_observed_percent", 0.0) or 0.0)
    target_mode = str(metadata.get("target_mode", "price"))
    recent_prices = [float(price) for price in metadata.get("recent_prices", []) if price is not None]
    recent_feature_rows_raw = [
        [float(value) for value in row]
        for row in metadata.get("recent_feature_rows_raw", [])
        if isinstance(row, list)
    ]
    if not recent_prices:
        recent_prices = [previous_price]

    if model_type != "sarima_elasticnet":
        predictions = []
        for step in range(total_steps):
            predict_date = first_prediction_date + timedelta(days=step)
            day_index = float((predict_date - origin_date).days)
            feature_values = _build_future_feature_vector(
                predict_date=predict_date,
                day_index=day_index,
                feature_columns=feature_columns,
                previous_change=previous_change,
                previous_percent=previous_percent,
                price_history=recent_prices,
            )
            scaled_feature_values = _scale_feature_vector_with_history(feature_values, recent_feature_rows_raw)
            predicted_output = _predict_for_features(model, model_type, scaled_feature_values)
            if target_mode == "price_diff":
                predicted_price = previous_price + float(predicted_output)
            else:
                predicted_price = float(predicted_output)
            predicted_price = max(float(predicted_price), 0.0)
            predicted_change = predicted_price - previous_price
            predicted_percent = _calculate_percent_change(predicted_change, previous_price)
            previous_price = predicted_price
            previous_change = predicted_change
            previous_percent = predicted_percent
            recent_prices.append(predicted_price)
            recent_feature_rows_raw.append([float(value) for value in feature_values])
            if len(recent_feature_rows_raw) > ROLLING_SCALER_WINDOW:
                recent_feature_rows_raw = recent_feature_rows_raw[-ROLLING_SCALER_WINDOW:]
            if len(recent_prices) > 30:
                recent_prices = recent_prices[-30:]
            predictions.append(
                {
                    X_COLUMN: predict_date.isoformat(),
                    Y_COLUMN: round(predicted_price, 2),
                }
            )
        return predictions[warmup_days:]

    elastic_model = model["elastic_model"]
    sarima_result = model["sarima_result"]
    elastic_predictions: list[float] = []
    for step in range(total_steps):
        predict_date = first_prediction_date + timedelta(days=step)
        day_index = float((predict_date - origin_date).days)
        feature_values = _build_future_feature_vector(
            predict_date=predict_date,
            day_index=day_index,
            feature_columns=feature_columns,
            previous_change=previous_change,
            previous_percent=previous_percent,
            price_history=recent_prices,
        )
        scaled_feature_values = _scale_feature_vector_with_history(feature_values, recent_feature_rows_raw)
        elastic_pred = float(elastic_model.predict([scaled_feature_values])[0])
        elastic_predictions.append(elastic_pred)

        if target_mode == "price_diff":
            estimated_price = max(previous_price + elastic_pred, 0.0)
        else:
            estimated_price = max(elastic_pred, 0.0)
        estimated_change = estimated_price - previous_price
        estimated_percent = _calculate_percent_change(estimated_change, previous_price)
        previous_price = estimated_price
        previous_change = estimated_change
        previous_percent = estimated_percent
        recent_prices.append(estimated_price)
        recent_feature_rows_raw.append([float(value) for value in feature_values])
        if len(recent_feature_rows_raw) > ROLLING_SCALER_WINDOW:
            recent_feature_rows_raw = recent_feature_rows_raw[-ROLLING_SCALER_WINDOW:]
        if len(recent_prices) > 30:
            recent_prices = recent_prices[-30:]

    sarima_forecast = _to_float_list(sarima_result.get_forecast(steps=total_steps).predicted_mean)
    predictions = []
    reconstruction_price = float(metadata.get("last_observed_price", 0.0) or 0.0)
    for idx in range(total_steps):
        predict_date = first_prediction_date + timedelta(days=idx)
        combined_output = (elastic_predictions[idx] + sarima_forecast[idx]) / 2.0
        if target_mode == "price_diff":
            predicted_price = max(reconstruction_price + combined_output, 0.0)
            reconstruction_price = predicted_price
        else:
            predicted_price = combined_output
        predictions.append(
            {
                X_COLUMN: predict_date.isoformat(),
                Y_COLUMN: round(max(predicted_price, 0.0), 2),
            }
        )
    return predictions[warmup_days:]
