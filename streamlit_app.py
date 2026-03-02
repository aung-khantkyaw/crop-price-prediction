import csv
import json
import math
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from agrosight_scraper import OUTPUT_COLUMNS, infer_output_prefix, safe_filename, scrape
from training_model import (
    DATE_FORMATS,
    X_COLUMN,
    Y_COLUMN,
    get_training_model_options,
    get_training_fit_rows,
    is_training_model_available,
    load_dataset_points,
    load_model_from_metadata,
    load_model_metadata,
    parse_date_value,
    parse_price_value,
    predict_next_days_from_model,
    save_model_and_metadata,
    train_selected_model_from_points,
)


BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "dataset" / "csv"
JSON_DIR = BASE_DIR / "dataset" / "json"
HISTORY_PATH = BASE_DIR / "history.json"
MODEL_DIR = BASE_DIR / "Model"
SERIAL_COLUMNS = ("စဥ်", "စဉ်")
CHANGE_COLUMN = "အတက်/အကျ"
PERCENT_COLUMN = "%"


def _get_y_axis_bounds(rows: list[dict], y_columns: list[str]) -> tuple[float, float] | None:
    values: list[float] = []
    for row in rows:
        for column in y_columns:
            value = row.get(column)
            if value is None:
                continue
            try:
                numeric_value = float(value)
                if math.isfinite(numeric_value):
                    values.append(numeric_value)
            except (TypeError, ValueError):
                continue

    if not values:
        return None

    min_value = min(values) - 300
    max_value = max(values) + 300
    if min_value == max_value:
        min_value -= 300
        max_value += 300
    return min_value, max_value


def _get_price_axis_with_interval(min_value: float, max_value: float, interval: int = 100) -> alt.Axis:
    if interval <= 0:
        return alt.Axis()

    if not math.isfinite(min_value) or not math.isfinite(max_value):
        return alt.Axis()

    if min_value > max_value:
        min_value, max_value = max_value, min_value

    value_range = max_value - min_value
    if value_range <= 0 or not math.isfinite(value_range):
        return alt.Axis()

    estimated_ticks = int((value_range / interval) + 3)
    if estimated_ticks <= 0 or estimated_ticks > 400:
        return alt.Axis()

    start_value = int(min_value // interval) * interval
    end_value = int((max_value + interval - 1) // interval) * interval
    tick_values = list(range(start_value, end_value + interval, interval))

    if len(tick_values) > 400:
        return alt.Axis()
    return alt.Axis(values=tick_values)


def _get_dynamic_interval_from_range(value_range: float) -> int:
    if value_range < 1000:
        return 100
    if value_range < 50000:
        return 1000
    return 5000


def _get_dynamic_price_interval(rows: list[dict], y_columns: list[str]) -> int:
    values: list[float] = []
    for row in rows:
        for column in y_columns:
            value = row.get(column)
            if value is None:
                continue
            try:
                numeric_value = float(value)
                if math.isfinite(numeric_value):
                    values.append(numeric_value)
            except (TypeError, ValueError):
                continue

    if not values:
        return 100

    value_range = max(values) - min(values)
    return _get_dynamic_interval_from_range(value_range)


def render_line_chart(rows: list[dict], x_column: str, y_columns: list[str]) -> None:
    if not rows or not y_columns:
        st.info("No chart data to display.")
        return

    y_bounds = _get_y_axis_bounds(rows, y_columns)
    if y_bounds is None:
        st.info("No numeric data found for chart.")
        return

    data_frame = pd.DataFrame(rows)

    if len(y_columns) == 1:
        y_column = y_columns[0]
        y_interval = _get_dynamic_price_interval(rows, y_columns)
        y_axis = _get_price_axis_with_interval(y_bounds[0], y_bounds[1], interval=y_interval)
        chart = (
            alt.Chart(data_frame)
            .mark_line()
            .encode(
                x=alt.X(f"{x_column}:N", title=x_column),
                y=alt.Y(
                    f"{y_column}:Q",
                    title=y_column,
                    scale=alt.Scale(domain=[y_bounds[0], y_bounds[1]]),
                    axis=y_axis,
                ),
                tooltip=[
                    alt.Tooltip(f"{x_column}:N", title=x_column),
                    alt.Tooltip(f"{y_column}:Q", title=Y_COLUMN),
                ],
            )
            .interactive()
        )
        st.altair_chart(chart, width='stretch')
        return

    melted = data_frame.melt(id_vars=[x_column], value_vars=y_columns, var_name="model", value_name="value")
    if set(y_columns) == {"Actual", "Predicted"}:
        color_encoding = alt.Color(
            "model:N",
            title="Series",
            legend=alt.Legend(orient="bottom"),
            scale=alt.Scale(domain=["Actual", "Predicted"], range=["#1f77b4", "#ff7f0e"]),
        )
    else:
        color_encoding = alt.Color("model:N", title="Algorithm", legend=alt.Legend(orient="bottom"))

    y_interval = _get_dynamic_price_interval(rows, y_columns)
    y_axis = _get_price_axis_with_interval(y_bounds[0], y_bounds[1], interval=y_interval)
    chart = (
        alt.Chart(melted)
        .mark_line()
        .encode(
            x=alt.X(f"{x_column}:N", title=x_column),
            y=alt.Y(
                "value:Q",
                title=Y_COLUMN,
                scale=alt.Scale(domain=[y_bounds[0], y_bounds[1]]),
                axis=y_axis,
            ),
            color=color_encoding,
            tooltip=[
                alt.Tooltip(f"{x_column}:N", title=x_column),
                alt.Tooltip("model:N", title="Algorithm"),
                alt.Tooltip("value:Q", title=Y_COLUMN),
            ],
        )
        .interactive()
    )
    st.altair_chart(chart, width='stretch')


def write_outputs(rows: list[dict], output_prefix: str) -> tuple[Path, Path]:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = CSV_DIR / f"{output_prefix}.csv"
    json_path = JSON_DIR / f"{output_prefix}.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return csv_path, json_path


def load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []

    try:
        with HISTORY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def append_history(entry: dict) -> None:
    history = load_history()
    history.append(entry)

    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def log_activity(
    url: str,
    max_page: int,
    output_prefix: str,
    status: str,
    rows_scraped: int = 0,
    csv_path: str | None = None,
    json_path: str | None = None,
    error: str | None = None,
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "max_page": max_page,
        "output_prefix": output_prefix,
        "status": status,
        "rows_scraped": rows_scraped,
        "csv_path": csv_path,
        "json_path": json_path,
        "error": error,
    }
    append_history(entry)


def show_home_page() -> None:
    st.title("CPP - Crop Price Prediction")

    supported_extensions = ("*.ubj", "*.txt", "*.cbm", "*.pkl")
    model_files = []
    for ext in supported_extensions:
        model_files.extend(MODEL_DIR.glob(ext))
    model_files = sorted(model_files, key=lambda p: p.stat().st_mtime, reverse=True)

    if not model_files:
        st.info("No trained models found. Please train models first.")
        return

    today = date.today()
    prediction_rows = []

    for model_path in model_files:
        metadata_path = MODEL_DIR / f"{model_path.stem}.meta.json"
        if not metadata_path.exists():
            legacy_metadata_path = MODEL_DIR / f"{model_path.stem}.json"
            if legacy_metadata_path.exists():
                metadata_path = legacy_metadata_path

        if not metadata_path.exists():
            continue

        try:
            metadata = load_model_metadata(metadata_path)
            last_date = date.fromisoformat(str(metadata["last_date"]))
        except Exception:
            continue

        if today <= last_date:
            prediction_date = last_date + timedelta(days=1)
            days = 1
            note = "Showing next day forecast"
        else:
            prediction_date = today
            days = (today - last_date).days
            note = "Today forecast"

        try:
            predictions = predict_next_days_from_model(metadata=metadata, model_dir=MODEL_DIR, days=days)
            predicted_price = predictions[-1][Y_COLUMN] if predictions else "N/A"
        except Exception as exc:
            predicted_price = f"Error: {exc}"

        prediction_rows.append(
            {
                "model": metadata.get("model_name", model_path.stem),
                "algorithm": metadata.get("model_display_name", metadata.get("model_type", "N/A")),
                "dataset": metadata.get("dataset_file", "N/A"),
                "prediction_date": prediction_date.isoformat(),
                "predicted_price": predicted_price,
                "note": note,
            }
        )

    if not prediction_rows:
        st.info("No valid model predictions available.")
        return

    dataset_groups: dict[str, list[dict]] = {}
    for row in prediction_rows:
        dataset_name = str(row.get("dataset", "N/A"))
        dataset_groups.setdefault(dataset_name, []).append(row)

    for dataset_name in sorted(dataset_groups.keys()):
        rows = dataset_groups[dataset_name]

        item_title = dataset_name
        dataset_path = CSV_DIR / dataset_name
        if dataset_name != "N/A" and dataset_path.exists():
            try:
                with dataset_path.open("r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    first_row = next(reader, None)
                    if first_row:
                        item_name = str(first_row.get("အမျိုးအမည်", "")).strip()
                        if item_name:
                            item_title = item_name
            except OSError:
                pass

        st.markdown(f"### {item_title}")
        st.caption(f"Dataset: {dataset_name}")

        display_rows = []
        for row in rows:
            display_rows.append(
                {
                    "model": row.get("model", "N/A"),
                    "algorithm": row.get("algorithm", "N/A"),
                    "prediction_date": row.get("prediction_date", "N/A"),
                    "predicted_price": row.get("predicted_price", "N/A"),
                    "note": row.get("note", "N/A"),
                }
            )

        st.dataframe(display_rows, width='stretch')


def show_about_system_page() -> None:
    st.title("About System")
    st.write(
        "CPP is a Streamlit-based application for scraping crop price data from Agrosight, managing datasets, training forecasting models, and comparing model predictions."
    )

    st.markdown("## Data Source")
    st.markdown("All datasets are collected from:")
    st.markdown("- https://agrosightinfo.com/")

    st.markdown("## Features")
    st.markdown("- Scrape crop price table data from Agrosight URLs")
    st.markdown("- Save outputs to CSV and JSON")
    st.markdown("- Keep activity logs in `history.json`")
    st.markdown("- Read, clean, and visualize datasets")
    st.markdown("- Train multiple forecasting algorithms")
    st.markdown("- Predict next 30 days prices")
    st.markdown("- Compare models trained on the same dataset")

    st.markdown("## App Pages")

    st.markdown("### Home")
    st.markdown("Project overview, workflow explanation, model options, and comparison flow.")

    st.markdown("### Scrap Dataset")
    st.markdown("- Input: URL, Max Page, Output Prefix (optional)")
    st.markdown("- Output: CSV in `dataset/csv`, JSON in `dataset/json`")
    st.markdown("- Logs action to `history.json`")

    st.markdown("### Dataset")
    st.markdown("- Select CSV file")
    st.markdown("- Data Cleaning State with save-to-CSV action")
    st.markdown("- Replace missing/zero price on 04/05/2025 from 03/05/2025")
    st.markdown("- Fill missing days across full date range using previous day row")
    st.markdown("- Fix `စဥ်/စဉ်` serial column to sequential values when needed")
    st.markdown("- Normalize `အတက်/အကျ` and `%` (`-` → `0` and `0.00%`, including row 1)")
    st.markdown("- Recalculate `အတက်/အကျ` from day-to-day price differences")
    st.markdown("- Preview dataset table")
    st.markdown("- Filter by date range")
    st.markdown("- Line graph uses date (`နေ့စွဲ`) and price (`စျေးနှုန်း (မြန်မာကျပ်)`)")
    st.markdown("- Interval adapts by range: 1 month = daily, 6 months = weekly, 1 year = every 15 days")
    st.markdown("- Line graph Y-axis: `min_price - 1000` to `max_price + 1000`")
    st.markdown("- Monthly boxplot Y-axis: `min_price - 5000` to `max_price + 5000`")

    st.markdown("### Traing Model")
    st.markdown("Train model from selected dataset and algorithm.")
    st.markdown("Training input columns: `နေ့စွဲ`, `စျေးနှုန်း (မြန်မာကျပ်)`, `အတက်/အကျ`, `%`")
    st.markdown("Supported algorithms:")
    st.markdown("- XGBoost Regressor")
    st.markdown("- LightGBM Regressor")
    st.markdown("- CatBoostRegressor")
    st.markdown("- SARIMA + ElasticNet")
    st.markdown("Training output includes model artifact + `.meta.json` metadata and metrics (Accuracy, R², MAE, RMSE, MAPE).")

    st.markdown("### Algorithm Details (Difference, Advantage, Disadvantage)")
    st.markdown("#### 1) XGBoost Regressor")
    st.markdown("- **How it works:** Gradient boosting decision trees optimized for strong predictive performance.")
    st.markdown("- **Advantages:** High accuracy, robust on non-linear patterns, good general-purpose model.")
    st.markdown("- **Disadvantages:** More parameters to tune, can be slower than LightGBM on larger data.")

    st.markdown("#### 2) LightGBM Regressor")
    st.markdown("- **How it works:** Histogram-based gradient boosting trees with efficient training.")
    st.markdown("- **Advantages:** Very fast training, good performance, scalable to larger datasets.")
    st.markdown("- **Disadvantages:** Can overfit small/noisy datasets without careful tuning.")

    st.markdown("#### 3) CatBoostRegressor")
    st.markdown("- **How it works:** Gradient boosting with ordered boosting and strong default handling for feature patterns.")
    st.markdown("- **Advantages:** Stable performance with minimal tuning, often robust on complex patterns.")
    st.markdown("- **Disadvantages:** Training can be slower than LightGBM; model files may be larger.")

    st.markdown("#### 4) SARIMA + ElasticNet")
    st.markdown("- **How it works:** Combines classical time-series trend/seasonality modeling (SARIMA) with regularized linear regression (ElasticNet).")
    st.markdown("- **Advantages:** More interpretable, captures temporal structure, useful baseline for time-series behavior.")
    st.markdown("- **Disadvantages:** Usually less flexible for highly non-linear patterns; sensitive to parameter choices.")

    st.markdown("#### Practical Difference Summary")
    st.markdown("- **Best overall accuracy (typical):** XGBoost / CatBoost")
    st.markdown("- **Fastest training:** LightGBM")
    st.markdown("- **Most interpretable:** SARIMA + ElasticNet")
    st.markdown("- **Recommended strategy:** Train all 4 and compare by RMSE/MAE/MAPE on the same dataset.")

    st.markdown("### Model")
    st.markdown("- Select model artifact file (`.ubj`, `.txt`, `.cbm`, `.pkl`)")
    st.markdown("- Auto-load matching metadata (`.meta.json`)")
    st.markdown("- Show model metadata + metrics")
    st.markdown("- Predict next 30 days prices")
    st.markdown("- Show prediction table and line graph")

    st.markdown("### Compare Model")
    st.markdown("- Select one dataset")
    st.markdown("- Select multiple models trained on that dataset")
    st.markdown("- Compare metrics side-by-side")
    st.markdown("- Compare next 30-day predictions in one chart")
    st.markdown("- Chart labels use algorithm names")

    st.markdown("### History")
    st.markdown("Shows scraping history from `history.json`.")

    st.markdown("## Model Artifacts")
    st.markdown("- XGBoost Regressor → `.ubj`")
    st.markdown("- LightGBM Regressor → `.txt`")
    st.markdown("- CatBoostRegressor → `.cbm`")
    st.markdown("- SARIMA + ElasticNet → `.pkl`")
    st.markdown("- Each model has corresponding metadata: `<model_name>.meta.json`")

    st.markdown("## Notes")
    st.markdown("- Ensure required libraries are installed from `requirement.txt`.")
    st.markdown("- If a model fails to load, verify artifact and matching `.meta.json` in `Model/`.")
    st.markdown("- Legacy metadata compatibility is supported in model loading.")


def show_scrap_page() -> None:
    st.title("Scrap Dataset")

    with st.form("scrape_form"):
        url = st.text_input("URL", placeholder="https://...", help="Agrosight search URL")
        max_page = st.number_input("Max Page", min_value=1, value=1, step=1)
        output_prefix_input = st.text_input("Output Prefix (optional)")
        submitted = st.form_submit_button("Scrape")

    if not submitted:
        return

    if not url.strip():
        st.error("URL is required.")
        return

    output_prefix = output_prefix_input.strip() or infer_output_prefix(url)
    output_prefix = safe_filename(output_prefix)

    try:
        with st.spinner("Scraping data..."):
            rows = scrape(url=url.strip(), max_page=int(max_page))
            csv_path, json_path = write_outputs(rows, output_prefix)

        log_activity(
            url=url.strip(),
            max_page=int(max_page),
            output_prefix=output_prefix,
            status="success",
            rows_scraped=len(rows),
            csv_path=str(csv_path),
            json_path=str(json_path),
        )

        st.success(f"Done. Rows scraped: {len(rows)}")
        st.write(f"CSV: {csv_path}")
        st.write(f"JSON: {json_path}")
        st.dataframe(rows, width='stretch')
    except Exception as exc:
        log_activity(
            url=url.strip(),
            max_page=int(max_page),
            output_prefix=output_prefix,
            status="failed",
            error=str(exc),
        )
        st.error(f"Scrape failed: {exc}")


def _sort_rows_by_date(rows: list[dict]) -> list[dict]:
    dated_rows: list[tuple[date, dict]] = []
    undated_rows: list[dict] = []
    for row in rows:
        parsed_date = parse_date_value(str(row.get(X_COLUMN, "")))
        if parsed_date is None:
            undated_rows.append(row)
            continue
        dated_rows.append((parsed_date, row))

    dated_rows.sort(key=lambda item: item[0])
    return [row for _, row in dated_rows] + undated_rows


def _infer_date_format(rows: list[dict]) -> str:
    format_counter: Counter[str] = Counter()
    for row in rows:
        raw_date = str(row.get(X_COLUMN, "")).strip()
        if not raw_date:
            continue

        for fmt in DATE_FORMATS:
            try:
                datetime.strptime(raw_date, fmt)
                format_counter[fmt] += 1
                break
            except ValueError:
                continue

    if not format_counter:
        return "%d/%m/%Y"
    return format_counter.most_common(1)[0][0]


def _get_serial_column(rows: list[dict]) -> str | None:
    for column in SERIAL_COLUMNS:
        if any(column in row for row in rows):
            return column
    return None


def _format_numeric_string(value: float) -> str:
    rounded_value = round(float(value), 6)
    if float(rounded_value).is_integer():
        return str(int(rounded_value))
    return f"{rounded_value:.6f}".rstrip("0").rstrip(".")


def _apply_dataset_cleaning(rows: list[dict]) -> tuple[list[dict], list[str]]:
    cleaned_rows = [dict(row) for row in rows]
    notes: list[str] = []
    date_format = _infer_date_format(cleaned_rows)

    rows_by_date: dict[date, dict] = {}
    for row in cleaned_rows:
        parsed_date = parse_date_value(str(row.get(X_COLUMN, "")))
        if parsed_date is None:
            continue
        rows_by_date[parsed_date] = row

    def replace_if_zero_or_null(target_date: date, source_date: date) -> None:
        target_row = rows_by_date.get(target_date)
        source_row = rows_by_date.get(source_date)
        if target_row is None or source_row is None:
            return

        target_price = parse_price_value(str(target_row.get(Y_COLUMN, "")))
        source_price_raw = str(source_row.get(Y_COLUMN, "")).strip()
        source_price = parse_price_value(source_price_raw)
        if source_price is None:
            return

        if target_price is None or target_price == 0:
            target_row[Y_COLUMN] = source_price_raw
            notes.append(
                f"Replaced {target_date.strftime('%d/%m/%Y')} price with {source_date.strftime('%d/%m/%Y')} price."
            )

    def fill_all_missing_days() -> None:
        if not rows_by_date:
            return

        existing_dates = sorted(rows_by_date.keys())
        start_date = existing_dates[0]
        end_date = existing_dates[-1]
        filled_count = 0

        cursor = start_date
        while cursor <= end_date:
            if cursor not in rows_by_date:
                source_date = cursor - timedelta(days=1)
                source_row = rows_by_date.get(source_date)
                if source_row is not None:
                    new_row = dict(source_row)
                    new_row[X_COLUMN] = cursor.strftime(date_format)
                    cleaned_rows.append(new_row)
                    rows_by_date[cursor] = new_row
                    filled_count += 1
            cursor += timedelta(days=1)

        if filled_count > 0:
            notes.append(f"Filled {filled_count} missing day(s) across the full dataset range.")

    def fix_duplicate_serial_numbers(sorted_rows: list[dict]) -> None:
        serial_column = _get_serial_column(sorted_rows)
        if serial_column is None:
            return

        current_values = [str(row.get(serial_column, "")).strip() for row in sorted_rows]
        expected_values = [str(index) for index in range(1, len(sorted_rows) + 1)]

        if current_values == expected_values:
            return

        non_empty_values = [value for value in current_values if value]
        has_duplicates = len(non_empty_values) != len(set(non_empty_values))

        for index, row in enumerate(sorted_rows, start=1):
            row[serial_column] = str(index)

        if has_duplicates:
            notes.append(f"Fixed duplicate serial numbers in {serial_column} column.")
        else:
            notes.append(f"Reordered {serial_column} column to sequential values.")

    def clean_price_change_column(sorted_rows: list[dict]) -> None:
        if not sorted_rows:
            return

        changed_count = 0
        previous_price: float | None = None

        for index, row in enumerate(sorted_rows):
            current_price = parse_price_value(str(row.get(Y_COLUMN, "")))
            raw_change = str(row.get(CHANGE_COLUMN, "")).strip()
            raw_percent = str(row.get(PERCENT_COLUMN, "")).strip()

            if raw_percent == "-":
                row[PERCENT_COLUMN] = "0.00%"
                changed_count += 1

            if index == 0:
                if raw_change != "0":
                    row[CHANGE_COLUMN] = "0"
                    changed_count += 1
                if str(row.get(PERCENT_COLUMN, "")).strip() != "0.00%":
                    row[PERCENT_COLUMN] = "0.00%"
                    changed_count += 1
                if current_price is not None:
                    previous_price = current_price
                continue

            if raw_change == "-":
                row[CHANGE_COLUMN] = "0"
                changed_count += 1
                if current_price is not None:
                    previous_price = current_price
                continue

            if current_price is None or previous_price is None:
                if current_price is not None:
                    previous_price = current_price
                continue

            has_decrease_sign = any(marker in raw_change for marker in ("-", "−", "▼", "↓", "ကျ"))
            has_increase_sign = any(marker in raw_change for marker in ("+", "▲", "↑", "တက်"))

            if has_decrease_sign and not has_increase_sign:
                computed_change = current_price - previous_price
            elif has_increase_sign and not has_decrease_sign:
                computed_change = current_price - previous_price
            else:
                computed_change = current_price - previous_price

            normalized_change = _format_numeric_string(computed_change)
            if raw_change != normalized_change:
                row[CHANGE_COLUMN] = normalized_change
                changed_count += 1

            previous_price = current_price

        if changed_count > 0:
            notes.append(
                f"Updated {changed_count} value(s) in {CHANGE_COLUMN} and {PERCENT_COLUMN} during cleaning."
            )

    replace_if_zero_or_null(target_date=date(2025, 5, 4), source_date=date(2025, 5, 3))
    fill_all_missing_days()
    sorted_rows = _sort_rows_by_date(cleaned_rows)
    clean_price_change_column(sorted_rows)
    fix_duplicate_serial_numbers(sorted_rows)
    return sorted_rows, notes


def _write_cleaned_csv(file_path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    effective_fieldnames = fieldnames or list(rows[0].keys()) if rows else []
    with file_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=effective_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def show_dataset_page() -> None:
    st.title("Dataset")
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(CSV_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        st.info("No CSV datasets found in dataset/csv.")
        return

    selected_file_name = st.selectbox(
        "Select CSV file to read",
        options=[f.name for f in csv_files],
    )

    selected_file = CSV_DIR / selected_file_name
    try:
        with selected_file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        cleaned_rows, cleaning_notes = _apply_dataset_cleaning(rows)

        st.subheader("Data Cleaning State")
        if cleaning_notes:
            st.info(f"Applied {len(cleaning_notes)} cleaning change(s).")
            for note in cleaning_notes:
                st.write(f"- {note}")
            if st.button("Save Cleaned Data to CSV", key=f"save_cleaned_{selected_file_name}"):
                _write_cleaned_csv(selected_file, fieldnames, cleaned_rows)
                st.success(f"Saved cleaned data to {selected_file.name}")
                st.rerun()
        else:
            st.success("No cleaning changes needed for the selected dataset.")

        rows = cleaned_rows

        st.write(f"Rows: {len(rows)}")
        if rows:
            st.dataframe(rows, width='stretch')

            x_column = X_COLUMN
            y_column = Y_COLUMN

            item_name = str(rows[0].get("အမျိုးအမည်", "")).strip()
            market_name = str(rows[0].get("ကုန်စည်ဒိုင်", "")).strip()
            st.markdown(f"### {item_name} ({market_name})")

            if x_column not in rows[0] or y_column not in rows[0]:
                st.info("Required columns for graph are missing in this CSV file.")
                return

            parsed_rows: list[tuple] = []
            for row in rows:
                raw_date = str(row.get(x_column, "")).strip()
                if not raw_date:
                    continue

                parsed_date = parse_date_value(raw_date)

                if parsed_date is not None:
                    parsed_rows.append((parsed_date, row))

            if not parsed_rows:
                st.info("Could not parse dates in the CSV date column.")
                return

            parsed_rows.sort(key=lambda item: item[0])
            min_date = parsed_rows[0][0]
            max_date = parsed_rows[-1][0]

            start_key = f"start_date_{selected_file_name}"
            end_key = f"end_date_{selected_file_name}"

            if start_key not in st.session_state:
                st.session_state[start_key] = min_date
            if end_key not in st.session_state:
                st.session_state[end_key] = max_date

            clear_col, start_col, end_col = st.columns([0.7, 1, 1])
            with clear_col:
                st.write("")
                if st.button("Clear Filter", key=f"clear_filter_{selected_file_name}"):
                    st.session_state[start_key] = min_date
                    st.session_state[end_key] = max_date
                    st.rerun()

            with start_col:
                start_date = st.date_input(
                    "Start Date",
                    key=start_key,
                    min_value=min_date,
                    max_value=max_date,
                )
            with end_col:
                end_date = st.date_input(
                    "End Date",
                    key=end_key,
                    min_value=min_date,
                    max_value=max_date,
                )

            if start_date > end_date:
                st.warning("Start Date must be earlier than or equal to End Date.")
                return

            range_days = (end_date - start_date).days + 1
            if range_days <= 31:
                interval_days = 1
                interval_label = "Every day"
            elif range_days <= 183:
                interval_days = 7
                interval_label = "Every week"
            else:
                interval_days = 15
                interval_label = "Every 15 days"

            st.caption(f"Graph interval: {interval_label}")

            grouped_prices: dict[str, float] = {}
            boxplot_rows: list[dict] = []
            for parsed_date, row in parsed_rows:
                if parsed_date < start_date or parsed_date > end_date:
                    continue
                y_num = parse_price_value(str(row.get(y_column, "")))
                if y_num is None:
                    continue

                boxplot_rows.append(
                    {
                        "month": parsed_date.strftime("%Y-%m"),
                        y_column: y_num,
                    }
                )

                if interval_days == 1:
                    bucket_date = parsed_date
                else:
                    offset_days = (parsed_date - start_date).days
                    bucket_date = start_date + timedelta(days=(offset_days // interval_days) * interval_days)

                grouped_prices[bucket_date.isoformat()] = y_num

            if grouped_prices:
                chart_rows = [
                    {x_column: point_date, y_column: grouped_prices[point_date]}
                    for point_date in sorted(grouped_prices.keys())
                ]
                render_line_chart(chart_rows, x_column=x_column, y_columns=[y_column])
            else:
                st.info("No numeric data found for selected Y axis column.")

            if boxplot_rows:
                st.subheader("Monthly Price Boxplot")
                boxplot_df = pd.DataFrame(boxplot_rows)
                boxplot_min = float(boxplot_df[y_column].min()) - 300
                boxplot_max = float(boxplot_df[y_column].max()) + 300
                if boxplot_min == boxplot_max:
                    boxplot_min -= 300
                    boxplot_max += 300

                boxplot_range = float(boxplot_df[y_column].max()) - float(boxplot_df[y_column].min())
                boxplot_interval = _get_dynamic_interval_from_range(boxplot_range)
                boxplot_axis = _get_price_axis_with_interval(boxplot_min, boxplot_max, interval=boxplot_interval)

                boxplot_chart = (
                    alt.Chart(boxplot_df)
                    .mark_boxplot()
                    .encode(
                        x=alt.X("month:N", title="Month"),
                        y=alt.Y(
                            f"{y_column}:Q",
                            title=y_column,
                            scale=alt.Scale(domain=[boxplot_min, boxplot_max]),
                            axis=boxplot_axis,
                        ),
                        tooltip=["month", y_column],
                    )
                )
                st.altair_chart(boxplot_chart, width='stretch')
        else:
            st.info("This CSV file is empty.")
    except OSError as exc:
        st.error(f"Could not read file: {exc}")


def show_training_model_page() -> None:
    st.title("Traing Model")
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(CSV_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        st.info("No CSV datasets found in dataset/csv.")
        return

    selected_dataset = st.selectbox("Select Dataset", options=[f.name for f in csv_files])
    selected_model_name = st.selectbox("Select Model", options=get_training_model_options())
    model_name_input = st.text_input("Model Name (optional)")

    available, message = is_training_model_available(selected_model_name)
    if not available:
        st.error(f"{selected_model_name} is unavailable: {message}")
        return

    if not st.button("Train Model", width='stretch'):
        return

    dataset_path = CSV_DIR / selected_dataset

    try:
        points = load_dataset_points(dataset_path)
        model, training_info = train_selected_model_from_points(points, selected_model_name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_model_name = f"{Path(selected_dataset).stem}_{timestamp}"
        model_name = safe_filename(model_name_input.strip() or default_model_name)
        model_path, metadata_path, model_payload = save_model_and_metadata(
            model=model,
            model_dir=MODEL_DIR,
            model_name=model_name,
            dataset_file=selected_dataset,
            training_info=training_info,
        )

        st.success("Model trained and saved successfully.")
        st.write(f"Model file: {model_path}")
        st.write(f"Metadata file: {metadata_path}")
        st.write(f"Algorithm: {model_payload.get('model_display_name', selected_model_name)}")
        st.write(f"Training rows: {training_info['train_size']}")
        st.subheader("Training Metrics")
        st.write(f"Accuracy (%): {model_payload['metrics']['accuracy_percent']}")
        st.write(f"R²: {model_payload['metrics']['r2']}")
        st.write(f"MAE: {model_payload['metrics']['mae']}")
        st.write(f"RMSE: {model_payload['metrics']['rmse']}")
        st.write(f"MAPE (%): {model_payload['metrics']['mape_percent']}")

        fit_rows = get_training_fit_rows(
            points=points,
            model=model,
            model_type=str(training_info.get("model_type", "")),
        )
        if fit_rows:
            st.subheader("Actual vs Predicted (Training Dataset)")
            render_line_chart(fit_rows, x_column=X_COLUMN, y_columns=["Actual", "Predicted"])
    except Exception as exc:
        st.error(f"Model training failed: {exc}")


def show_model_page() -> None:
    st.title("Model")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    supported_extensions = ("*.ubj", "*.txt", "*.cbm", "*.pkl")
    model_files = []
    for ext in supported_extensions:
        model_files.extend(MODEL_DIR.glob(ext))
    model_files = sorted(model_files, key=lambda p: p.stat().st_mtime, reverse=True)

    if not model_files:
        st.info("No model artifacts found in Model folder (.ubj, .txt, .cbm, .pkl).")
        return

    selected_model_name = st.selectbox("Select Model", options=[f.name for f in model_files])
    selected_model_path = MODEL_DIR / selected_model_name
    selected_metadata_path = MODEL_DIR / f"{selected_model_path.stem}.meta.json"

    if not selected_metadata_path.exists():
        legacy_metadata_path = MODEL_DIR / f"{selected_model_path.stem}.json"
        if legacy_metadata_path.exists():
            selected_metadata_path = legacy_metadata_path

    if not selected_metadata_path.exists():
        st.error(f"Metadata file not found: {selected_metadata_path.name}")
        return

    try:
        model = load_model_metadata(selected_metadata_path)
    except (json.JSONDecodeError, OSError) as exc:
        st.error(f"Could not read model file: {exc}")
        return

    required_keys = {"model_file", "origin_date", "last_date"}
    if not required_keys.issubset(model.keys()):
        st.error("Selected model file is invalid.")
        return

    model_file = MODEL_DIR / str(model["model_file"])
    if not model_file.exists():
        st.error(f"Model artifact not found: {model_file.name}")
        return

    x_column = X_COLUMN
    y_column = Y_COLUMN

    metric_data = model.get("metrics", {})

    metadata_col, fit_col = st.columns([1, 1])
    with metadata_col:
        st.subheader("Model Metadata")
        st.write(f"Model: {model.get('model_name', selected_model_name)}")
        st.write(f"Algorithm: {model.get('model_display_name', model.get('model_type', 'N/A'))}")
        st.write(f"Dataset: {model.get('dataset_file', 'N/A')}")
        st.write(f"Created At: {model.get('created_at', 'N/A')}")
        st.write(f"Training Rows: {model.get('train_size', 'N/A')}")

        if not metric_data:
            st.info("Metrics are not available for this model metadata.")
        st.write(f"Accuracy (%): {metric_data.get('accuracy_percent', 'N/A')}")
        st.write(f"R²: {metric_data.get('r2', 'N/A')}")
        st.write(f"MAE: {metric_data.get('mae', 'N/A')}")
        st.write(f"RMSE: {metric_data.get('rmse', 'N/A')}")
        st.write(f"MAPE (%): {metric_data.get('mape_percent', 'N/A')}")

    with fit_col:
        st.subheader("Actual vs Predicted (Training Dataset)")
        dataset_name = str(model.get("dataset_file", "")).strip()
        dataset_path = CSV_DIR / dataset_name if dataset_name else None

        if dataset_path is None or not dataset_path.exists():
            st.info("Training dataset file is not available for this model.")
        else:
            try:
                model_object = load_model_from_metadata(model, MODEL_DIR)
                training_points = load_dataset_points(dataset_path)
                fit_rows = get_training_fit_rows(
                    points=training_points,
                    model=model_object,
                    model_type=str(model.get("model_type", "")),
                )
                if fit_rows:
                    render_line_chart(fit_rows, x_column=X_COLUMN, y_columns=["Actual", "Predicted"])
                else:
                    st.info("No valid rows available for training fit plot.")
            except Exception as exc:
                st.info(f"Could not generate training fit plot: {exc}")

    predictions = predict_next_days_from_model(metadata=model, model_dir=MODEL_DIR, days=30)

    st.subheader("Next 30 Days Price Prediction")
    st.dataframe(predictions, width='stretch')
    render_line_chart(predictions, x_column=x_column, y_columns=[y_column])


def show_compare_model_page() -> None:
    st.title("Compare Model")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metadata_paths = sorted(MODEL_DIR.glob("*.meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not metadata_paths:
        st.info("No model metadata found in Model folder.")
        return

    model_metadata: list[dict] = []
    for path in metadata_paths:
        try:
            data = load_model_metadata(path)
        except (json.JSONDecodeError, OSError):
            continue

        if not {"model_file", "dataset_file", "origin_date", "last_date"}.issubset(data.keys()):
            continue

        model_file = MODEL_DIR / str(data.get("model_file", ""))
        if not model_file.exists():
            continue

        data["__metadata_file"] = path.name
        data["__model_file"] = model_file.name
        model_metadata.append(data)

    if not model_metadata:
        st.info("No valid model metadata found for comparison.")
        return

    dataset_options = sorted({str(item.get("dataset_file", "N/A")) for item in model_metadata})
    selected_dataset = st.selectbox("Select Dataset", options=dataset_options)

    filtered_models = [item for item in model_metadata if item.get("dataset_file") == selected_dataset]
    if len(filtered_models) < 2:
        st.info("Need at least 2 models from the same dataset for comparison.")
        return

    model_labels = []
    label_to_model = {}
    for item in filtered_models:
        label = (
            f"{item.get('model_name', item['__metadata_file'])} | "
            f"{item.get('model_display_name', item.get('model_type', 'N/A'))}"
        )
        model_labels.append(label)
        label_to_model[label] = item

    selected_labels = st.multiselect(
        "Select Models to Compare",
        options=model_labels,
        default=model_labels[: min(3, len(model_labels))],
    )

    if len(selected_labels) < 2:
        st.warning("Please select at least 2 models.")
        return

    selected_models = [label_to_model[label] for label in selected_labels]
    selected_algorithms = {
        str(item.get("model_display_name", item.get("model_type", "N/A"))) for item in selected_models
    }
    if len(selected_algorithms) < 2:
        st.warning("Select models with different algorithms for better comparison.")

    comparison_rows = []
    for item in selected_models:
        metric_data = item.get("metrics", {}) or {}
        comparison_rows.append(
            {
                "model": item.get("model_name", "N/A"),
                "algorithm": item.get("model_display_name", item.get("model_type", "N/A")),
                "train_size": item.get("train_size", "N/A"),
                "accuracy_percent": metric_data.get("accuracy_percent", "N/A"),
                "r2": metric_data.get("r2", "N/A"),
                "mae": metric_data.get("mae", "N/A"),
                "rmse": metric_data.get("rmse", "N/A"),
                "mape_percent": metric_data.get("mape_percent", "N/A"),
                "model_file": item.get("model_file", "N/A"),
            }
        )

    st.subheader("Metrics Comparison")
    st.dataframe(comparison_rows, width='stretch')

    prediction_series = {}
    algorithm_label_counts: dict[str, int] = {}
    all_dates = set()
    for item in selected_models:
        try:
            predictions = predict_next_days_from_model(metadata=item, model_dir=MODEL_DIR, days=30)
        except Exception as exc:
            st.error(f"Prediction failed for {item.get('model_name', 'N/A')}: {exc}")
            continue

        base_label = str(item.get("model_display_name", item.get("model_type", "N/A")))
        algorithm_label_counts[base_label] = algorithm_label_counts.get(base_label, 0) + 1
        label_index = algorithm_label_counts[base_label]
        series_name = base_label if label_index == 1 else f"{base_label} ({label_index})"
        series_map = {row[X_COLUMN]: row[Y_COLUMN] for row in predictions}
        prediction_series[series_name] = series_map
        all_dates.update(series_map.keys())

    if not prediction_series:
        st.info("Could not build prediction comparison chart.")
        return

    chart_rows = []
    for predict_date in sorted(all_dates):
        row = {X_COLUMN: predict_date}
        for series_name, series_map in prediction_series.items():
            row[series_name] = series_map.get(predict_date)
        chart_rows.append(row)

    st.subheader("Next 30 Days Prediction Comparison")
    render_line_chart(chart_rows, x_column=X_COLUMN, y_columns=list(prediction_series.keys()))


def show_history_page() -> None:
    st.title("History")
    history = load_history()

    if not history:
        st.info("No activity found in history.json.")
        return

    st.dataframe(list(reversed(history)), width='stretch')


def main() -> None:
    st.set_page_config(page_title="Crop Price Prediction", layout="wide")

    st.sidebar.title("Crop Price Prediction")
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    if st.sidebar.button("Home", width='stretch'):
        st.session_state["page"] = "Home"
    if st.sidebar.button("Scrap Dataset", width='stretch'):
        st.session_state["page"] = "Scrap"
    if st.sidebar.button("Dataset", width='stretch'):
        st.session_state["page"] = "Dataset"
    if st.sidebar.button("Traing Model", width='stretch'):
        st.session_state["page"] = "Traing Model"
    if st.sidebar.button("Model", width='stretch'):
        st.session_state["page"] = "Model"
    if st.sidebar.button("Compare Model", width='stretch'):
        st.session_state["page"] = "Compare Model"
    if st.sidebar.button("About System", width='stretch'):
        st.session_state["page"] = "About System"
    # if st.sidebar.button("History", width='stretch'):
    #     st.session_state["page"] = "History"

    page = st.session_state["page"]
    st.sidebar.caption(f"Current page: {page}")

    if page == "Home":
        show_home_page()
    elif page == "About System":
        show_about_system_page()
    elif page == "Scrap":
        show_scrap_page()
    elif page == "Dataset":
        show_dataset_page()
    elif page == "Traing Model":
        show_training_model_page()
    elif page == "Model":
        show_model_page()
    elif page == "Compare Model":
        show_compare_model_page()
    else:
        show_history_page()


if __name__ == "__main__":
    main()
