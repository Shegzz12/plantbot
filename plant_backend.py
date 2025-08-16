# app.py
"""
Flask hybrid server with logging + frontend endpoints:
- Loads numeric CSV (exact rule lookups)
- Loads ML fallback model (plant_model.pkl)
- /predict accepts raw values (0..200) from ESP32, maps to categories 0..6, runs rule lookup or ML fallback,
  returns JSON with numeric codes only: {"summary": <int>, "advice": <int>, "method": "rule"|"model"}
- Logs each request to logs/predict_logs.csv (raw values, mapped categories, method, summary_text, advice_text)
- Adds these helper endpoints used by the frontend:
    GET  /api/recent?limit=N     -> JSON array of recent log rows (oldest -> newest)
    GET  /download_logs          -> download CSV logs/predict_logs.csv
    GET  /download_dataset       -> download plant_dataset.csv
    POST /delete_logs            -> truncate and recreate the logs CSV header (thread-safe)
"""

from flask import Flask, request, jsonify, send_file, abort
import pandas as pd
import joblib
import os
from datetime import datetime, timezone, timedelta
import threading
from io import StringIO
from flask import render_template   # at top
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# move index_realtime.html into ./templates/
@app.route("/", methods=["GET"])
def index():
    return render_template("index_realtime.html")

DATA_CSV = "plant_dataset.csv"
MODEL_PKL = "plant_model.pkl"
LOG_DIR = "logs"
LOG_CSV = os.path.join(LOG_DIR, "predict_logs.csv")

# Column names expected by the dataset & model (must match training)
FEATURE_COLS = [
    "SoilMoistureCat",
    "AirHumidityCat",
    "AirTempCat",
    "PlantTempCat",
    "SoilTempCat"
]

# Required dataset columns for exact lookup + outputs
REQUIRED_COLUMNS = FEATURE_COLS + ["SummaryDescription", "Advice"]

# Text mappings for summary descriptions and advice (0..9)
SUMMARY_TEXT = {
    0: "All parameters optimal",
    1: "Dry soil with cool plant",
    2: "Dry soil with heat stress",
    3: "Waterlogged soil with cool plant",
    4: "Waterlogged soil with heat stress",
    5: "Low humidity combined with heat (transpiration stress)",
    6: "High humidity and waterlogging risk (disease-prone)",
    7: "Systemic heat stress (air + soil + leaves hot)",
    8: "Systemic cold stress (air/soil/leaf too cool)",
    9: "Minor/mixed deviations — small adjustments recommended"
}

ADVICE_TEXT = {
    0: "Conditions are within ideal ranges. Maintain irrigation and nutrition schedules, scout weekly for pests, keep mulch 3–5cm, and record conditions for tuning.",
    1: "Soil moisture is too low while leaf temps are cool: irrigate deeply to wet the root zone, add warm mulch, protect plants from cold at night using covers.",
    2: "Drought + heat: apply deep irrigation immediately, deploy shade cloth during peak sun, mist in cooler hours, and avoid midday foliar wetting to prevent burns.",
    3: "Waterlogged & cool: stop irrigation, improve surface drainage, gently aerate soil, apply dark mulch to raise soil temperature slightly and reduce root-rot risk.",
    4: "Waterlogged & hot: improve drainage immediately, reduce standing water, provide partial shade, pause further watering until soil drains, and monitor for disease.",
    5: "Low humidity with high temperatures: raise local humidity with intermittent misting or trays, add partial shade, and monitor plant turgor and VPD to avoid excessive transpiration.",
    6: "High humidity combined with wet soil: high disease risk — increase ventilation, space plants, remove standing water, and prune dense growth to improve airflow.",
    7: "System-wide heat stress: combine shade cloth, enhanced ventilation, deep early-morning irrigation, and consider evaporative cooling if available to protect canopy and roots.",
    8: "Systemic cold stress: protect plants at night with covers, minimize cold irrigation, consider temporary heating in protected structures, and delay transplanting until warmer.",
    9: "Minor or mixed deviations: make small targeted adjustments (slightly tweak irrigation, add thin mulch, introduce partial shade or a fan) and monitor daily for trends."
}

# Ensure data and model exist (fail early — same behaviour as before)
if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(f"{DATA_CSV} not found. Place your numeric CSV here.")

if not os.path.exists(MODEL_PKL):
    raise FileNotFoundError(f"{MODEL_PKL} not found. Place your trained model here (plant_model.pkl).")

# Load dataset used for rule lookups (must contain required columns)
df = pd.read_csv(DATA_CSV)
missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"CSV missing required columns: {missing}")

# Convert to int for safety
for c in REQUIRED_COLUMNS:
    df[c] = df[c].astype(int)

# Load ML model
model = joblib.load(MODEL_PKL)

# Make sure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Prepare log CSV header if not exists
log_lock = threading.Lock()
LOG_HEADER_COLUMNS = [
    "timestamp",
    # raw values from ESP32 (0..200)
    "soil_moist_raw", "air_hum_raw", "air_temp_raw", "plant_temp_raw", "soil_temp_raw",
    # mapped categories (0..6) saved as *_cat
    "soil_moist_cat", "air_hum_cat", "air_temp_cat", "plant_temp_cat", "soil_temp_cat",
    "method",
    "summary_text", "advice_text"
]

if not os.path.exists(LOG_CSV):
    header_df = pd.DataFrame(columns=LOG_HEADER_COLUMNS)
    header_df.to_csv(LOG_CSV, index=False)

def rule_based_lookup(readings: dict):
    """
    Exact match lookup against the dataset.
    readings keys: soil_moist, air_hum, air_temp, plant_temp, soil_temp (integers in 0..6)
    Returns (summary_code:int, advice_code:int) or (None, None) if no match.
    """
    cond = (
        (df["SoilMoistureCat"] == int(readings["soil_moist"])) &
        (df["AirHumidityCat"] == int(readings["air_hum"])) &
        (df["AirTempCat"] == int(readings["air_temp"])) &
        (df["PlantTempCat"] == int(readings["plant_temp"])) &
        (df["SoilTempCat"] == int(readings["soil_temp"]))
    )
    matched = df[cond]
    if not matched.empty:
        row = matched.iloc[0]
        return int(row["SummaryDescription"]), int(row["Advice"])
    return None, None

def ml_predict(readings: dict):
    """
    ML fallback. Build a DataFrame using the exact feature column names used during training
    to avoid scikit-learn warnings about feature names.
    readings keys: soil_moist, air_hum, air_temp, plant_temp, soil_temp (0..6)
    Returns (summary_code, advice_code)
    """
    # Map readings to the expected column names
    row = {
        "SoilMoistureCat": int(readings["soil_moist"]),
        "AirHumidityCat": int(readings["air_hum"]),
        "AirTempCat": int(readings["air_temp"]),
        "PlantTempCat": int(readings["plant_temp"]),
        "SoilTempCat": int(readings["soil_temp"])
    }
    X = pd.DataFrame([row], columns=FEATURE_COLS)

    pred = model.predict(X)  # Expecting multi-output: shape (1,2)
    # Handle different return shapes robustly:
    try:
        import numpy as np
        pred_arr = np.asarray(pred)
        if pred_arr.ndim == 2 and pred_arr.shape[1] >= 2:
            summary_pred = int(pred_arr[0, 0])
            advice_pred = int(pred_arr[0, 1])
        elif pred_arr.ndim == 1 and pred_arr.size >= 2:
            # e.g. [summary, advice]
            summary_pred = int(pred_arr[0])
            advice_pred = int(pred_arr[1])
        elif pred_arr.ndim == 1 and pred_arr.size == 1:
            # Single output, treat as summary, set advice same
            summary_pred = int(pred_arr[0])
            advice_pred = summary_pred
        else:
            # Unexpected shape
            summary_pred = int(pred_arr.flatten()[0])
            advice_pred = int(pred_arr.flatten()[-1])
    except Exception:
        # As a last resort, try indexing pred directly
        try:
            summary_pred = int(pred[0][0])
            advice_pred = int(pred[0][1])
        except Exception:
            # fallback defaults
            summary_pred = 9
            advice_pred = 9

    return summary_pred, advice_pred

# Thresholds (explicit) for mapping raw 0..200 -> categories 0..6:
# buckets: [0..28] ->0, [29..57]->1, [58..86]->2, [87..115]->3, [116..144]->4, [145..173]->5, [174..200]->6
THRESHOLDS = [28, 57, 86, 115, 144, 173, 200]

def map_raw_to_category_threshold(raw_value: int, thresholds=THRESHOLDS) -> int:
    """
    Map raw sensor value (0..200) to category 0..6 using explicit thresholds.
    Returns the category index where raw_value <= threshold.
    """
    try:
        val = int(raw_value)
    except Exception:
        val = 0
    if val < 0:
        val = 0
    if val > thresholds[-1]:
        val = thresholds[-1]
    for idx, thr in enumerate(thresholds):
        if val <= thr:
            return idx
    return len(thresholds) - 1

def log_request_response(timestamp, raw_readings, mapped_readings, method, summary_code, advice_code):
    """
    Append a log row to LOG_CSV in a thread-safe way.
    Saves raw incoming values (0..200), their mapped categories (0..6), and human-readable summary/advice text.
    """
    row = {
        "timestamp": timestamp,
        # raw values
        "soil_moist_raw": raw_readings.get("soil_moist"),
        "air_hum_raw": raw_readings.get("air_hum"),
        "air_temp_raw": raw_readings.get("air_temp"),
        "plant_temp_raw": raw_readings.get("plant_temp"),
        "soil_temp_raw": raw_readings.get("soil_temp"),
        # mapped categories
        "soil_moist_cat": mapped_readings.get("soil_moist"),
        "air_hum_cat": mapped_readings.get("air_hum"),
        "air_temp_cat": mapped_readings.get("air_temp"),
        "plant_temp_cat": mapped_readings.get("plant_temp"),
        "soil_temp_cat": mapped_readings.get("soil_temp"),
        "method": method,
        "summary_text": SUMMARY_TEXT.get(summary_code, ""),
        "advice_text": ADVICE_TEXT.get(advice_code, ""),
    }
    row_df = pd.DataFrame([row])
    with log_lock:
        row_df.to_csv(LOG_CSV, mode="a", header=False, index=False)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # parse JSON (force True to accept body even if content-type is missing)
        data = request.get_json(force=True)

        required_keys = ["soil_moist", "air_hum", "air_temp", "plant_temp", "soil_temp"]
        if not isinstance(data, dict) or not all(k in data for k in required_keys):
            return jsonify({"error": "Missing keys. Required: soil_moist, air_hum, air_temp, plant_temp, soil_temp"}), 400

        # convert to ints and validate range (ESP32 sends 0..200) -> store as raw_readings
        raw_readings = {}
        for k in required_keys:
            try:
                raw_val = int(data[k])
            except Exception:
                return jsonify({"error": f"Sensor value '{k}' must be an integer (raw 0..200)."}), 400

            # strict range check
            if raw_val < 0 or raw_val > 200:
                return jsonify({"error": f"Sensor value '{k}' out of range (0..200): {raw_val}"}), 400

            raw_readings[k] = raw_val

        # Convert raw (0..200) -> categories (0..6) using threshold mapping (these mapped values will be used by model/dataset)
        mapped_readings = {}
        for k in required_keys:
            mapped_readings[k] = map_raw_to_category_threshold(raw_readings[k])

        # 1) try exact rule lookup using mapped categories
        summary_code, advice_code = rule_based_lookup(mapped_readings)
        method = "rule"
        if summary_code is None:
            # 2) fallback to ML using mapped categories
            summary_code, advice_code = ml_predict(mapped_readings)
            method = "model"

        # Log everything (timestamp in WAT, UTC+1)
        timestamp = datetime.now(timezone(timedelta(hours=1))).replace(tzinfo=None).isoformat()
        try:
            log_request_response(timestamp, raw_readings, mapped_readings, method, summary_code, advice_code)
        except Exception as le:
            app.logger.exception("Failed to write log CSV: %s", le)

        # Return numeric-only JSON (keeping your API behaviour)
        return jsonify({"summary": int(summary_code), "advice": int(advice_code), "method": method})

    except Exception as e:
        app.logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 500


# ---------------------------
# New endpoints required by frontend
# ---------------------------

@app.route("/api/recent", methods=["GET"])
def api_recent():
    """
    Returns recent rows from the CSV logs as JSON array.
    Query param: limit (int) default 200
    Rows returned in chronological order (oldest -> newest).
    Each row is a dict matching the CSV header columns.
    """
    limit = request.args.get("limit", default=200, type=int)
    if limit <= 0:
        return jsonify([])

    # If no log file exists yet, return empty
    if not os.path.exists(LOG_CSV):
        return jsonify([])

    try:
        # Read CSV and return last `limit` rows (oldest->newest)
        df_logs = pd.read_csv(LOG_CSV)
        if df_logs.empty:
            return jsonify([])

        # Keep only the last `limit` rows but maintain chronological order
        if len(df_logs) > limit:
            df_subset = df_logs.tail(limit)
        else:
            df_subset = df_logs

        # Convert NaNs to empty strings and coerce to python types
        df_subset = df_subset.fillna("")
        records = df_subset.to_dict(orient="records")
        return jsonify(records)
    except Exception as e:
        app.logger.exception("Failed to read logs CSV")
        return jsonify({"error": str(e)}), 500


@app.route("/download_logs", methods=["GET"])
def download_logs():
    """
    Returns the logs CSV for download.
    """
    if not os.path.exists(LOG_CSV):
        return jsonify({"error": "Log file not found."}), 404
    try:
        # send_file will set correct headers for attachment download
        return send_file(LOG_CSV, as_attachment=True, download_name=os.path.basename(LOG_CSV), mimetype="text/csv")
    except TypeError:
        # For older Flask versions without download_name
        return send_file(LOG_CSV, as_attachment=True, attachment_filename=os.path.basename(LOG_CSV), mimetype="text/csv")


@app.route("/download_dataset", methods=["GET"])
def download_dataset():
    """
    Returns the original dataset CSV (plant_dataset.csv) for download.
    """
    if not os.path.exists(DATA_CSV):
        return jsonify({"error": f"{DATA_CSV} not found."}), 404
    try:
        return send_file(DATA_CSV, as_attachment=True, download_name=os.path.basename(DATA_CSV), mimetype="text/csv")
    except TypeError:
        return send_file(DATA_CSV, as_attachment=True, attachment_filename=os.path.basename(DATA_CSV), mimetype="text/csv")


@app.route("/delete_logs", methods=["POST"])
def delete_logs():
    """
    Truncate and recreate the logs CSV with header. Thread-safe.
    Called by the frontend when user requests clearing logs.
    """
    try:
        with log_lock:
            # Remove file if exists, then recreate header
            if os.path.exists(LOG_CSV):
                os.remove(LOG_CSV)
            header_df = pd.DataFrame(columns=LOG_HEADER_COLUMNS)
            header_df.to_csv(LOG_CSV, index=False)
        return jsonify({"message": "Logs cleared"}), 200
    except Exception as e:
        app.logger.exception("Failed to delete logs")
        return jsonify({"error": str(e)}), 500


# ---------------------------
# End new endpoints
# ---------------------------


if __name__ == "__main__":
    # For local testing only. In production, run under gunicorn/uwsgi/wsgi server.
    app.run(host="0.0.0.0", port=5000, debug=True)
