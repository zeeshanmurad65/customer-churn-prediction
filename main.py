"""
Customer Churn Prediction API
------------------------------
Two prediction modes:
  1. POST /predict/single -> single customer JSON in, prediction JSON out
  2. POST /predict/csv    -> CSV file in, CSV file with predictions out

Assumes a fitted sklearn/imblearn Pipeline (preprocessor + model) was saved
with joblib, e.g. joblib.dump(random_search.best_estimator_, "xgb_best.pkl").
The pipeline must include the ColumnTransformer step, so raw columns can be
passed in directly without manual encoding.
"""

import io
import math
import sqlite3
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = "xgb_best.pkl"          # change if your saved model file is named differently
BEST_THRESHOLD = 0.50                # TODO: replace with your final tuned threshold once locked in
DB_PATH = "churn.db"                  # created by running: sqlite3 churn.db < schema.sql
MODEL_VERSION = "xgb_v1"              # bump this string whenever you retrain/redeploy a new model

# Columns the model expects, in the order your training data used them.
# customerID and Churn (the target) are excluded — this is model input only.
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# ---------------------------------------------------------------------------
# App + model load
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn from Telco-style customer data.",
    version="1.0.0",
)

# Allow the local HTML/JS frontend (served from any origin/file during dev)
# to call this API from the browser. Tighten allow_origins before deploying
# this publicly — "*" is fine for local development only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None  # app still starts; endpoints will raise a clear error if hit


# ---------------------------------------------------------------------------
# Request schema for single prediction
# ---------------------------------------------------------------------------

class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, description="Months with the company")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 845.20,
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: Literal["Yes", "No"]
    threshold_used: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_model_loaded():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Expected file at '{MODEL_PATH}' — "
                   f"check it exists and was saved with joblib.dump().",
        )


def _predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the pipeline's predict_proba on a raw-feature dataframe and
    appends probability/prediction columns using BEST_THRESHOLD."""
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing required columns: {sorted(missing)}. "
                f"Columns pandas actually parsed from your file: {list(df.columns)}"
            ),
        )

    X = df[FEATURE_COLUMNS].copy()

    # Known Telco churn dataset quirk: brand-new customers (tenure == 0)
    # have a blank/space TotalCharges since they haven't been billed yet.
    # Auto-fill those specifically with 0 rather than rejecting them.
    X["tenure"] = pd.to_numeric(X["tenure"], errors="coerce")
    X["MonthlyCharges"] = pd.to_numeric(X["MonthlyCharges"], errors="coerce")
    total_charges_numeric = pd.to_numeric(X["TotalCharges"], errors="coerce")

    zero_tenure_blank = total_charges_numeric.isna() & (X["tenure"] == 0)
    total_charges_numeric[zero_tenure_blank] = 0.0
    X["TotalCharges"] = total_charges_numeric

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    if X[numeric_cols].isna().any().any():
        bad_rows = X[X[numeric_cols].isna().any(axis=1)].index.tolist()
        raise HTTPException(
            status_code=400,
            detail=(
                f"Non-numeric or blank values found in {numeric_cols} "
                f"at row(s): {bad_rows} that aren't explained by the "
                f"tenure=0/TotalCharges-blank pattern. Fix these and re-upload."
            ),
        )

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= BEST_THRESHOLD).astype(int)

    result = df.copy()
    result["churn_probability"] = probs.round(4)
    result["churn_prediction"] = ["Yes" if p == 1 else "No" for p in preds]
    return result


def _get_db_connection():
    return sqlite3.connect(DB_PATH)


def _log_batch(filename: str, row_count: int) -> int:
    """Inserts a batches row and returns its batch_id."""
    conn = _get_db_connection()
    try:
        cur = conn.execute(
            "INSERT INTO batches (filename, row_count) VALUES (?, ?)",
            (filename, row_count),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _log_predictions(result_df: pd.DataFrame, batch_id: int | None = None):
    """Writes one row per prediction into the predictions table.
    result_df must already have churn_probability / churn_prediction columns
    (i.e. it's the output of _predict_dataframe)."""
    conn = _get_db_connection()
    try:
        rows = [
            (
                batch_id,
                r["gender"], int(r["SeniorCitizen"]), r["Partner"], r["Dependents"],
                int(r["tenure"]), r["PhoneService"], r["MultipleLines"], r["InternetService"],
                r["OnlineSecurity"], r["OnlineBackup"], r["DeviceProtection"], r["TechSupport"],
                r["StreamingTV"], r["StreamingMovies"], r["Contract"], r["PaperlessBilling"],
                r["PaymentMethod"], float(r["MonthlyCharges"]), float(r["TotalCharges"]),
                float(r["churn_probability"]), r["churn_prediction"], BEST_THRESHOLD, MODEL_VERSION,
            )
            for _, r in result_df.iterrows()
        ]
        conn.executemany(
            """
            INSERT INTO predictions (
                batch_id, gender, senior_citizen, partner, dependents,
                tenure, phone_service, multiple_lines, internet_service,
                online_security, online_backup, device_protection, tech_support,
                streaming_tv, streaming_movies, contract, paperless_billing,
                payment_method, monthly_charges, total_charges,
                churn_probability, churn_prediction, threshold_used, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "endpoints": {
            "single_prediction": "POST /predict/single",
            "csv_prediction": "POST /predict/csv",
        },
        "model_loaded": model is not None,
    }


@app.post("/predict/single", response_model=PredictionResponse)
def predict_single(customer: CustomerData):
    _ensure_model_loaded()

    df = pd.DataFrame([customer.model_dump()])
    result = _predict_dataframe(df)
    _log_predictions(result, batch_id=None)

    return PredictionResponse(
        churn_probability=float(result["churn_probability"].iloc[0]),
        churn_prediction=result["churn_prediction"].iloc[0],
        threshold_used=BEST_THRESHOLD,
    )


@app.post("/predict/csv")
def predict_csv(file: UploadFile = File(...)):
    _ensure_model_loaded()

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    result = _predict_dataframe(df)

    batch_id = _log_batch(filename=file.filename, row_count=len(result))
    _log_predictions(result, batch_id=batch_id)

    # Stream the result back as a downloadable CSV
    stream = io.StringIO()
    result.to_csv(stream, index=False)
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
    )
    response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
    return response


@app.get("/monitoring/drift")
def check_drift(min_samples: int = 30):
    """Compares recent live prediction inputs against training baseline
    stats. Numeric features: flags a mean shift beyond 1.5 std devs.
    Categorical features: flags PSI > 0.2 (Population Stability Index),
    the standard threshold for 'significant drift' in industry practice."""
    conn = _get_db_connection()
    try:
        live_df = pd.read_sql("SELECT * FROM predictions", conn)
        baseline_df = pd.read_sql("SELECT * FROM training_reference_stats", conn)
    finally:
        conn.close()

    if len(live_df) < min_samples:
        return {
            "status": "insufficient_data",
            "message": f"Only {len(live_df)} logged predictions; need at least {min_samples} for a meaningful drift check.",
        }

    if baseline_df.empty:
        raise HTTPException(
            status_code=503,
            detail="No training_reference_stats found. Run the baseline-population script against your training data first.",
        )

    # DB column names are snake_case; FEATURE_COLUMNS are the original casing.
    # Map one to the other so we can look features up consistently.
    numeric_map = {
        "tenure": "tenure",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
    }
    categorical_map = {
        "gender": "gender", "SeniorCitizen": "senior_citizen", "Partner": "partner",
        "Dependents": "dependents", "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines", "InternetService": "internet_service",
        "OnlineSecurity": "online_security", "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection", "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv", "StreamingMovies": "streaming_movies",
        "Contract": "contract", "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
    }

    numeric_results = {}
    for feature_name, db_col in numeric_map.items():
        base_row = baseline_df[(baseline_df.feature_name == feature_name)]
        base_mean = base_row[base_row.stat_type == "mean"]["stat_value"].iloc[0]
        base_std = base_row[base_row.stat_type == "std"]["stat_value"].iloc[0]

        live_mean = live_df[db_col].astype(float).mean()
        shift_in_stds = abs(live_mean - base_mean) / base_std if base_std > 0 else 0

        numeric_results[feature_name] = {
            "training_mean": round(float(base_mean), 2),
            "live_mean": round(float(live_mean), 2),
            "shift_in_std_devs": round(float(shift_in_stds), 2),
            "drifted": bool(shift_in_stds > 1.5),
        }

    categorical_results = {}
    for feature_name, db_col in categorical_map.items():
        base_freqs = baseline_df[
            (baseline_df.feature_name == feature_name) & (baseline_df.stat_type == "category_freq")
        ].set_index("stat_key")["stat_value"]

        live_freqs = live_df[db_col].astype(str).value_counts(normalize=True)

        # PSI needs the same categories on both sides; missing ones get a
        # small floor value (0.0001) instead of zero, to avoid div-by-zero.
        all_categories = set(base_freqs.index) | set(live_freqs.index)
        psi = 0.0
        for cat in all_categories:
            b = max(base_freqs.get(cat, 0.0001), 0.0001)
            l = max(live_freqs.get(cat, 0.0001), 0.0001)
            psi += (l - b) * math.log(l / b)

        categorical_results[feature_name] = {
            "psi": round(float(psi), 4),
            "drifted": bool(psi > 0.2),
        }

    any_drift = any(v["drifted"] for v in numeric_results.values()) or \
                any(v["drifted"] for v in categorical_results.values())

    return {
        "status": "drift_detected" if any_drift else "stable",
        "samples_checked": len(live_df),
        "numeric_features": numeric_results,
        "categorical_features": categorical_results,
    }


@app.get("/monitoring/summary")
def monitoring_summary(days: int = 14):
    """Aggregated stats for the monitoring dashboard: daily prediction
    volume, daily churn rate, and overall totals. Drift status is fetched
    separately by the dashboard via /monitoring/drift."""
    conn = _get_db_connection()
    try:
        df = pd.read_sql("SELECT * FROM predictions", conn)
    finally:
        conn.close()

    if df.empty:
        return {
            "total_predictions": 0,
            "overall_churn_rate": None,
            "daily": [],
        }

    df["requested_at"] = pd.to_datetime(df["requested_at"])
    df["date"] = df["requested_at"].dt.date.astype(str)

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    recent = df[df["requested_at"] >= cutoff]

    daily = (
        recent.groupby("date")
        .agg(
            total=("prediction_id", "count"),
            churned=("churn_prediction", lambda s: (s == "Yes").sum()),
        )
        .reset_index()
    )
    daily["churn_rate"] = (daily["churned"] / daily["total"]).round(4)
    daily_records = daily.to_dict(orient="records")

    return {
        "total_predictions": int(len(df)),
        "overall_churn_rate": round(float((df["churn_prediction"] == "Yes").mean()), 4),
        "daily": daily_records,
    }
