-- ============================================================
-- Customer Churn Prediction — Logging & Monitoring Schema
-- Target: SQLite (works unmodified on Postgres too, minor type
-- tweaks aside — see notes at the bottom)
-- ============================================================

-- One row per batch CSV upload, so individual predictions can be
-- traced back to the file they came from.
CREATE TABLE batches (
    batch_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    uploaded_at     TEXT NOT NULL DEFAULT (datetime('now')),
    row_count       INTEGER NOT NULL
);

-- One row per prediction — whether it came from /predict/single
-- or a row inside a /predict/csv upload. This is the core log
-- everything else (dashboard, drift detection) reads from.
CREATE TABLE predictions (
    prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id            INTEGER REFERENCES batches(batch_id),  -- NULL for single predictions
    requested_at         TEXT NOT NULL DEFAULT (datetime('now')),

    -- Input features (raw, pre-encoding — matches FEATURE_COLUMNS in main.py)
    gender               TEXT NOT NULL,
    senior_citizen       INTEGER NOT NULL,
    partner              TEXT NOT NULL,
    dependents           TEXT NOT NULL,
    tenure               INTEGER NOT NULL,
    phone_service        TEXT NOT NULL,
    multiple_lines       TEXT NOT NULL,
    internet_service     TEXT NOT NULL,
    online_security      TEXT NOT NULL,
    online_backup        TEXT NOT NULL,
    device_protection    TEXT NOT NULL,
    tech_support         TEXT NOT NULL,
    streaming_tv         TEXT NOT NULL,
    streaming_movies     TEXT NOT NULL,
    contract             TEXT NOT NULL,
    paperless_billing    TEXT NOT NULL,
    payment_method       TEXT NOT NULL,
    monthly_charges      REAL NOT NULL,
    total_charges        REAL NOT NULL,

    -- Model output
    churn_probability    REAL NOT NULL,
    churn_prediction     TEXT NOT NULL CHECK (churn_prediction IN ('Yes', 'No')),
    threshold_used       REAL NOT NULL,
    model_version         TEXT NOT NULL DEFAULT 'xgb_v1'   -- bump this when you retrain/redeploy
);

CREATE INDEX idx_predictions_requested_at ON predictions(requested_at);
CREATE INDEX idx_predictions_batch_id ON predictions(batch_id);

-- Stores summary statistics from the TRAINING data, computed once
-- after training. Live traffic in `predictions` gets compared against
-- these to detect drift later (step 3).
CREATE TABLE training_reference_stats (
    feature_name    TEXT NOT NULL,
    stat_type       TEXT NOT NULL,   -- e.g. 'mean', 'std', 'category_freq'
    stat_key        TEXT,            -- category label when stat_type = 'category_freq'; NULL for numeric stats
    stat_value      REAL NOT NULL,
    computed_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (feature_name, stat_type, stat_key)
);

-- ============================================================
-- Notes:
-- 1. SQLite has no native ENUM/CHECK enforcement guarantee across
--    versions before 3.37 for some constraint types — the CHECK on
--    churn_prediction works fine on any recent SQLite (3.x) bundled
--    with Python's sqlite3 module, no action needed.
-- 2. If you migrate to Postgres later: change AUTOINCREMENT to
--    GENERATED ALWAYS AS IDENTITY, and TEXT timestamp columns to
--    TIMESTAMP WITH TIME ZONE DEFAULT now().
-- 3. senior_citizen stored as INTEGER (0/1) to match the model's
--    Literal[0, 1] input — not a boolean type, to stay consistent
--    with how the API already encodes it.
-- ============================================================
