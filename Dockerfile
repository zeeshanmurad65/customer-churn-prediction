# ---- Base image ----
FROM python:3.11-slim

WORKDIR /app

# ---- Install dependencies first (better layer caching — only
# reinstalls if requirements.txt actually changes, not on every
# code edit) ----
COPY requirements.txt .

# Split into separate layers, smallest packages first. Each successful
# layer is cached by Docker — if a later, larger package times out on a
# slow connection, re-running the build resumes from the last cached
# layer instead of re-downloading everything from scratch.
RUN pip install --no-cache-dir --default-timeout=180 --retries=15 fastapi "uvicorn[standard]" joblib python-multipart
RUN pip install --no-cache-dir --default-timeout=180 --retries=15 pandas
RUN pip install --no-cache-dir --default-timeout=180 --retries=15 scikit-learn imbalanced-learn
RUN pip install --no-cache-dir --default-timeout=180 --retries=15 xgboost

# ---- Copy application code, model, and schema ----
COPY main.py .
COPY schema.sql .
COPY xgb_best.pkl .

# ---- Initialize the SQLite DB at build time so it's baked into
# the image. If you'd rather start fresh each container run, remove
# this line and let entrypoint.sh handle it instead (see comment below). ----
RUN python -c "import sqlite3; conn = sqlite3.connect('churn.db'); conn.executescript(open('schema.sql').read()); conn.commit()"

EXPOSE 8000

# Render (and most cloud platforms) assign the port dynamically via $PORT.
# Defaults to 8000 for local `docker run` where $PORT isn't set.
ENV PORT=8000
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}