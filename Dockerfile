# ---- Base image ----
FROM python:3.11-slim

WORKDIR /app

# ---- Install dependencies first (better layer caching — only
# reinstalls if requirements.txt actually changes, not on every
# code edit) ----
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries=10 -r requirements.txt

# ---- Copy application code, model, and schema ----
COPY main.py .
COPY schema.sql .
COPY xgb_best.pkl .

# ---- Initialize the SQLite DB at build time so it's baked into
# the image. If you'd rather start fresh each container run, remove
# this line and let entrypoint.sh handle it instead (see comment below). ----
RUN python -c "import sqlite3; conn = sqlite3.connect('churn.db'); conn.executescript(open('schema.sql').read()); conn.commit()"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]