const API_BASE = "http://127.0.0.1:8000";
const GAUGE_CIRCUMFERENCE = 314; // matches stroke-dasharray in CSS

// ---------- Single customer prediction ----------

const form = document.getElementById("customer-form");
const formError = document.getElementById("form-error");
const gaugeFill = document.getElementById("gauge-fill");
const gaugeValue = document.getElementById("gauge-value");
const gaugeCaption = document.getElementById("gauge-caption");
const verdictBlock = document.getElementById("verdict-block");
const verdictTag = document.getElementById("verdict-tag");
const verdictText = document.getElementById("verdict-text");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  formError.textContent = "";

  const formData = new FormData(form);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    if (key === "SeniorCitizen") {
      payload[key] = parseInt(value, 10);
    } else if (key === "tenure") {
      payload[key] = parseInt(value, 10);
    } else if (key === "MonthlyCharges" || key === "TotalCharges") {
      payload[key] = parseFloat(value);
    } else {
      payload[key] = value;
    }
  }

  const submitBtn = form.querySelector(".btn-primary");
  submitBtn.disabled = true;
  submitBtn.textContent = "Reading…";

  try {
    const res = await fetch(`${API_BASE}/predict/single`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }

    const result = await res.json();
    renderGauge(result.churn_probability, result.churn_prediction);
  } catch (err) {
    formError.textContent = `Couldn't reach the model: ${err.message}. Is the API running at ${API_BASE}?`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Read signal";
  }
});

function renderGauge(probability, prediction) {
  const pct = Math.round(probability * 100);
  const offset = GAUGE_CIRCUMFERENCE * (1 - probability);

  gaugeFill.style.strokeDashoffset = offset;
  gaugeFill.style.stroke = prediction === "Yes" ? "var(--amber)" : "var(--teal)";
  gaugeValue.textContent = `${pct}%`;
  gaugeCaption.textContent = "Predicted churn probability";

  verdictBlock.hidden = false;
  verdictTag.textContent = prediction === "Yes" ? "At risk" : "Likely to stay";
  verdictTag.className = `verdict-tag ${prediction === "Yes" ? "risk" : "safe"}`;
  verdictText.textContent =
    prediction === "Yes"
      ? "This account's profile matches customers who tend to leave. Worth a retention touchpoint."
      : "This account's profile matches customers who tend to stay. No action needed right now.";
}

// ---------- Batch CSV prediction ----------

const dropzone = document.getElementById("dropzone");
const csvInput = document.getElementById("csv-input");
const browseBtn = document.getElementById("browse-btn");
const dropzoneFilename = document.getElementById("dropzone-filename");
const batchSubmit = document.getElementById("batch-submit");
const batchError = document.getElementById("batch-error");
const batchResults = document.getElementById("batch-results");
const batchSummary = document.getElementById("batch-summary");
const downloadLink = document.getElementById("download-link");

let selectedFile = null;

browseBtn.addEventListener("click", () => csvInput.click());

csvInput.addEventListener("change", () => {
  if (csvInput.files.length) setSelectedFile(csvInput.files[0]);
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  })
);

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) setSelectedFile(file);
});

function setSelectedFile(file) {
  if (!file.name.endsWith(".csv")) {
    batchError.textContent = "That file isn't a CSV — pick a .csv file.";
    return;
  }
  batchError.textContent = "";
  selectedFile = file;
  dropzoneFilename.textContent = file.name;
  batchSubmit.disabled = false;
}

batchSubmit.addEventListener("click", async () => {
  if (!selectedFile) return;

  batchError.textContent = "";
  batchResults.hidden = true;
  batchSubmit.disabled = true;
  batchSubmit.textContent = "Scoring…";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch(`${API_BASE}/predict/csv`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    downloadLink.href = url;

    const rowCount = await countCsvRows(blob);
    batchSummary.innerHTML = `Scored <strong>${rowCount}</strong> customer${rowCount === 1 ? "" : "s"}.`;
    batchResults.hidden = false;
  } catch (err) {
    batchError.textContent = `Couldn't score the file: ${err.message}. Is the API running at ${API_BASE}?`;
  } finally {
    batchSubmit.disabled = false;
    batchSubmit.textContent = "Score file";
  }
});

async function countCsvRows(blob) {
  const text = await blob.text();
  const lines = text.trim().split("\n");
  return Math.max(lines.length - 1, 0); // minus header row
}
