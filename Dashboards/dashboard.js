const API_BASE = "http://127.0.0.1:8000";

async function loadDashboard() {
  try {
    const [summaryRes, driftRes] = await Promise.all([
      fetch(`${API_BASE}/monitoring/summary`),
      fetch(`${API_BASE}/monitoring/drift`),
    ]);

    const summary = await summaryRes.json();
    const drift = await driftRes.json();

    renderStats(summary, drift);
    renderChart(summary.daily);
    renderDriftDetail(drift);
  } catch (err) {
    document.getElementById("stat-total").textContent = "Error";
    console.error("Failed to load dashboard:", err);
  }
}

function renderStats(summary, drift) {
  document.getElementById("stat-total").textContent = summary.total_predictions ?? "0";

  const churnRate = summary.overall_churn_rate;
  document.getElementById("stat-churn-rate").textContent =
    churnRate == null ? "—" : `${Math.round(churnRate * 100)}%`;

  const driftCard = document.getElementById("stat-drift-card");
  const driftValue = document.getElementById("stat-drift");

  if (drift.status === "insufficient_data") {
    driftValue.textContent = "Not enough data";
    driftCard.className = "stat-card";
  } else if (drift.status === "drift_detected") {
    driftValue.textContent = "Drift detected";
    driftCard.className = "stat-card drift-alert";
  } else {
    driftValue.textContent = "Stable";
    driftCard.className = "stat-card drift-stable";
  }
}

function renderChart(daily) {
  const ctx = document.getElementById("volume-chart");
  const labels = daily.map((d) => d.date);
  const volumes = daily.map((d) => d.total);
  const churnRates = daily.map((d) => Math.round(d.churn_rate * 100));

  new Chart(ctx, {
    data: {
      labels,
      datasets: [
        {
          type: "bar",
          label: "Predictions",
          data: volumes,
          backgroundColor: "#232A31",
          yAxisID: "yVolume",
        },
        {
          type: "line",
          label: "Churn rate (%)",
          data: churnRates,
          borderColor: "#E2A33B",
          backgroundColor: "#E2A33B",
          tension: 0.3,
          yAxisID: "yRate",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#8A929B" } },
      },
      scales: {
        x: { ticks: { color: "#8A929B" }, grid: { color: "#262C33" } },
        yVolume: {
          position: "left",
          ticks: { color: "#8A929B" },
          grid: { color: "#262C33" },
          title: { display: true, text: "Predictions", color: "#8A929B" },
        },
        yRate: {
          position: "right",
          ticks: { color: "#8A929B" },
          grid: { display: false },
          title: { display: true, text: "Churn rate (%)", color: "#8A929B" },
          min: 0,
          max: 100,
        },
      },
    },
  });
}

function renderDriftDetail(drift) {
  const grid = document.getElementById("drift-grid");
  const empty = document.getElementById("drift-empty");

  if (drift.status === "insufficient_data") {
    empty.textContent = drift.message || "Not enough data yet.";
    return;
  }
  empty.remove();

  const items = [];

  for (const [feature, stats] of Object.entries(drift.numeric_features || {})) {
    items.push(`
      <div class="drift-item ${stats.drifted ? "flagged" : "ok"}">
        <span class="feature-name">${feature}</span>
        <span class="feature-metric">train mean ${stats.training_mean} → live ${stats.live_mean}</span>
        <span class="feature-metric">${stats.shift_in_std_devs} std devs shifted</span>
      </div>
    `);
  }

  for (const [feature, stats] of Object.entries(drift.categorical_features || {})) {
    items.push(`
      <div class="drift-item ${stats.drifted ? "flagged" : "ok"}">
        <span class="feature-name">${feature}</span>
        <span class="feature-metric">PSI ${stats.psi}</span>
      </div>
    `);
  }

  grid.insertAdjacentHTML("beforeend", items.join(""));
}

loadDashboard();
