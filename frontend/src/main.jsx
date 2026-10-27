import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const metrics = [
  { model: "Ensemble", ap: 0.8514, auc: 0.9825 },
  { model: "Random Forest", ap: 0.8378, auc: 0.9826 },
  { model: "XGBoost", ap: 0.8027, auc: 0.9782 },
  { model: "LightGBM", ap: 0.7792, auc: 0.9757 },
];
const features = Array.from({ length: 28 }, (_, i) => `V${i + 1}`);
function Icon({ children }) { return <span className="icon" aria-hidden="true">{children}</span>; }

function App() {
  const [page, setPage] = useState("Overview");
  const [apiStatus, setApiStatus] = useState("checking");
  const [amount, setAmount] = useState("150");
  const [values, setValues] = useState(Object.fromEntries(features.map((f) => [f, "0"])));
  const [threshold, setThreshold] = useState(0.5);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [batchRows, setBatchRows] = useState([]);
  const [batchResult, setBatchResult] = useState(null);
  const [capacity, setCapacity] = useState(0.5);

  React.useEffect(() => {
    fetch(`${API}/health`).then(r => r.json()).then(d => setApiStatus(d.model_loaded ? "online" : "missing"))
      .catch(() => setApiStatus("offline"));
  }, []);

  const highAP = useMemo(() => Math.max(...metrics.map(m => m.ap)), []);
  async function analyze(e) {
    e.preventDefault(); setBusy(true); setError(""); setResult(null);
    try {
      const response = await fetch(`${API}/predict`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ amount: Number(amount), features: Object.fromEntries(features.map(f => [f, Number(values[f])])), threshold })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Prediction failed.");
      setResult(data);
    } catch (err) { setError(err.message || "Could not connect to the Python API."); }
    finally { setBusy(false); }
  }
  async function runBatch() {
    setBusy(true); setError(""); setBatchResult(null);
    try {
      const response = await fetch(`${API}/predict/batch`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows: batchRows, policy: "capacity", capacity_percent: capacity, threshold })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Batch prediction failed.");
      setBatchResult(data);
    } catch (err) { setError(err.message || "Could not connect to the Python API."); }
    finally { setBusy(false); }
  }
  function readCsv(file) {
    if (!file) return;
    setError(""); setBatchResult(null);
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const lines = String(reader.result).trim().split(/\r?\n/);
        if (lines.length < 2) throw new Error("CSV needs a header and at least one transaction.");
        const headers = lines[0].split(",").map(s => s.trim());
        const required = [...features, headers.includes("scaled_amount") ? "scaled_amount" : "Amount"];
        const missing = required.filter(h => !headers.includes(h));
        if (missing.length) throw new Error(`Missing columns: ${missing.join(", ")}`);
        const rows = lines.slice(1).filter(Boolean).map(line => {
          const cells = line.split(",");
          return Object.fromEntries(headers.map((h, i) => [h, cells[i]?.trim() ?? ""]));
        });
        setBatchRows(rows);
      } catch (err) { setError(err.message); }
    };
    reader.readAsText(file);
  }
  function downloadCsv() {
    if (!batchResult?.results?.length) return;
    const rows = batchResult.results;
    const keys = Object.keys(rows[0]);
    const csv = [keys.join(","), ...rows.map(row => keys.map(k => JSON.stringify(row[k] ?? "")).join(","))].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = document.createElement("a"); a.href = url; a.download = "fraudwatch-results.csv"; a.click(); URL.revokeObjectURL(url);
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand-mark">F<span>!</span></div>
      <div className="brand-copy"><strong>FRAUDWATCH</strong><small>RISK INTELLIGENCE / 01</small></div>
      <div className="side-label">WORKSPACE</div>
      {[
        ["Overview", "▦"], ["Analyze transaction", "◎"], ["Batch review", "↗"], ["Model lab", "⌁"]
      ].map(([label, icon]) => <button key={label} className={page === label ? "nav-item active" : "nav-item"} onClick={() => { setPage(label); setError(""); }}>
        <span>{icon}</span>{label}<b>{page === label ? "↗" : ""}</b>
      </button>)}
      <div className="sidebar-bottom"><div className="status-line"><span className={apiStatus === "online" ? "dot live" : "dot"} /> API {apiStatus === "online" ? "CONNECTED" : apiStatus === "checking" ? "CHECKING" : apiStatus === "missing" ? "MODEL MISSING" : "OFFLINE"}</div><small>ENSEMBLE MODEL · V1.0</small></div>
    </aside>

    <main className="main">
      <header className="topbar"><div><span className="eyebrow">FRAUD DETECTION SYSTEM / LIVE WORKSPACE</span><h1>{page === "Overview" ? "Risk overview" : page}</h1></div><div className="topbar-right"><span className="date-tag">● MODEL MONITOR</span><div className="avatar">FW</div></div></header>
      {error && <div className="error-banner"><b>Something needs attention</b><span>{error}</span><button onClick={() => setError("")}>×</button></div>}

      {page === "Overview" && <><section className="hero-card"><div><span className="hero-kicker">MODEL SNAPSHOT / TEST EXPERIMENT</span><h2>Catch the signal.<br/><em>Prioritize the risk.</em></h2><p>Imbalanced-data learning, ensemble predictions, and review-capacity controls in one workspace.</p><button className="button black" onClick={() => setPage("Analyze transaction")}>Analyze a transaction <span>↗</span></button></div><div className="hero-art"><div className="target target-one"/><div className="target target-two"/><div className="hero-stamp">RISK<br/>≠<br/>CERTAINTY</div><span className="hero-index">FW / 2026</span></div></section>
        <section className="metric-grid">
          <div className="metric-card yellow"><div className="metric-label">ENSEMBLE AVG. PRECISION <span>↗</span></div><div className="metric-value">0.8514</div><div className="metric-foot"><span className="pill">AP / AUPRC</span><span>Reported test result</span></div></div>
          <div className="metric-card"><div className="metric-label">ROC — AUC <span>↗</span></div><div className="metric-value">0.9825</div><div className="metric-foot"><span className="pill green">STRONG RANKING</span><span>Reported test result</span></div></div>
          <div className="metric-card"><div className="metric-label">BASELINE LEADER <span>↗</span></div><div className="metric-value small-value">R. Forest</div><div className="metric-foot"><span className="pill">ROC-AUC 0.9826</span><span>Near-tie with ensemble</span></div></div>
          <div className="metric-card"><div className="metric-label">FEATURE SPACE <span>↗</span></div><div className="metric-value">28<span className="unit"> PCA</span></div><div className="metric-foot"><span className="pill">V1—V28</span><span>Plus transaction amount</span></div></div>
        </section>
        <section className="content-grid"><div className="panel chart-panel"><div className="panel-heading"><div><span className="eyebrow">MODEL LAB / 01</span><h3>Model performance</h3></div><span className="tag">AP COMPARISON</span></div><div className="chart-wrap"><ResponsiveContainer width="100%" height="100%"><BarChart data={metrics} margin={{ top: 12, right: 12, left: -20, bottom: 6 }}><CartesianGrid stroke="#dedbd2" strokeDasharray="4 4" vertical={false}/><XAxis dataKey="model" tick={{ fill: "#333", fontSize: 11 }} axisLine={false} tickLine={false}/><YAxis domain={[0,1]} tick={{ fill: "#555", fontSize: 11 }} axisLine={false} tickLine={false}/><Tooltip contentStyle={{ border: "2px solid #171717", borderRadius: 0, boxShadow: "4px 4px 0 #171717" }}/><Bar dataKey="ap" name="Average Precision" fill="#f4d44d" stroke="#171717" strokeWidth={1.5} /></BarChart></ResponsiveContainer></div><div className="chart-legend"><span><i className="legend-dot yellow-dot"/> Average Precision (higher is better)</span><button className="text-button" onClick={() => setPage("Model lab")}>Explore metrics ↗</button></div></div>
          <div className="panel insight-panel"><div className="panel-heading"><div><span className="eyebrow">READOUT / 02</span><h3>What this means</h3></div><span className="asterisk">✳</span></div><div className="insight"><span className="insight-no">01</span><div><b>Ensemble leads on AP</b><p>0.8514 versus 0.8378 for Random Forest in the reported experiment.</p></div></div><div className="insight"><span className="insight-no">02</span><div><b>Ranking ≠ final decision</b><p>Threshold and review capacity change alert volume; neither changes model probabilities.</p></div></div><div className="insight"><span className="insight-no">03</span><div><b>Research-grade caution</b><p>Results are historical notebook metrics, not a live performance guarantee.</p></div></div></div></section>
        <div className="footnote"><b>DATA NOTE</b> This dashboard shows reported experiment metrics. Live prediction requires local model and scaler artifacts in the Python API. Never treat a model flag as confirmed fraud without investigation.</div>
      </>}

      {page === "Analyze transaction" && <section className="panel tool-panel"><div className="panel-heading"><div><span className="eyebrow">INFERENCE / 01</span><h3>Analyze one transaction</h3><p>Enter the transaction amount and the 28 anonymized PCA components.</p></div><span className="tag">SINGLE RECORD</span></div><form onSubmit={analyze}><label className="field-label">Transaction amount <span>RAW AMOUNT</span></label><input className="amount-input" type="number" min="0" step="0.01" value={amount} onChange={e => setAmount(e.target.value)} required/><div className="feature-header"><b>Principal components</b><span>V1 — V28</span><button type="button" className="text-button" onClick={() => setValues(Object.fromEntries(features.map(f => [f, "0"])))}>Reset all</button></div><div className="feature-grid">{features.map(f => <label key={f} className="feature-field"><span>{f}</span><input type="number" step="any" value={values[f]} onChange={e => setValues(v => ({...v, [f]: e.target.value}))} required/></label>)}</div><div className="threshold-row"><label>Decision threshold <b>{threshold.toFixed(2)}</b><input type="range" min="0.01" max="0.99" step="0.01" value={threshold} onChange={e => setThreshold(Number(e.target.value))}/></label><button className="button black" disabled={busy}>{busy ? "Analyzing…" : "Run risk analysis ↗"}</button></div></form>{result && <div className={result.prediction === 1 ? "result-card danger" : "result-card safe"}><div><span className="eyebrow">MODEL OUTPUT</span><h3>{result.prediction === 1 ? "Flagged for review" : "Below decision threshold"}</h3><p>{result.prediction === 1 ? "This transaction crossed the selected probability threshold. It is not a confirmed fraud." : "The predicted score is below the selected threshold; this does not guarantee the transaction is legitimate."}</p></div><div className="result-score"><strong>{(result.probability * 100).toFixed(2)}%</strong><span>FRAUD PROBABILITY</span><div className="score-track"><i style={{width: `${result.probability * 100}%`}}/></div></div></div>}</section>}

      {page === "Batch review" && <section className="panel tool-panel"><div className="panel-heading"><div><span className="eyebrow">BATCH INFERENCE / 02</span><h3>Review queue</h3><p>Upload a CSV with V1–V28 and Amount (or scaled_amount).</p></div><span className="tag">CSV / BULK</span></div><label className="upload-box"><span className="upload-icon">⇧</span><b>Drop a CSV or browse files</b><small>Expected columns: V1–V28 and Amount / scaled_amount</small><input type="file" accept=".csv,text/csv" onChange={e => readCsv(e.target.files?.[0])}/></label>{batchRows.length > 0 && <><div className="batch-meta"><b>{batchRows.length.toLocaleString()} transactions loaded</b><label>Review capacity <select value={capacity} onChange={e => setCapacity(Number(e.target.value))}>{[0.5,1,2,5,10].map(n => <option key={n} value={n}>{n}%</option>)}</select></label><button className="button black" disabled={busy} onClick={runBatch}>{busy ? "Scoring…" : "Score & prioritize ↗"}</button></div></>}{batchResult && <><div className="batch-stats"><div><small>ROWS SCORED</small><b>{batchResult.total.toLocaleString()}</b></div><div><small>PRIORITY REVIEW</small><b>{batchResult.review_count.toLocaleString()}</b></div><div><small>CAPACITY</small><b>{capacity}%</b></div></div><div className="table-wrap"><table><thead><tr><th>ROW</th><th>RISK SCORE</th><th>PROBABILITY</th><th>REVIEW STATUS</th></tr></thead><tbody>{batchResult.results.slice(0,50).map((r,i) => <tr key={i}><td>TX-{String(r.row_number).padStart(4,"0")}</td><td><b>{r.risk_score.toFixed(2)}</b></td><td>{(r.probability*100).toFixed(2)}%</td><td><span className={r.priority ? "status-chip review" : "status-chip"}>{r.priority ? "PRIORITY REVIEW" : "NORMAL"}</span></td></tr>)}</tbody></table></div><button className="button yellow-button" onClick={downloadCsv}>Download all results ↓</button><p className="microcopy">Priority review is a ranking decision, not confirmation of fraud.</p></>}</section>}

      {page === "Model lab" && <><section className="metric-grid model-metrics"><div className="metric-card yellow"><div className="metric-label">BEST AVERAGE PRECISION</div><div className="metric-value">0.8514</div><div className="metric-foot">Proposed ensemble</div></div><div className="metric-card"><div className="metric-label">BEST ROC-AUC</div><div className="metric-value">0.9826</div><div className="metric-foot">Random Forest</div></div><div className="metric-card"><div className="metric-label">ENSEMBLE INFERENCE</div><div className="metric-value">1.43<span className="unit"> s</span></div><div className="metric-foot">Reported measurement</div></div><div className="metric-card"><div className="metric-label">CLASS IMBALANCE</div><div className="metric-value">0.173<span className="unit">%</span></div><div className="metric-foot">Fraud in source dataset</div></div></section><section className="panel"><div className="panel-heading"><div><span className="eyebrow">BENCHMARK / REPORTED EXPERIMENT</span><h3>Compare model metrics</h3></div><span className="tag">HELD-OUT TEST</span></div><div className="table-wrap"><table><thead><tr><th>MODEL</th><th>AVERAGE PRECISION</th><th>ROC-AUC</th><th>PREDICTION TIME</th></tr></thead><tbody>{[{model:"Proposed Ensemble",ap:.8514,auc:.9825,time:1.4253},{model:"Random Forest",ap:.8378,auc:.9826,time:.783},{model:"XGBoost",ap:.8027,auc:.9782,time:.2038},{model:"LightGBM",ap:.7792,auc:.9757,time:.2954}].map(m=><tr key={m.model}><td><b>{m.model}</b></td><td>{m.ap.toFixed(4)}<div className="mini-bar"><i style={{width:`${m.ap*100}%`}}/></div></td><td>{m.auc.toFixed(4)}</td><td>{m.time.toFixed(4)} s</td></tr>)}</tbody></table></div><div className="footnote"><b>IMPORTANT</b> These are reported notebook experiment results, not live metrics from this browser. AP is particularly informative for rare fraud classes. The paper-inspired improvements documented in the README remain proposed experiments until tested.</div></section></>}
      <footer className="footer"><span>FRAUDWATCH / BUILT FOR RESEARCH</span><span>NEO-BRUTALIST INTERFACE · PYTHON ML API</span></footer>
    </main>
  </div>;
}
createRoot(document.getElementById("root")).render(<App />);
