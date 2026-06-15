# app.py  —  Agentic AI IDS  |  Live Dashboard
# Run:  streamlit run app.py
# Keep orchestrator_master.py running in a second terminal simultaneously.

import streamlit as st
import json, time, os, csv
from collections import deque
import pandas as pd

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic AI IDS — Healthcare Network",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for dark card feel ──────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"]   { font-size: 1.6rem; font-weight: 700; }
[data-testid="stMetricLabel"]   { font-size: 0.8rem; color: #aaa; }
.attack-banner  { background:#3d0000; border:1px solid #ff4b4b;
                  border-radius:8px; padding:12px 20px;
                  color:#ff4b4b; font-size:1.3rem; font-weight:700; }
.benign-banner  { background:#003d1a; border:1px solid #21c45d;
                  border-radius:8px; padding:12px 20px;
                  color:#21c45d; font-size:1.3rem; font-weight:700; }
.complete-banner{ background:#001f3d; border:1px solid #3b82f6;
                  border-radius:8px; padding:12px 20px;
                  color:#3b82f6; font-size:1.1rem; font-weight:600; }
.section-header { font-size:1.05rem; font-weight:600;
                  color:#ddd; margin-bottom:4px; }
</style>
""", unsafe_allow_html=True)

FEED_FILE    = "dashboard_feed.json"
RESULTS_FILE = "Master_Evaluation_Table.csv"
QUEUE_FILE   = "master_retrain_queue.csv"

DATASET_ORDER = ["CIC_IoMT_2024", "BOT_IOT", "TON_IOT", "CIC_DDoS", "CIC_IDS"]
DATASET_NICE  = {
    "CIC_IoMT_2024": "CIC IoMT 2024",
    "BOT_IOT":       "BoT-IoT",
    "TON_IOT":       "ToN-IoT",
    "CIC_DDoS":      "CIC-DDoS 2019",
    "CIC_IDS":       "CIC-IDS 2018",
}

# ── Persistent history in session state ───────────────────────
if "acc_history"   not in st.session_state: st.session_state.acc_history   = deque(maxlen=200)
if "trust_history" not in st.session_state: st.session_state.trust_history = deque(maxlen=200)
if "completed"     not in st.session_state: st.session_state.completed     = {}   # ds_name → accuracy string
if "last_ds"       not in st.session_state: st.session_state.last_ds       = None

# ── Helper: read JSON safely ───────────────────────────────────
def read_feed():
    try:
        with open(FEED_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None

# ── Helper: read completed results CSV ────────────────────────
def read_results():
    if not os.path.exists(RESULTS_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(RESULTS_FILE)
    except Exception:
        return pd.DataFrame()

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("## 🛡️ Agentic AI IDS — Healthcare Network Protection")
st.caption("Live orchestrator feed  •  Three-agent Bayesian fusion  •  Active learning trust adaptation")
st.divider()

# ══════════════════════════════════════════════════════════════
#  ROW 1 — Top KPI metrics  (placeholders updated every tick)
# ══════════════════════════════════════════════════════════════
kpi_cols = st.columns(7)
ph_dataset   = kpi_cols[0].empty()
ph_progress  = kpi_cols[1].empty()
ph_accuracy  = kpi_cols[2].empty()
ph_tr_recon  = kpi_cols[3].empty()
ph_tr_mqtt   = kpi_cols[4].empty()
ph_tr_vol    = kpi_cols[5].empty()
ph_queue     = kpi_cols[6].empty()

st.divider()

# ══════════════════════════════════════════════════════════════
#  ROW 2 — Agent probabilities + decision
# ══════════════════════════════════════════════════════════════
agent_col, decision_col = st.columns([3, 2])

with agent_col:
    st.markdown('<p class="section-header">Agent Threat Probabilities</p>', unsafe_allow_html=True)
    ph_bar_recon = st.empty()
    ph_bar_mqtt  = st.empty()
    ph_bar_vol   = st.empty()
    ph_bar_fused = st.empty()

with decision_col:
    st.markdown('<p class="section-header">Orchestrator Decision</p>', unsafe_allow_html=True)
    ph_decision  = st.empty()
    ph_flow_info = st.empty()

st.divider()

# ══════════════════════════════════════════════════════════════
#  ROW 3 — Live charts
# ══════════════════════════════════════════════════════════════
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown('<p class="section-header">📈 Live Rolling Accuracy</p>', unsafe_allow_html=True)
    ph_acc_chart = st.empty()

with chart_col2:
    st.markdown('<p class="section-header">🤝 Agent Trust Scores Over Time</p>', unsafe_allow_html=True)
    ph_trust_chart = st.empty()

st.divider()

# ══════════════════════════════════════════════════════════════
#  ROW 4 — Dataset progress pipeline + completed results
# ══════════════════════════════════════════════════════════════
pipe_col, results_col = st.columns([2, 3])

with pipe_col:
    st.markdown('<p class="section-header">🗂️ Dataset Pipeline</p>', unsafe_allow_html=True)
    ph_pipeline = st.empty()

with results_col:
    st.markdown('<p class="section-header">✅ Completed Dataset Results</p>', unsafe_allow_html=True)
    ph_results = st.empty()

st.divider()

# ══════════════════════════════════════════════════════════════
#  ROW 5 — Retrain queue status
# ══════════════════════════════════════════════════════════════
st.markdown('<p class="section-header">🔄 Active Learning — Misclassification Queue</p>', unsafe_allow_html=True)
ph_queue_detail = st.empty()

# ══════════════════════════════════════════════════════════════
#  LIVE LOOP
# ══════════════════════════════════════════════════════════════
while True:
    data = read_feed()

    if data is None:
        ph_decision.markdown(
            '<div class="complete-banner">⏳ Waiting for orchestrator to start…<br>'
            '<small>Run: <code>python orchestrator_master.py</code> in a second terminal</small></div>',
            unsafe_allow_html=True
        )
        time.sleep(1)
        continue

    ds   = data.get("dataset_name", "—")
    idx  = data.get("packets_processed", 0)
    tot  = data.get("total_packets", 5000)
    acc  = data.get("accuracy", 0.0)
    tr_r = data.get("trust_recon", 0.33)
    tr_m = data.get("trust_mqtt",  0.33)
    tr_v = data.get("trust_vol",   0.34)
    p_r  = data.get("p_recon",  0.0)
    p_m  = data.get("p_mqtt",   0.0)
    p_v  = data.get("p_vol",    0.0)
    fp   = data.get("latest_fused_prob", 0.0)
    is_a = data.get("is_attack", False)
    al   = data.get("actual_label", False)
    qsz  = data.get("retrain_queue_size", 0)
    stat = data.get("status", "running")

    # ── Track completed datasets ──────────────────────────────
    if stat == "complete" and ds not in st.session_state.completed:
        st.session_state.completed[ds] = f"{acc:.2f}%"

    # ── Append to history ──────────────────────────────────────
    st.session_state.acc_history.append({"Flow": idx, "Accuracy": acc})
    st.session_state.trust_history.append({
        "Flow": idx,
        "Recon": round(tr_r, 4),
        "MQTT":  round(tr_m, 4),
        "Volumetric": round(tr_v, 4),
    })

    # ─────────────────────────────────────────────────────────
    #  KPI METRICS
    # ─────────────────────────────────────────────────────────
    ph_dataset.metric("Dataset", DATASET_NICE.get(ds, ds))
    pct_done = min(idx / max(tot, 1), 1.0) * 100
    ph_progress.metric("Progress", f"{pct_done:.1f}%", f"{idx}/{tot} flows")
    ph_accuracy.metric("Live Accuracy", f"{acc:.2f}%")
    ph_tr_recon.metric("Recon Trust",  f"{tr_r:.4f}")
    ph_tr_mqtt.metric("MQTT Trust",    f"{tr_m:.4f}")
    ph_tr_vol.metric("Vol. Trust",     f"{tr_v:.4f}")
    ph_queue.metric("Retrain Queue",   qsz)

    # ─────────────────────────────────────────────────────────
    #  AGENT PROBABILITY BARS
    # ─────────────────────────────────────────────────────────
    ph_bar_recon.progress(min(p_r / 100, 1.0), text=f"🔍 Reconnaissance Agent:  {p_r:.1f}%")
    ph_bar_mqtt.progress( min(p_m / 100, 1.0), text=f"📡 MQTT Expert Agent:     {p_m:.1f}%")
    ph_bar_vol.progress(  min(p_v / 100, 1.0), text=f"📊 Volumetric Agent:      {p_v:.1f}%")
    fused_clamp = min(fp / 100, 1.0)
    ph_bar_fused.progress(fused_clamp,          text=f"⚡ Fused Threat Score:    {fp:.1f}%")

    # ─────────────────────────────────────────────────────────
    #  DECISION BANNER
    # ─────────────────────────────────────────────────────────
    if stat == "complete":
        ph_decision.markdown(
            f'<div class="complete-banner">✅ {DATASET_NICE.get(ds, ds)} complete<br>'
            f'Final accuracy: <strong>{acc:.2f}%</strong></div>',
            unsafe_allow_html=True
        )
    elif is_a:
        ph_decision.markdown(
            f'<div class="attack-banner">🚨 ATTACK DETECTED<br>'
            f'<small>Confidence: {fp:.1f}% &nbsp;|&nbsp; Ground truth: {"Attack ✓" if al else "Benign ✗"}</small></div>',
            unsafe_allow_html=True
        )
    else:
        ph_decision.markdown(
            f'<div class="benign-banner">✅ BENIGN FLOW<br>'
            f'<small>Threat score: {fp:.1f}% &nbsp;|&nbsp; Ground truth: {"Benign ✓" if not al else "Attack ✗"}</small></div>',
            unsafe_allow_html=True
        )

    ph_flow_info.caption(f"Flow #{idx} — Dataset: {DATASET_NICE.get(ds, ds)}")

    # ─────────────────────────────────────────────────────────
    #  LIVE ACCURACY CHART
    # ─────────────────────────────────────────────────────────
    if len(st.session_state.acc_history) > 1:
        df_acc = pd.DataFrame(st.session_state.acc_history)
        ph_acc_chart.line_chart(df_acc.set_index("Flow")["Accuracy"], height=200)

    # ─────────────────────────────────────────────────────────
    #  TRUST SCORES CHART
    # ─────────────────────────────────────────────────────────
    if len(st.session_state.trust_history) > 1:
        df_trust = pd.DataFrame(st.session_state.trust_history)
        ph_trust_chart.line_chart(df_trust.set_index("Flow")[["Recon","MQTT","Volumetric"]], height=200)

    # ─────────────────────────────────────────────────────────
    #  DATASET PIPELINE
    # ─────────────────────────────────────────────────────────
    pipeline_md = ""
    for dname in DATASET_ORDER:
        nice = DATASET_NICE[dname]
        if dname in st.session_state.completed:
            pipeline_md += f"✅ **{nice}** — {st.session_state.completed[dname]}\n\n"
        elif dname == ds and stat == "running":
            pipeline_md += f"⚙️ **{nice}** ← *running ({pct_done:.0f}%)*\n\n"
        else:
            pipeline_md += f"⏳ {nice}\n\n"
    ph_pipeline.markdown(pipeline_md)

    # ─────────────────────────────────────────────────────────
    #  COMPLETED RESULTS TABLE  (reads CSV if orchestrator wrote it)
    # ─────────────────────────────────────────────────────────
    df_res = read_results()
    if not df_res.empty:
        ph_results.dataframe(df_res, use_container_width=True, hide_index=True)
    elif st.session_state.completed:
        df_live = pd.DataFrame([
            {"Dataset": DATASET_NICE.get(k, k), "Accuracy": v}
            for k, v in st.session_state.completed.items()
        ])
        ph_results.dataframe(df_live, use_container_width=True, hide_index=True)
    else:
        ph_results.caption("Results will appear here as datasets complete.")

    # ─────────────────────────────────────────────────────────
    #  RETRAIN QUEUE DETAIL
    # ─────────────────────────────────────────────────────────
    if qsz == 0:
        ph_queue_detail.success("Queue empty — system operating at peak confidence.")
    elif qsz < 50:
        ph_queue_detail.info(f"🔎 {qsz} misclassified flows queued. Accumulating batch for retraining.")
    else:
        ph_queue_detail.warning(
            f"⚠️ {qsz} flows in retrain queue. "
            f"Trust adaptation is active — agents adjusting weights in real time."
        )

    time.sleep(0.5)
