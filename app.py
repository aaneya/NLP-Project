"""SentiBERT — document sentiment analysis with BERT and a TF-IDF baseline."""
import io
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from pypdf import PdfReader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import LABELS, METRICS_FILE, MODELS_DIR, clean_text

FALLBACK = "distilbert-base-uncased-finetuned-sst-2-english"
st.set_page_config(page_title="SentiBERT | Document intelligence", page_icon="🎭", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#e8edf5; --muted:#9aa8bc; --line:#263248; --panel:#111b2b; --bg:#0b1220; --cyan:#55d6c2; --blue:#4d8dff; }
.stApp { background:radial-gradient(ellipse at 85% 4%,rgba(49,119,228,.12),transparent 27%),radial-gradient(ellipse at 12% 0%,rgba(40,205,182,.12),transparent 30%),linear-gradient(180deg,#0b1220 0%,#080f1b 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif!important; letter-spacing:-.035em; }
h1 { font-size:2.65rem!important; }
.block-container { max-width:1260px; padding-top:2rem; padding-bottom:4rem; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#152238,#111b2b); border:1px solid var(--line); border-radius:16px; padding:18px 20px; }
[data-testid="stMetricValue"] { font-family:'Space Grotesk',sans-serif; }

/* Premium hero */
.hero { position:relative; display:flex; align-items:center; justify-content:space-between; gap:28px; min-height:230px; overflow:hidden; isolation:isolate; border:1px solid rgba(104,153,205,.25); border-radius:26px; padding:32px 38px; margin:8px 0 24px; background:linear-gradient(112deg,rgba(20,39,63,.98),rgba(14,28,47,.92) 60%,rgba(16,39,58,.9)); box-shadow:0 24px 70px rgba(0,0,0,.24),inset 0 1px rgba(255,255,255,.04); }
.hero::before { content:''; position:absolute; z-index:-1; width:390px; height:390px; right:8%; top:-250px; border-radius:50%; background:radial-gradient(circle,rgba(55,215,194,.17),rgba(77,141,255,.06) 48%,transparent 70%); filter:blur(4px); }
.hero-copy { position:relative; z-index:2; max-width:760px; }
.hero-kicker { display:inline-flex; align-items:center; gap:9px; color:#8ceadd; text-transform:uppercase; letter-spacing:.15em; font-size:.7rem; font-weight:700; padding:8px 12px; border:1px solid rgba(85,214,194,.22); border-radius:999px; background:rgba(35,185,171,.08); }
.kicker-dot { width:7px; height:7px; border-radius:50%; background:#55d6c2; box-shadow:0 0 12px rgba(85,214,194,.9); }
.hero h1 { margin:18px 0 8px; color:#f5f8ff; font-size:clamp(2.2rem,4vw,3.35rem)!important; line-height:1.05; }
.hero-title-accent { color:#55d6c2; background:linear-gradient(100deg,#70ecd5 5%,#62b8ff 95%); background-clip:text; -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero-sub { color:#aebbd0; max-width:660px; margin:0; font-size:1rem; line-height:1.7; }
.hero-pills { display:flex; flex-wrap:wrap; gap:9px; margin-top:19px; }
.hero-pills span { color:#cbd8e9; font-size:.76rem; padding:7px 11px; border:1px solid rgba(159,184,217,.15); border-radius:999px; background:rgba(8,17,31,.32); }
.hero-art { position:relative; flex:0 0 205px; width:205px; height:175px; margin-right:20px; }
.hero-core { position:absolute; z-index:2; display:grid; place-items:center; width:92px; height:92px; top:41px; left:54px; border:1px solid rgba(173,247,235,.55); border-radius:31px; background:linear-gradient(145deg,rgba(71,220,194,.96),rgba(59,133,232,.92)); box-shadow:0 0 40px rgba(53,205,195,.25),inset 0 1px rgba(255,255,255,.52); transform:rotate(-10deg); animation:core-float 6s ease-in-out infinite; }
.hero-core span { color:#effffb; font-size:2.2rem; text-shadow:0 3px 14px rgba(4,32,49,.35); transform:rotate(10deg); }
.hero-orbit { position:absolute; border:1px solid rgba(126,200,229,.22); border-radius:50%; transform:rotate(-24deg); }
.hero-orbit-one { width:176px; height:86px; top:46px; left:11px; }
.hero-orbit-two { width:156px; height:126px; top:27px; left:20px; transform:rotate(35deg); border-color:rgba(85,214,194,.2); }
.hero-bubble { position:absolute; z-index:3; display:grid; place-items:center; border:1px solid rgba(193,228,255,.34); border-radius:50%; background:linear-gradient(145deg,rgba(65,105,165,.65),rgba(35,58,91,.82)); color:#b9f5eb; box-shadow:0 10px 24px rgba(0,0,0,.24),inset 0 1px rgba(255,255,255,.2); animation:bubble-float 5s ease-in-out infinite; }
.hero-bubble-one { width:38px; height:38px; top:13px; right:18px; font-size:.75rem; }
.hero-bubble-two { width:27px; height:27px; bottom:20px; right:7px; color:#a9c8ff; animation-delay:-1.8s; }
.hero-bubble-three { width:19px; height:19px; bottom:26px; left:8px; animation-delay:-3s; }
@keyframes bubble-float { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-9px); } }
@keyframes core-float { 0%,100% { transform:translateY(0) rotate(-10deg); } 50% { transform:translateY(-5px) rotate(-6deg); } }

/* Rounded, active-state navigation */
div[data-testid="stTabs"] [data-baseweb="tab-list"] { display:flex; width:max-content; max-width:100%; gap:7px; padding:6px; border:1px solid rgba(123,153,192,.16); border-radius:999px; background:rgba(12,22,37,.82); }
div[data-testid="stTabs"] button[data-baseweb="tab"] { min-height:42px; padding:0 18px; border:1px solid transparent; border-radius:999px; color:#aab9cd; font-weight:600; transition:background .22s ease,color .22s ease,box-shadow .22s ease,transform .22s ease; }
div[data-testid="stTabs"] button[data-baseweb="tab"]:hover { color:#e9f5ff; background:rgba(75,127,180,.14); }
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] { color:#f4ffff; border-color:rgba(85,214,194,.26); background:linear-gradient(110deg,rgba(42,181,166,.24),rgba(67,119,205,.24)); box-shadow:0 5px 18px rgba(38,174,176,.12); }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display:none; }

/* Document dropzone and text entry */
.upload-intro { display:flex; align-items:center; gap:13px; margin:2px 0 11px; }
.upload-icon { display:grid; place-items:center; width:42px; height:42px; flex:none; border:1px solid rgba(85,214,194,.25); border-radius:14px; color:#89f0df; background:linear-gradient(145deg,rgba(46,207,184,.17),rgba(73,139,241,.16)); box-shadow:0 8px 22px rgba(11,36,58,.3); font-size:1.35rem; }
.upload-copy strong { display:block; color:#e8f2ff; font-size:.95rem; }
.upload-copy span { color:#92a4ba; font-size:.78rem; }
div[data-testid="stFileUploader"] { padding:5px; border:1px dashed rgba(91,177,205,.34); border-radius:20px; background:linear-gradient(145deg,rgba(20,39,60,.72),rgba(14,27,44,.72)); transition:border-color .22s ease,box-shadow .22s ease,transform .22s ease; }
div[data-testid="stFileUploader"]:hover { border-color:rgba(85,214,194,.72); box-shadow:0 12px 30px rgba(20,162,174,.1); transform:translateY(-1px); }
div[data-testid="stFileUploader"] section { border:0!important; border-radius:15px; background:transparent!important; }
div[data-testid="stFileUploader"] button { border-radius:999px!important; border:1px solid rgba(119,167,205,.25)!important; background:rgba(40,76,112,.42)!important; color:#d9ecfa!important; transition:all .2s ease!important; }
div[data-testid="stFileUploader"] button:hover { border-color:rgba(85,214,194,.56)!important; background:rgba(42,126,139,.3)!important; transform:translateY(-1px); }
div[data-testid="stTextArea"] [data-baseweb="textarea"] { overflow:hidden; border:1px solid rgba(111,143,180,.26); border-radius:18px; background:linear-gradient(145deg,rgba(20,32,49,.94),rgba(16,27,43,.96)); transition:border-color .2s ease,box-shadow .2s ease,transform .2s ease; }
div[data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within { border-color:rgba(85,214,194,.75); box-shadow:0 0 0 3px rgba(57,197,185,.12),0 12px 30px rgba(7,20,34,.24); transform:translateY(-1px); }
div[data-testid="stTextArea"] textarea { color:#edf5ff; background:transparent; }
div[data-testid="stTextArea"] textarea::placeholder { color:#72849c; }
div.stButton > button[kind="primary"] { min-height:50px; border:1px solid rgba(151,255,236,.3); border-radius:999px; background:linear-gradient(105deg,#27d6b7 0%,#269fda 52%,#477ff3 100%); color:#051522; font-weight:700; letter-spacing:.01em; padding:.78rem 1.35rem; box-shadow:0 10px 28px rgba(27,174,192,.24),inset 0 1px rgba(255,255,255,.38); transition:transform .2s ease,box-shadow .2s ease,filter .2s ease; }
div.stButton > button[kind="primary"]:hover { transform:translateY(-2px); filter:saturate(1.12) brightness(1.06); box-shadow:0 14px 34px rgba(36,186,205,.34),0 0 0 4px rgba(67,188,205,.08),inset 0 1px rgba(255,255,255,.42); }
div.stButton > button[kind="primary"]:active { transform:translateY(0); }
div.stButton > button:focus-visible { outline:2px solid #8deee0; outline-offset:3px; }
div.stButton > button:not([kind="primary"]) { border-radius:999px; transition:transform .2s ease,background .2s ease; }
div.stButton > button:not([kind="primary"]):hover { transform:translateY(-1px); }
.section-note { color:#9aa8bc; font-size:.92rem; }
div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:14px; }
small { color:var(--muted); }
@media (max-width:760px) { .hero { min-height:unset; padding:24px 22px; } .hero-art { flex:0 0 120px; width:120px; height:130px; margin:0 -8px 0 0; transform:scale(.78); transform-origin:center right; } .hero-core { left:24px; } .hero-orbit-one { left:-16px; } .hero-orbit-two { left:-9px; } .hero h1 { font-size:2.2rem!important; } .hero-sub { font-size:.9rem; } div[data-testid="stTabs"] [data-baseweb="tab-list"] { width:100%; overflow-x:auto; } div[data-testid="stTabs"] button[data-baseweb="tab"] { padding:0 13px; white-space:nowrap; } }
@media (max-width:520px) { .hero { padding:22px 18px; } .hero-art { display:none; } .hero h1 { font-size:2rem!important; } .hero-pills { gap:6px; } .hero-pills span { font-size:.7rem; } }
@media (prefers-reduced-motion:reduce) { *,*::before,*::after { animation-duration:.01ms!important; animation-iteration-count:1!important; scroll-behavior:auto!important; transition-duration:.01ms!important; } }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading sentiment model…")
def load_bert():
    local = MODELS_DIR / "bert"
    path, is_local = (str(local), True) if local.exists() else (FALLBACK, False)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    return tok, model, is_local


@st.cache_resource
def load_baseline():
    p = MODELS_DIR / "baseline.joblib"
    return joblib.load(p) if p.exists() else None


def bert_predict(text: str, max_len: int, overlap: int, max_windows: int, aggregation: str):
    tok, model, _ = load_bert()
    enc = tok(text, truncation=True, max_length=max_len, stride=overlap,
              return_overflowing_tokens=True, padding=True, return_tensors="pt")
    enc.pop("overflow_to_sample_mapping", None)
    enc = {k: v[:max_windows] for k, v in enc.items()}
    with torch.inference_mode():
        probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().tolist()
    overall = sum(probs) / len(probs) if aggregation == "Mean" else sorted(probs)[len(probs) // 2]
    return overall, probs


def verdict(p_pos: float, neutral_band: float):
    if abs(p_pos - 0.5) <= neutral_band:
        return "Neutral / mixed", 1 - abs(p_pos - 0.5) * 2
    return ("Positive", p_pos) if p_pos > 0.5 else ("Negative", 1 - p_pos)


def read_upload(f) -> str:
    if f.name.lower().endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(f.getvalue())).pages)
    return f.getvalue().decode("utf-8", errors="ignore")


def probability_chart(probabilities):
    df = pd.DataFrame({"Window": list(range(1, len(probabilities) + 1)),
                       "Positive probability": probabilities})
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=.5, fillcolor="#ff7185", opacity=.045, line_width=0)
    fig.add_hrect(y0=.5, y1=1, fillcolor="#55d6c2", opacity=.045, line_width=0)
    fig.add_hline(y=.5, line_dash="dot", line_color="#8492a8", opacity=.7,
                  annotation_text="decision boundary", annotation_position="top left")
    fig.add_trace(go.Scatter(x=df["Window"], y=df["Positive probability"], mode="lines+markers",
        line=dict(color="#55d6c2", width=3), marker=dict(size=8, color="#55d6c2",
        line=dict(color="#0b1220", width=2)), hovertemplate="Window %{x}<br>P(positive): %{y:.1%}<extra></extra>"))
    fig.update_layout(height=330, margin=dict(l=10,r=18,t=28,b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1", family="DM Sans"),
        xaxis_title="Document window", yaxis_title="P(positive)", yaxis=dict(range=[0,1],tickformat=".0%",gridcolor="#263248"),
        xaxis=dict(dtick=max(1, len(probabilities)//12), gridcolor="#1c293d", zeroline=False),
        showlegend=False, hovermode="x")
    return fig


def probability_gauge(p):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=p*100,
        number={"suffix":"%", "font":{"size":34,"color":"#e8edf5"}},
        gauge={"axis":{"range":[0,100],"ticksuffix":"%","tickcolor":"#9aa8bc"},
            "bar":{"color":"#55d6c2","thickness":.28},"bgcolor":"#17243a","borderwidth":0,
            "steps":[{"range":[0,50],"color":"#3a2637"},{"range":[50,100],"color":"#183a3c"}],
            "threshold":{"line":{"color":"#f0c36a","width":3},"thickness":.8,"value":50}}))
    fig.update_layout(height=205, margin=dict(l=18,r=18,t=16,b=4), paper_bgcolor="rgba(0,0,0,0)",
                      font={"color":"#cbd5e1","family":"DM Sans"})
    return fig


st.markdown("""
<div class="hero">
  <div class="hero-copy">
    <div class="hero-kicker"><span class="kicker-dot"></span>Document intelligence · NLP</div>
    <h1>🎭 <span class="hero-title-accent">SentiBERT</span></h1>
    <p class="hero-sub">Explore sentiment at a glance, then inspect how it shifts across your document.</p>
    <div class="hero-pills"><span>Fine-tuned BERT</span><span>Document analysis</span><span>Window-by-window insights</span></div>
  </div>
  <div class="hero-art" aria-hidden="true">
    <div class="hero-orbit hero-orbit-one"></div>
    <div class="hero-orbit hero-orbit-two"></div>
    <div class="hero-core"><span>✦</span></div>
    <div class="hero-bubble hero-bubble-one">AI</div>
    <div class="hero-bubble hero-bubble-two">+</div>
    <div class="hero-bubble hero-bubble-three"></div>
  </div>
</div>
""", unsafe_allow_html=True)
tab_analyze, tab_compare = st.tabs(["✦  Analyze document", "◫  Model benchmark"])

with st.sidebar:
    st.markdown("### Analysis settings")
    neutral_band = st.slider("Neutral zone", 0.0, 0.30, 0.10, 0.01,
        help="Classify scores this close to 50% as neutral/mixed. The underlying model remains binary.")
    max_len = st.select_slider("Tokens per window", options=[128, 192, 256, 384, 512], value=256,
        help="More tokens give wider context but use more memory. Must fit the loaded model's position limit.")
    overlap = st.slider("Window overlap (tokens)", 0, min(128, max_len//2), min(32, max_len//8), 8,
        help="Context repeated between neighboring windows.")
    max_windows = st.slider("Maximum windows", 1, 64, 32,
        help="Longer documents are capped at this many windows to control runtime.")
    aggregation = st.radio("Combine windows", ["Mean", "Median"], horizontal=True,
        help="Mean reflects the overall average; median reduces the effect of unusually strong passages.")
    st.caption("Settings affect document analysis only. Training and benchmark metrics are unchanged.")

with tab_analyze:
    _, _, is_local = load_bert()
    if is_local:
        st.success("Fine-tuned local model loaded from `models/bert`.")
    else:
        st.info(f"Using public model `{FALLBACK}`. Train a local model with `python train_bert.py` for project-specific results.")
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.markdown("""
        <div class="upload-intro">
          <div class="upload-icon">↥</div>
          <div class="upload-copy"><strong>Add your document</strong><span>Upload a TXT/PDF or paste text below</span></div>
        </div>
        """, unsafe_allow_html=True)
        up = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"], label_visibility="collapsed")
        default = read_upload(up) if up else ""
        text = st.text_area("Paste a review, report, or document", value=default, height=235,
                            placeholder="Paste text here, or upload a TXT/PDF above…")
        st.caption("PDFs with scanned pages need OCR before their text can be analyzed.")
        analyze = st.button("Analyze sentiment", type="primary", use_container_width=True)
    with right:
        st.markdown("### What you’ll see")
        st.markdown('<p class="section-note">A document level score, a confidence estimate, and a window-by-window view that helps reveal changes in tone.</p>', unsafe_allow_html=True)
        st.markdown("**Quick guidance**")
        st.markdown("- Upload a TXT/PDF or paste text.\n- Adjust the neutral zone and context window in the sidebar.\n- Documents over the window limit are truncated at the end.")
        st.caption("The model was trained on movie reviews; results on other topics may differ.")

    if analyze:
        cleaned = clean_text(text)
        if len(cleaned.split()) < 3:
            st.error("Please provide at least three words to analyze.")
        else:
            with st.spinner("Reading the document…"):
                p_bert, windows = bert_predict(cleaned, max_len, overlap, max_windows, aggregation)
                base = load_baseline()
                p_base = base.predict_proba([cleaned])[0][1] if base else None
            label, conf = verdict(p_bert, neutral_band)
            st.divider()
            st.markdown("### Analysis overview")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Overall sentiment", label)
            m2.metric("Confidence", f"{conf:.1%}")
            m3.metric("Positive probability", f"{p_bert:.1%}")
            m4.metric("Document size", f"{len(cleaned.split()):,} words")
            if len(windows) >= max_windows and len(windows) == max_windows:
                st.warning(f"Analysis reached the {max_windows}-window cap. Increase Maximum windows to cover more of this document.")
            if len(windows) > 1:
                st.markdown("### Sentiment through the document")
                st.markdown('<p class="section-note">Each point represents an overlapping text window. Hover for its positive probability.</p>', unsafe_allow_html=True)
                st.plotly_chart(probability_chart(windows), use_container_width=True, config={"displaylogo":False})
            else:
                st.caption("This document fits in one window; add more text to see the trend chart.")
            with st.expander("Model details and comparison", expanded=p_base is not None):
                detail_cols = st.columns([1,1])
                with detail_cols[0]:
                    st.plotly_chart(probability_gauge(p_bert), use_container_width=True, config={"displayModeBar":False})
                    st.caption(f"BERT probability. Aggregation: {aggregation.lower()} of {len(windows)} windows.")
                with detail_cols[1]:
                    if p_base is not None:
                        bl, bc = verdict(p_base, neutral_band)
                        st.markdown("#### On this document")
                        st.dataframe(pd.DataFrame([
                            {"Model":"BERT", "Sentiment":label, "Confidence":f"{conf:.1%}", "P(positive)":f"{p_bert:.1%}"},
                            {"Model":"TF-IDF + Logistic Regression", "Sentiment":bl, "Confidence":f"{bc:.1%}", "P(positive)":f"{p_base:.1%}"},
                        ]), use_container_width=True, hide_index=True)
                        delta = abs(p_bert-p_base)
                        st.caption(f"The models differ by {delta:.1%} in positive probability.")
                    else:
                        st.info("Train the TF-IDF baseline with `python train_baseline.py` to compare a second model here.")

with tab_compare:
    st.markdown("### Benchmark results")
    st.markdown('<p class="section-note">Held-out evaluation metrics saved by the training scripts.</p>', unsafe_allow_html=True)
    if METRICS_FILE.exists():
        data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        rows = [{"Model": n, **{k: v.get(k, 0) for k in ("accuracy", "precision", "recall", "f1")},
                 "Train samples": v.get("train_samples", "—"), "Train time (s)": v.get("train_seconds", "—")} for n,v in data.items()]
        df = pd.DataFrame(rows)
        metric_cols = st.columns(4)
        for col, metric in zip(metric_cols, ["accuracy", "precision", "recall", "f1"]):
            col.metric(metric.title(), f"{df[metric].max():.1%}", help="Best value among saved models")
        chart_df = df.melt(id_vars="Model", value_vars=["accuracy","precision","recall","f1"], var_name="Metric", value_name="Score")
        fig = px.bar(chart_df, x="Metric", y="Score", color="Model", barmode="group", text=chart_df["Score"].map(lambda x:f"{x:.1%}"),
                     color_discrete_sequence=["#55d6c2","#78a9ff","#f0c36a"])
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=390, margin=dict(l=10,r=20,t=20,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1",family="DM Sans"), yaxis=dict(range=[0,1.1],tickformat=".0%",gridcolor="#263248"),
            xaxis=dict(gridcolor="#1c293d"), legend_title_text="", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})
        st.dataframe(df, use_container_width=True, hide_index=True, column_config={
            "accuracy":st.column_config.NumberColumn("Accuracy",format="%.1%"),
            "precision":st.column_config.NumberColumn("Precision",format="%.1%"),
            "recall":st.column_config.NumberColumn("Recall",format="%.1%"),
            "f1":st.column_config.NumberColumn("F1",format="%.1%")})
        for n,v in data.items():
            if "confusion_matrix" in v:
                with st.expander(f"{n} · confusion matrix"):
                    cm = v["confusion_matrix"]
                    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in LABELS], columns=[f"Predicted {l}" for l in LABELS])
                    heat = px.imshow(cm_df, text_auto=True, color_continuous_scale=[[0,"#17243a"],[1,"#55d6c2"]], aspect="auto", labels=dict(color="Samples"))
                    heat.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"))
                    st.plotly_chart(heat, use_container_width=True, config={"displaylogo":False})
    else:
        st.info("No evaluation results found yet. Run `python train_baseline.py` and `python train_bert.py` to populate this view.")
