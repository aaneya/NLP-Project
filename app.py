"""SentiBERT — document sentiment analysis with BERT and a TF-IDF baseline."""
import io
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from pypdf import PdfReader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import LABELS, METRICS_FILE, MODELS_DIR, clean_text

FALLBACK = "distilbert-base-uncased-finetuned-sst-2-english"
st.set_page_config(page_title="SentiBERT | Document intelligence", page_icon="🎭", layout="wide")

# Presentation layer only: the stylesheet contains no application logic.
_CSS = (Path(__file__).resolve().parent / "assets" / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


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


CHART_INK = "#292936"
CHART_GREY = "#777582"
CHART_ROSE = "#D2A3B9"
CHART_MAUVE = "#BD91AC"
CHART_LAVENDER = "#E8E1F1"
CHART_PINK = "#F1E6EC"
CHART_OFF = "#F8F7F8"
CHART_GRID = "rgba(41,41,54,0.075)"
CHART_AXIS = "rgba(41,41,54,0.16)"
CHART_FONT = "Inter, system-ui, sans-serif"
CHART_SERIF = "Instrument Serif, Georgia, serif"
METRIC_KEYS = ["accuracy", "precision", "recall", "f1"]
METRIC_LABELS = ["Accuracy", "Precision", "Recall", "F1"]


def _short_model(name: str) -> str:
    lowered = name.lower()
    if "tf-idf" in lowered:
        return "TF-IDF + LogReg"
    if "distilbert" in lowered or "bert" in lowered:
        return "Fine-tuned DistilBERT"
    return name


def _chart_base(fig, height=330, margin=None, showlegend=False):
    fig.update_layout(
        height=height,
        margin=margin or dict(l=8, r=18, t=18, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=CHART_FONT, size=10, color=CHART_GREY),
        hoverlabel=dict(bgcolor=CHART_INK, bordercolor=CHART_INK,
                        font=dict(family=CHART_FONT, size=11, color=CHART_OFF)),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=9, color=CHART_GREY)),
    )
    return fig


def _chart_axes(fig, x_grid=False):
    fig.update_xaxes(showgrid=x_grid, gridcolor=CHART_GRID, zeroline=False,
                     linecolor=CHART_AXIS, tickfont=dict(size=9, color=CHART_GREY),
                     title_font=dict(size=9, color=CHART_GREY))
    fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False,
                     linecolor=CHART_AXIS, tickfont=dict(size=9, color=CHART_GREY),
                     title_font=dict(size=9, color=CHART_GREY))
    return fig


def probability_chart(probabilities, aggregation="Mean"):
    df = pd.DataFrame({"Window": list(range(1, len(probabilities) + 1)),
                       "Positive probability": probabilities})
    aggregate = df["Positive probability"].mean() if aggregation == "Mean" else df["Positive probability"].median()
    aggregate_label = "window mean" if aggregation == "Mean" else "window median"
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=.5, fillcolor=CHART_ROSE, opacity=.055, line_width=0)
    fig.add_hrect(y0=.5, y1=1, fillcolor=CHART_LAVENDER, opacity=.5, line_width=0)
    fig.add_hline(y=.5, line_dash="dot", line_color=CHART_GREY, opacity=.55,
                  annotation_text="decision boundary · 50%", annotation_position="top left",
                  annotation_font=dict(size=9, color=CHART_GREY, family=CHART_FONT))
    fig.add_hline(y=aggregate, line_dash="dash", line_color=CHART_MAUVE, opacity=.85,
                  annotation_text=f"{aggregate_label} · {aggregate:.0%}", annotation_position="bottom right",
                  annotation_font=dict(size=9, color=CHART_MAUVE, family=CHART_FONT))
    fig.add_trace(go.Scatter(x=df["Window"], y=df["Positive probability"], mode="lines+markers",
        line=dict(color=CHART_MAUVE, width=2.4, shape="spline", smoothing=.55),
        marker=dict(size=7, color=CHART_ROSE, line=dict(color=CHART_OFF, width=2)),
        fill="tozeroy", fillcolor="rgba(210,163,185,.13)",
        hovertemplate="Window %{x}<br>P(positive) %{y:.1%}<extra></extra>"))
    _chart_base(fig, height=355, margin=dict(l=6, r=20, t=24, b=6))
    _chart_axes(fig, x_grid=False)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title_text="Document window", dtick=max(1, len(probabilities) // 12),
                     gridcolor=CHART_GRID, zeroline=False)
    fig.update_yaxes(title_text="P(positive)", range=[0, 1], tickformat=".0%",
                     gridcolor=CHART_GRID, zeroline=False)
    return fig


def probability_gauge(p):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=p * 100,
        number={"suffix": "%", "font": {"size": 42, "color": CHART_INK, "family": CHART_SERIF}},
        gauge={"axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": CHART_GREY,
                        "tickfont": {"size": 9, "color": CHART_GREY, "family": CHART_FONT},
                        "dtick": 25, "tickwidth": 1},
            "bar": {"color": CHART_MAUVE, "thickness": .22},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [{"range": [0, 50], "color": CHART_PINK},
                      {"range": [50, 100], "color": CHART_LAVENDER}],
            "threshold": {"line": {"color": CHART_INK, "width": 2}, "thickness": .72, "value": 50}}))
    _chart_base(fig, height=215, margin=dict(l=34, r=34, t=10, b=4))
    return fig


def window_distribution_chart(probabilities):
    fig = go.Figure(go.Histogram(x=list(probabilities), xbins=dict(start=0, end=1, size=.1),
        marker=dict(color=CHART_ROSE, line=dict(color=CHART_OFF, width=1)), opacity=.9,
        hovertemplate="P(positive) %{x:.0%}<br>%{y} window(s)<extra></extra>"))
    mean = sum(probabilities) / len(probabilities)
    fig.add_vline(x=.5, line_dash="dot", line_color=CHART_GREY, opacity=.5)
    fig.add_vline(x=mean, line_dash="dash", line_color=CHART_MAUVE, opacity=.9,
                  annotation_text=f"mean {mean:.0%}", annotation_position="top",
                  annotation_font=dict(size=9, color=CHART_MAUVE, family=CHART_FONT))
    _chart_base(fig, height=285, margin=dict(l=6, r=12, t=22, b=30))
    _chart_axes(fig)
    fig.update_layout(bargap=.06, hovermode="x")
    fig.update_xaxes(title_text="P(positive)", range=[0, 1], tickformat=".0%", nticks=5,
                     title_standoff=4)
    fig.update_yaxes(title_text=None, gridcolor=CHART_GRID, rangemode="tozero", nticks=3)
    return fig


def window_mix_chart(probabilities, neutral_band):
    counts = {"Negative": 0, "Neutral / mixed": 0, "Positive": 0}
    for p in probabilities:
        label, _ = verdict(p, neutral_band)
        counts[label] += 1
    color_map = {"Negative": CHART_ROSE, "Neutral / mixed": CHART_LAVENDER, "Positive": CHART_MAUVE}
    labels = [k for k, v in counts.items() if v > 0]
    values = [counts[k] for k in labels]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=.66, sort=False, direction="clockwise",
        marker=dict(colors=[color_map[k] for k in labels], line=dict(color=CHART_OFF, width=2)),
        textinfo="none",
        hovertemplate="%{label}<br>%{value} window(s) · %{percent}<extra></extra>"))
    fig.add_annotation(text=f"<b>{len(probabilities)}</b><br><span style='font-size:9px;letter-spacing:.14em'>WINDOWS</span>",
                       showarrow=False, font=dict(family=CHART_SERIF, size=28, color=CHART_INK))
    _chart_base(fig, height=285, margin=dict(l=8, r=8, t=10, b=38), showlegend=True)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-.04, x=0,
                                  font=dict(size=9, color=CHART_GREY)))
    return fig


def model_comparison_chart(p_bert, p_base):
    names = ["Fine-tuned BERT", "TF-IDF + LogReg"]
    values = [p_bert, p_base]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", width=.42,
        marker=dict(color=[CHART_MAUVE, CHART_ROSE]),
        text=[f"{v:.1%}" for v in values], textposition="outside", cliponaxis=False,
        textfont=dict(size=11, color=CHART_INK, family=CHART_FONT),
        hovertemplate="%{y}<br>P(positive) %{x:.1%}<extra></extra>"))
    fig.add_vline(x=.5, line_dash="dot", line_color=CHART_GREY, opacity=.55,
                  annotation_text="50%", annotation_position="top",
                  annotation_font=dict(size=9, color=CHART_GREY, family=CHART_FONT))
    _chart_base(fig, height=230, margin=dict(l=6, r=64, t=24, b=8))
    _chart_axes(fig, x_grid=True)
    fig.update_xaxes(title_text="P(positive)", range=[0, 1.18], tickformat=".0%", nticks=5)
    fig.update_yaxes(showgrid=False, automargin=True)
    return fig


def benchmark_radar(df):
    colors = [CHART_MAUVE, CHART_ROSE, CHART_INK, CHART_GREY]
    fills = ["rgba(189,145,172,.14)", "rgba(210,163,185,.14)",
             "rgba(41,41,54,.06)", "rgba(119,117,130,.06)"]
    fig = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        values = [float(row.get(key, 0) or 0) for key in METRIC_KEYS]
        fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=METRIC_LABELS + [METRIC_LABELS[0]],
            fill="toself", fillcolor=fills[i % len(fills)],
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=5, color=colors[i % len(colors)]),
            name=_short_model(str(row.get("Model", "Model"))),
            hovertemplate="%{theta}<br>%{r:.1%}<extra>%{fullData.name}</extra>"))
    _chart_base(fig, height=350, margin=dict(l=48, r=48, t=28, b=26), showlegend=True)
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%",
                            tickangle=45, tickvals=[0, .5, 1],
                            gridcolor=CHART_GRID, linecolor=CHART_AXIS,
                            tickfont=dict(size=7, color=CHART_GREY, family=CHART_FONT)),
            angularaxis=dict(gridcolor=CHART_GRID, linecolor=CHART_AXIS,
                             tickfont=dict(size=8.5, color=CHART_INK, family=CHART_FONT))),
        legend=dict(orientation="h", yanchor="top", y=-.09, x=0,
                    font=dict(size=8, color=CHART_GREY)))
    return fig


def benchmark_delta_chart(df):
    if len(df) < 2:
        return None
    first, second = df.iloc[0], df.iloc[1]
    deltas = [float(first.get(key, 0) or 0) - float(second.get(key, 0) or 0) for key in METRIC_KEYS]
    colors = [CHART_MAUVE if d >= 0 else CHART_ROSE for d in deltas]
    limit = max(abs(d) for d in deltas) * 1.35 or .02
    fig = go.Figure(go.Bar(x=deltas, y=METRIC_LABELS, orientation="h", width=.4,
        marker=dict(color=colors), text=[f"{d:+.1%}" for d in deltas],
        textposition="outside", cliponaxis=False,
        textfont=dict(size=10, color=CHART_INK, family=CHART_FONT),
        hovertemplate="%{y}<br>Δ %{x:+.1%}<extra></extra>"))
    fig.add_vline(x=0, line_color=CHART_AXIS, line_width=1)
    _chart_base(fig, height=330, margin=dict(l=6, r=44, t=18, b=8))
    _chart_axes(fig, x_grid=True)
    left = _short_model(str(first.get('Model', 'First'))).replace("Fine-tuned ", "")
    right = _short_model(str(second.get('Model', 'Second'))).replace(" + LogReg", "")
    fig.update_xaxes(title_text=f"{left} − {right}",
                     range=[-limit, limit], tickformat=".0%", nticks=5)
    fig.update_yaxes(showgrid=False, automargin=True)
    return fig


def benchmark_error_chart(data):
    rows = []
    for name, values in data.items():
        cm = values.get("confusion_matrix")
        if not cm or len(cm) < 2:
            continue
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        negative_total = tn + fp
        positive_total = fn + tp
        rows.append({"name": _short_model(name).replace("Fine-tuned ", ""),
                     "fp": fp / negative_total if negative_total else 0.0,
                     "fn": fn / positive_total if positive_total else 0.0})
    if not rows:
        return None
    names = [row["name"] for row in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[row["fp"] for row in rows], y=names, orientation="h", name="False positive",
        marker=dict(color=CHART_ROSE), text=[f"{row['fp']:.1%}" for row in rows],
        textposition="outside", cliponaxis=False,
        textfont=dict(size=9, color=CHART_INK, family=CHART_FONT),
        hovertemplate="%{y}<br>False positive rate %{x:.1%}<extra></extra>"))
    fig.add_trace(go.Bar(x=[row["fn"] for row in rows], y=names, orientation="h", name="False negative",
        marker=dict(color=CHART_MAUVE), text=[f"{row['fn']:.1%}" for row in rows],
        textposition="outside", cliponaxis=False,
        textfont=dict(size=9, color=CHART_INK, family=CHART_FONT),
        hovertemplate="%{y}<br>False negative rate %{x:.1%}<extra></extra>"))
    highest = max(max(row["fp"], row["fn"]) for row in rows) * 1.28 or .05
    _chart_base(fig, height=305, margin=dict(l=92, r=40, t=26, b=8), showlegend=True)
    _chart_axes(fig, x_grid=True)
    fig.update_layout(barmode="group", bargap=.34, bargroupgap=.12)
    fig.update_xaxes(title_text="Error rate", range=[0, highest], tickformat=".0%",
                     nticks=4, tickangle=0)
    fig.update_yaxes(showgrid=False, categoryorder="array",
                     categoryarray=list(reversed(names)))
    return fig


def benchmark_time_chart(data):
    items = [(name, values.get("train_seconds")) for name, values in data.items()]
    items = [(name, float(seconds)) for name, seconds in items
             if isinstance(seconds, (int, float))]
    if not items:
        return None
    names = [_short_model(name).replace("Fine-tuned ", "") for name, _ in items]
    times = [seconds for _, seconds in items]
    palette = [CHART_MAUVE, CHART_ROSE, CHART_LAVENDER, CHART_GREY]
    fig = go.Figure(go.Bar(x=times, y=names, orientation="h", width=.4,
        marker=dict(color=[palette[index % len(palette)] for index in range(len(names))]),
        text=[f"{seconds:,.1f}s" for seconds in times], textposition="outside", cliponaxis=False,
        textfont=dict(size=9, color=CHART_INK, family=CHART_FONT),
        hovertemplate="%{y}<br>%{x:,.1f} seconds<extra></extra>"))
    _chart_base(fig, height=235, margin=dict(l=88, r=56, t=14, b=8))
    _chart_axes(fig, x_grid=True)
    fig.update_xaxes(title_text="Training seconds", range=[0, max(times) * 1.35],
                     nticks=4, tickangle=0)
    fig.update_yaxes(showgrid=False, automargin=True, categoryorder="array",
                     categoryarray=list(reversed(names)))
    return fig


def benchmark_data_chart(data):
    keys = [("train_samples", "Training"), ("validation_samples", "Validation"), ("test_samples", "Test")]
    names = [_short_model(name).replace("Fine-tuned ", "") for name in data]
    series = {}
    for key, label in keys:
        values = [entry.get(key, 0) or 0 for entry in data.values()]
        if any(values):
            series[label] = values
    if not series:
        return None
    colors = {"Training": CHART_LAVENDER, "Validation": CHART_ROSE, "Test": CHART_MAUVE}
    fig = go.Figure()
    for label, values in series.items():
        fig.add_trace(go.Bar(x=values, y=names, orientation="h", name=label,
            marker=dict(color=colors[label]),
            hovertemplate="%{y}<br>" + label + " %{x:,.0f}<extra></extra>"))
    _chart_base(fig, height=235, margin=dict(l=88, r=16, t=14, b=8), showlegend=True)
    _chart_axes(fig, x_grid=True)
    fig.update_layout(barmode="stack", bargap=.34, legend=dict(
        orientation="h", yanchor="bottom", y=1.02, x=0, traceorder="normal",
        bgcolor="rgba(0,0,0,0)", font=dict(size=8, color=CHART_GREY)))
    fig.update_xaxes(title_text="Samples", tickformat=",.0f", nticks=4)
    fig.update_yaxes(showgrid=False, automargin=True, categoryorder="array",
                     categoryarray=list(reversed(names)))
    return fig


def confusion_heatmap(cm, model_name):
    values = [[int(v) for v in row] for row in cm]
    labels = ["Negative", "Positive"]
    text = [[f"{v:,}" for v in row] for row in values]
    peak = max(max(row) for row in values) or 1
    fig = go.Figure(go.Heatmap(
        z=values, x=labels, y=labels, zmin=0, zmax=peak, xgap=3, ygap=3,
        colorscale=[[0, CHART_OFF], [.42, CHART_LAVENDER],
                    [.72, CHART_ROSE], [1, CHART_MAUVE]],
        text=text, texttemplate="%{text}",
        textfont=dict(size=12, color=CHART_INK, family=CHART_FONT),
        hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z:,} samples<extra></extra>",
        showscale=False))
    _chart_base(fig, height=300, margin=dict(l=52, r=12, t=8, b=48))
    fig.update_xaxes(side="bottom", tickfont=dict(size=9, color=CHART_GREY),
                     title_text="Predicted", title_font=dict(size=8, color=CHART_GREY),
                     title_standoff=6, showgrid=False, zeroline=False, tickangle=0)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=9, color=CHART_GREY),
                     title_text="Actual", title_font=dict(size=8, color=CHART_GREY),
                     title_standoff=6, showgrid=False, zeroline=False)
    return fig


st.markdown("""
<div class="masthead">
  <div class="brand">
    <span class="brand-mark"></span>
    <span class="brand-name">SentiBERT</span>
    <span class="brand-suffix">NLP / Project 10</span>
  </div>
  <div class="masthead-meta">
    <span>Document intelligence</span>
    <span>Binary sentiment</span>
    <span class="masthead-status"><i></i><span class="status-label">Model ready</span></span>
  </div>
</div>
<div class="hero">
  <div class="hero__copy">
    <div class="hero__eyebrow">Fine-tuned BERT · windowed reading</div>
    <h1>Senti<em>BERT</em></h1>
    <p class="hero__lede">Read a document the way a careful editor would — one window at a time, with the tone mapped across the whole page.</p>
    <div class="hero__meta">
      <div class="hero__meta-item"><span class="hero__meta-label">Model</span><span class="hero__meta-value">DistilBERT</span></div>
      <div class="hero__meta-item"><span class="hero__meta-label">Baseline</span><span class="hero__meta-value">TF-IDF · LogReg</span></div>
      <div class="hero__meta-item"><span class="hero__meta-label">Corpus</span><span class="hero__meta-value">IMDb reviews</span></div>
    </div>
  </div>
  <div class="hero__art" aria-hidden="true">
    <span class="art-index">01 / 02</span>
    <div class="art-ring art-ring--one"></div>
    <div class="art-ring art-ring--two"></div>
    <div class="art-ring art-ring--three"></div>
    <div class="art-disc"></div>
    <div class="art-cross"></div>
    <div class="art-card">
      <div class="art-card__head"><span>Signal / window</span><i></i></div>
      <div class="art-card__read"><strong>P</strong><span>positive</span></div>
      <div class="art-card__bars">
        <i style="height:38%"></i><i style="height:52%"></i><i style="height:30%"></i><i style="height:64%"></i>
        <i style="height:46%"></i><i style="height:72%"></i><i style="height:58%"></i><i style="height:40%"></i>
        <i style="height:66%"></i><i style="height:50%"></i><i style="height:34%"></i><i style="height:62%"></i>
        <i style="height:44%"></i><i style="height:70%"></i><i style="height:55%"></i><i style="height:78%"></i>
      </div>
      <div class="art-card__foot"><span>tone</span><span>01 — 32</span></div>
    </div>
    <span class="art-tag art-tag--a">Windowed inference</span>
    <span class="art-tag art-tag--b">Neutral zone</span>
    <span class="art-tag art-tag--c">Read the drift</span>
  </div>
</div>
""", unsafe_allow_html=True)
tab_analyze, tab_compare = st.tabs(["Analyze document", "Model benchmark"])

with st.sidebar:
    st.markdown("""
    <div class="side-head">
      <span class="side-head__kicker">Instrument / 01</span>
      <h2>Analysis settings</h2>
      <p>Shape how each document is read. These controls affect analysis only — benchmark metrics are unchanged.</p>
    </div>
    """, unsafe_allow_html=True)
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
    st.markdown('<div class="side-foot"><span>Local inference</span><i></i><span>v1.0</span></div>', unsafe_allow_html=True)

with tab_analyze:
    _, _, is_local = load_bert()
    st.markdown("""
    <div class="section-head">
      <span class="section-head__index">01</span>
      <h2>Read the <em>room.</em></h2>
      <p class="section-head__note">Upload a document or paste a passage. The model reads it in overlapping windows, then reports where the tone settles.</p>
    </div>
    """, unsafe_allow_html=True)
    if is_local:
        st.success("Fine-tuned local model loaded from `models/bert`.")
    else:
        st.info(f"Using public model `{FALLBACK}`. Train a local model with `python train_bert.py` for project-specific results.")
    left, right = st.columns([1.1, .9], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("""
            <div class="field-head">
              <span class="field-head__label">01 / Compose</span>
              <span class="field-head__hint">TXT · PDF</span>
            </div>
            <p class="chart-note" style="margin:-0.35rem 0 0.85rem;">Upload a document, or paste a passage directly below.</p>
            """, unsafe_allow_html=True)
            up = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"], label_visibility="collapsed")
            default = read_upload(up) if up else ""
            text = st.text_area("Paste a review, report, or document", value=default, height=235,
                                placeholder="Paste text here, or upload a TXT/PDF above…")
            st.caption("PDFs with scanned pages need OCR before their text can be analyzed.")
            analyze = st.button("Analyze sentiment", type="primary", use_container_width=True)
    with right:
        st.markdown("""
        <div class="guide">
          <div class="guide__eyebrow">Reading guide / three steps</div>
          <h3 class="guide__title">What you’ll <em>see.</em></h3>
          <p class="guide__intro">A document-level score, a confidence estimate, and a window-by-window view that reveals how the tone shifts.</p>
          <div class="guide-step">
            <span class="guide-step__index">01</span>
            <div><span class="guide-step__title">Upload or paste</span><span class="guide-step__text">A TXT or PDF, or a passage typed directly into the field.</span></div>
          </div>
          <div class="guide-step">
            <span class="guide-step__index">02</span>
            <div><span class="guide-step__title">Choose the lens</span><span class="guide-step__text">Set the neutral zone and context window in the sidebar.</span></div>
          </div>
          <div class="guide-step">
            <span class="guide-step__index">03</span>
            <div><span class="guide-step__title">Read the movement</span><span class="guide-step__text">The trend view maps each overlapping window; documents over the cap are truncated at the end.</span></div>
          </div>
          <div class="guide__foot"><b>Note.</b> The model was trained on movie reviews; results on other topics may differ. Scanned PDFs need OCR.</div>
        </div>
        """, unsafe_allow_html=True)

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
            st.markdown("""
            <div class="section-head section-head--compact">
              <span class="section-head__index">02</span>
              <h2>The <em>reading.</em></h2>
              <p class="section-head__note">A document-level verdict, its confidence, and the window-by-window movement behind it.</p>
            </div>
            """, unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Overall sentiment", label)
            m2.metric("Confidence", f"{conf:.1%}")
            m3.metric("Positive probability", f"{p_bert:.1%}")
            m4.metric("Document size", f"{len(cleaned.split()):,} words")
            if len(windows) >= max_windows and len(windows) == max_windows:
                st.warning(f"Analysis reached the {max_windows}-window cap. Increase Maximum windows to cover more of this document.")
            if len(windows) > 1:
                with st.container(border=True):
                    st.markdown("""
                    <div class="panel-head">
                      <div class="panel-head__copy">
                        <span class="panel-head__eyebrow">Trend / per window</span>
                        <h3>Movement across the document</h3>
                      </div>
                      <p class="panel-head__note">Each point is one overlapping text window. Hover for its positive probability.</p>
                    </div>
                    <div class="panel-head__rule"></div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(probability_chart(windows, aggregation), use_container_width=True, config={"displaylogo":False})
                spread = max(windows) - min(windows)
                s1, s2 = st.columns(2, gap="large")
                with s1:
                    with st.container(border=True):
                        st.markdown(f"""
                        <div class="panel-head">
                          <div class="panel-head__copy">
                            <span class="panel-head__eyebrow">Distribution</span>
                            <h3>Where the windows sit</h3>
                          </div>
                          <p class="panel-head__note">{len(windows)} windows · {spread:.0%} spread</p>
                        </div>
                        <div class="panel-head__rule"></div>
                        """, unsafe_allow_html=True)
                        st.plotly_chart(window_distribution_chart(windows), use_container_width=True, config={"displaylogo":False})
                with s2:
                    with st.container(border=True):
                        st.markdown(f"""
                        <div class="panel-head">
                          <div class="panel-head__copy">
                            <span class="panel-head__eyebrow">Composition</span>
                            <h3>Window mix</h3>
                          </div>
                          <p class="panel-head__note">Neutral zone ±{neutral_band:.2f}</p>
                        </div>
                        <div class="panel-head__rule"></div>
                        """, unsafe_allow_html=True)
                        st.plotly_chart(window_mix_chart(windows, neutral_band), use_container_width=True, config={"displaylogo":False})
                if p_base is not None:
                    with st.container(border=True):
                        st.markdown(f"""
                        <div class="panel-head">
                          <div class="panel-head__copy">
                            <span class="panel-head__eyebrow">Cross-check</span>
                            <h3>Model comparison</h3>
                          </div>
                          <p class="panel-head__note">{abs(p_bert - p_base):.1%} apart</p>
                        </div>
                        <div class="panel-head__rule"></div>
                        """, unsafe_allow_html=True)
                        st.plotly_chart(model_comparison_chart(p_bert, p_base), use_container_width=True, config={"displaylogo":False})
            else:
                st.caption("This document fits in one window; add more text to see the trend chart.")
            with st.expander("Model details and comparison", expanded=p_base is not None):
                gauge_col, reading_col = st.columns([.64, .36], gap="large")
                with gauge_col:
                    st.markdown('<div class="panel-head__eyebrow">Confidence</div>', unsafe_allow_html=True)
                    st.plotly_chart(probability_gauge(p_bert), use_container_width=True, config={"displayModeBar":False})
                with reading_col:
                    st.markdown("#### Reading")
                    st.caption(f"BERT probability. Aggregation: {aggregation.lower()} of {len(windows)} windows.")
                    if p_base is None:
                        st.info("Train the TF-IDF baseline with `python train_baseline.py` to compare a second model here.")
                if p_base is not None:
                    bl, bc = verdict(p_base, neutral_band)
                    st.markdown("#### On this document")
                    st.dataframe(pd.DataFrame([
                        {"Model":"BERT", "Reading":f"{label} · {conf:.1%}", "P(pos.)":f"{p_bert:.1%}"},
                        {"Model":"TF-IDF", "Reading":f"{bl} · {bc:.1%}", "P(pos.)":f"{p_base:.1%}"},
                    ]), use_container_width=True, hide_index=True)
                    delta = abs(p_bert-p_base)
                    st.caption(f"TF-IDF = TF-IDF + Logistic Regression. The models differ by {delta:.1%} in positive probability.")

with tab_compare:
    st.markdown("""
    <div class="section-head">
      <span class="section-head__index">02</span>
      <h2>Measure the <em>signal.</em></h2>
      <p class="section-head__note">Held-out evaluation metrics saved by the training scripts. Every chart below reads directly from the saved results file.</p>
    </div>
    """, unsafe_allow_html=True)
    if METRICS_FILE.exists():
        data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        rows = [{"Model": n, **{k: v.get(k, 0) for k in ("accuracy", "precision", "recall", "f1")},
                 "Train samples": v.get("train_samples", "—"), "Train time (s)": v.get("train_seconds", "—")} for n,v in data.items()]
        df = pd.DataFrame(rows)
        metric_cols = st.columns(4)
        for col, metric in zip(metric_cols, ["accuracy", "precision", "recall", "f1"]):
            col.metric(metric.title(), f"{df[metric].max():.1%}", help="Best value among saved models")

        first_col, second_col = st.columns([1.08, .92], gap="large")
        with first_col:
            with st.container(border=True):
                st.markdown("""
                <div class="panel-head">
                  <div class="panel-head__copy">
                    <span class="panel-head__eyebrow">Performance / overview</span>
                    <h3>Performance profile</h3>
                  </div>
                  <p class="panel-head__note">Four held-out metrics, one shape per model.</p>
                </div>
                <div class="panel-head__rule"></div>
                """, unsafe_allow_html=True)
                st.plotly_chart(benchmark_radar(df), use_container_width=True, config={"displaylogo":False})

        with second_col:
            delta_chart = benchmark_delta_chart(df)
            with st.container(border=True):
                if delta_chart is not None:
                    first_name = _short_model(str(df.iloc[0].get("Model", "First")))
                    second_name = _short_model(str(df.iloc[1].get("Model", "Second")))
                    st.markdown(f"""
                    <div class="panel-head">
                      <div class="panel-head__copy">
                        <span class="panel-head__eyebrow">Difference / by metric</span>
                        <h3>Where they diverge</h3>
                      </div>
                      <p class="panel-head__note">{first_name} minus {second_name}.</p>
                    </div>
                    <div class="panel-head__rule"></div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(delta_chart, use_container_width=True, config={"displaylogo":False})
                else:
                    st.markdown("""
                    <div class="panel-head">
                      <div class="panel-head__copy">
                        <span class="panel-head__eyebrow">Difference / by metric</span>
                        <h3>Where they diverge</h3>
                      </div>
                    </div>
                    <div class="panel-head__rule"></div>
                    """, unsafe_allow_html=True)
                    st.caption("A second model is needed to compare metric deltas.")

        third_col, fourth_col = st.columns([.95, 1.05], gap="large")
        with third_col:
            error_chart = benchmark_error_chart(data)
            with st.container(border=True):
                st.markdown("""
                <div class="panel-head">
                  <div class="panel-head__copy">
                    <span class="panel-head__eyebrow">Errors / by class</span>
                    <h3>Error profile</h3>
                  </div>
                  <p class="panel-head__note">Class-conditional error rates from each confusion matrix.</p>
                </div>
                <div class="panel-head__rule"></div>
                """, unsafe_allow_html=True)
                if error_chart is not None:
                    st.plotly_chart(error_chart, use_container_width=True, config={"displaylogo":False})
                else:
                    st.caption("Confusion matrices are not available for these results.")

        with fourth_col:
            with st.container(border=True):
                st.markdown("""
                <div class="panel-head">
                  <div class="panel-head__copy">
                    <span class="panel-head__eyebrow">Cost / coverage</span>
                    <h3>Training and evaluation</h3>
                  </div>
                  <p class="panel-head__note">Training seconds and split sizes, as saved.</p>
                </div>
                <div class="panel-head__rule"></div>
                """, unsafe_allow_html=True)
                time_chart = benchmark_time_chart(data)
                if time_chart is not None:
                    st.plotly_chart(time_chart, use_container_width=True, config={"displaylogo":False})
                data_chart = benchmark_data_chart(data)
                if data_chart is not None:
                    st.plotly_chart(data_chart, use_container_width=True, config={"displaylogo":False})
                if time_chart is None and data_chart is None:
                    st.caption("Training time and split sizes are not available in the saved results.")

        cm_items = [(name, values) for name, values in data.items() if "confusion_matrix" in values]
        if cm_items:
            st.markdown("""
            <div class="section-head section-head--compact" style="margin-top:1.6rem;">
              <span class="section-head__index">03</span>
              <h2>Where the <em>errors</em> land.</h2>
              <p class="section-head__note">Actual classes run down the rows; predicted classes run across the columns.</p>
            </div>
            """, unsafe_allow_html=True)
            for start in range(0, len(cm_items), 2):
                cm_cols = st.columns(2, gap="large")
                for col, (name, values) in zip(cm_cols, cm_items[start:start + 2]):
                    with col:
                        with st.expander(f"{name} · confusion matrix"):
                            with st.container(border=True):
                                st.markdown(f"""
                                <div class="panel-head">
                                  <div class="panel-head__copy">
                                    <span class="panel-head__eyebrow">Confusion matrix</span>
                                    <h3>{_short_model(name)}</h3>
                                  </div>
                                </div>
                                <div class="panel-head__rule"></div>
                                """, unsafe_allow_html=True)
                                st.plotly_chart(confusion_heatmap(values["confusion_matrix"], name),
                                                use_container_width=True, config={"displaylogo":False})

        with st.container(border=True):
            st.markdown("""
            <div class="panel-head">
              <div class="panel-head__copy">
                <span class="panel-head__eyebrow">Results / all models</span>
                <h3>Saved benchmark</h3>
              </div>
              <p class="panel-head__note">Sort or inspect the raw held-out values.</p>
            </div>
            <div class="panel-head__rule"></div>
            """, unsafe_allow_html=True)
            df_view = df.copy()
            df_view["Model"] = df_view["Model"].map(_short_model)
            st.dataframe(df_view, use_container_width=True, hide_index=True, column_config={
                "Model":st.column_config.TextColumn("Model",width="medium"),
                "accuracy":st.column_config.NumberColumn("Accuracy",format="percent",width="small"),
                "precision":st.column_config.NumberColumn("Precision",format="percent",width="small"),
                "recall":st.column_config.NumberColumn("Recall",format="percent",width="small"),
                "f1":st.column_config.NumberColumn("F1",format="percent",width="small")})
            st.caption("Full model names · " + "; ".join(
                f"{_short_model(name)} = {name}" for name in data))
    else:
        st.info("No evaluation results found yet. Run `python train_baseline.py` and `python train_bert.py` to populate this view.")

st.markdown("""
<div class="page-foot">
  <span>SentiBERT · Document intelligence</span>
  <span>DistilBERT + TF-IDF baseline</span>
  <span>NLP / Project 10 · 2026</span>
</div>
""", unsafe_allow_html=True)
