import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from pathlib import Path
import pickle
import joblib

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wine EDA Dashboard",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}

/* Dark wine-themed background */
.stApp {
    background: linear-gradient(135deg, #1a0a0e 0%, #2d1017 40%, #1e0f1a 100%);
    color: #f0e6d3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2d1017 0%, #1a0a0e 100%);
    border-right: 1px solid #5c2130;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(92, 33, 48, 0.3);
    border: 1px solid #7a2d3e;
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(4px);
}
div[data-testid="metric-container"] label {
    color: #c9a882 !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #f0e6d3 !important;
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
}

/* Section headers */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: #c9a882;
    border-bottom: 1px solid #5c2130;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    color: #c9a882 !important;
    letter-spacing: 0.05em;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #f0e6d3 !important;
    border-bottom: 2px solid #9b3a52 !important;
}

/* Select boxes & sliders */
.stSelectbox label, .stMultiSelect label, .stSlider label {
    color: #c9a882 !important;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #5c2130 !important;
    border-radius: 8px;
}

/* Main title */
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: #f0e6d3;
    letter-spacing: -0.02em;
}
.main-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    color: #c9a882;
    font-size: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: -8px;
}

/* Insight boxes */
.insight-box {
    background: rgba(155, 58, 82, 0.15);
    border-left: 3px solid #9b3a52;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #e8d5be;
}

/* Prediction result boxes */
.result-good {
    background: rgba(39, 174, 96, 0.2);
    border: 2px solid #27ae60;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
}
.result-bad {
    background: rgba(192, 57, 43, 0.2);
    border: 2px solid #c0392b;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
}
.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    margin: 8px 0 4px 0;
}
.result-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    color: #c9a882;
}

/* Number input labels */
.stNumberInput label {
    color: #c9a882 !important;
    font-size: 0.8rem;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# ── Color palette ────────────────────────────────────────────────────────────
COLORS = {
    "red":      "#c0392b",
    "white":    "#e8c97c",
    "accent":   "#9b3a52",
    "gold":     "#c9a882",
    "bg":       "#1a0a0e",
    "text":     "#f0e6d3",
}
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,10,20,0.5)",
        font=dict(color="#f0e6d3", family="DM Sans"),
        colorway=["#c0392b", "#e8c97c", "#9b3a52", "#6c8ebf", "#82b366"],
        xaxis=dict(gridcolor="#3a1520", linecolor="#5c2130"),
        yaxis=dict(gridcolor="#3a1520", linecolor="#5c2130"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#f0e6d3")),
    )
)

# ── Feature names (from scaler.pkl) ─────────────────────────────────────────
FEATURE_NAMES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
    "density", "pH", "sulphates", "alcohol"
]

# ── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    primary_path = Path(r"C:\Users\VEDANT\Machine Learning\Project - Alcohol\wine_cleaned.csv")
    local_path = Path(__file__).resolve().parent / "wine_cleaned.csv"
    if primary_path.exists():
        data_path = primary_path
    elif local_path.exists():
        data_path = local_path
    else:
        st.error(
            "Could not find the dataset file.\n"
            f"Expected: {primary_path}\n"
            f"Or: {local_path}"
        )
        st.stop()

    df = pd.read_csv(r"C:\Users\VEDANT\Machine Learning\Project - Alcohol\wine_cleaned.csv")
    df.columns = df.columns.str.strip()
    df["quality"] = df["quality"].astype(int)
    df["good"] = df["good"].astype(int)
    return df

df = load_data()
num_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in ["quality", "good"]]

# ── Load ML Model ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Try to load model.pkl from common locations."""
    candidates = [
        Path(__file__).resolve().parent / "model.pkl",
        Path(r"C:\Users\VEDANT\Machine Learning\Project - Alcohol\model.pkl"),
        Path("model.pkl"),
    ]
    for p in candidates:
        if p.exists():
            return joblib.load(str(p))
    return None

@st.cache_resource
def build_scaler_from_data(data):
    """Build a StandardScaler fitted on the loaded dataset as fallback."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(data[FEATURE_NAMES])
    return scaler

model = load_model()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍷 Wine Explorer")
    st.markdown("---")

    color_filter = st.multiselect(
        "Wine Color", options=["red", "white"], default=["red", "white"]
    )
    quality_filter = st.slider(
        "Quality Range", min_value=int(df.quality.min()),
        max_value=int(df.quality.max()),
        value=(int(df.quality.min()), int(df.quality.max()))
    )

    st.markdown("---")
    st.markdown(
        "<div style='color:#c9a882;font-size:0.75rem;line-height:1.6'>"
        "Dataset: <b style='color:#f0e6d3'>4,418 wines</b><br>"
        "Features: <b style='color:#f0e6d3'>11 physicochemical</b><br>"
        "Colors: <b style='color:#f0e6d3'>Red & White</b><br>"
        "Quality: <b style='color:#f0e6d3'>Scale 4–7</b>"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Filter ───────────────────────────────────────────────────────────────────
dff = df[
    df["color"].isin(color_filter) &
    df["quality"].between(*quality_filter)
].copy()

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Wine Quality EDA</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Exploratory Data Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown("")

# ── KPI Row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Wines", f"{len(dff):,}")
c2.metric("Red Wines",   f"{(dff.color=='red').sum():,}")
c3.metric("White Wines", f"{(dff.color=='white').sum():,}")
c4.metric("Avg Quality", f"{dff.quality.mean():.2f}")
c5.metric("% Good Wine", f"{dff.good.mean()*100:.1f}%")

st.markdown("")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🔬 Distributions",
    "🔗 Correlations",
    "🍷 Quality Analysis",
    "🤖 Predict Quality",
    "📋 Raw Data",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1])

    # Quality distribution
    with col_left:
        st.markdown('<div class="section-title">Quality Distribution</div>', unsafe_allow_html=True)
        qd = dff.groupby(["quality", "color"]).size().reset_index(name="count")
        fig = px.bar(
            qd, x="quality", y="count", color="color",
            barmode="group",
            color_discrete_map={"red": COLORS["red"], "white": COLORS["white"]},
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Color split pie
    with col_right:
        st.markdown('<div class="section-title">Color Split</div>', unsafe_allow_html=True)
        pie_data = dff["color"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=pie_data.index, values=pie_data.values,
            hole=0.55,
            marker=dict(colors=[COLORS["red"], COLORS["white"]],
                        line=dict(color="#1a0a0e", width=2)),
            textfont=dict(color="#f0e6d3"),
        ))
        fig2.update_layout(**PLOTLY_TEMPLATE["layout"], height=320,
                           margin=dict(l=0, r=0, t=10, b=0),
                           showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    # Summary statistics
    st.markdown('<div class="section-title">Summary Statistics</div>', unsafe_allow_html=True)
    stats_df = dff[num_cols + ["quality"]].describe().T.round(3)
    st.dataframe(
        stats_df.style.background_gradient(
            cmap="RdYlGn", subset=["mean", "50%"]
        ).format("{:.3f}"),
        use_container_width=True, height=380,
    )

    # Key insights
    st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown(f'<div class="insight-box">🍷 <b>Alcohol</b> ranges from {dff.alcohol.min():.1f}% to {dff.alcohol.max():.1f}%, with a mean of {dff.alcohol.mean():.1f}%.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box">🧪 <b>Volatile acidity</b> is notably higher in red wines, which can affect taste.</div>', unsafe_allow_html=True)
    with ic2:
        good_pct = dff.good.mean() * 100
        st.markdown(f'<div class="insight-box">⭐ Only <b>{good_pct:.1f}%</b> of wines are classified as "good" (quality ≥ 7).</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight-box">📈 Most wines cluster around quality scores <b>5 and 6</b>, creating a near-normal distribution.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    col_sel, col_type = st.columns([2, 1])
    with col_sel:
        feat = st.selectbox("Select Feature", num_cols, index=num_cols.index("alcohol") if "alcohol" in num_cols else 0)
    with col_type:
        split_by = st.selectbox("Split By", ["None", "color", "good"])

    col_a, col_b = st.columns(2)

    # Histogram
    with col_a:
        st.markdown('<div class="section-title">Histogram</div>', unsafe_allow_html=True)
        if split_by == "None":
            fig = px.histogram(dff, x=feat, nbins=40,
                               color_discrete_sequence=[COLORS["accent"]])
        else:
            cmap = {"red": COLORS["red"], "white": COLORS["white"],
                    0: "#c0392b", 1: "#27ae60"}
            fig = px.histogram(dff, x=feat, color=split_by, nbins=40,
                               barmode="overlay", opacity=0.75,
                               color_discrete_map=cmap)
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Box plot
    with col_b:
        st.markdown('<div class="section-title">Box Plot</div>', unsafe_allow_html=True)
        if split_by == "None":
            fig = px.box(dff, y=feat, color_discrete_sequence=[COLORS["accent"]])
        else:
            fig = px.box(dff, x=split_by, y=feat,
                         color=split_by,
                         color_discrete_map={"red": COLORS["red"], "white": COLORS["white"],
                                             0: "#c0392b", 1: "#27ae60"})
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=320,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # All features grid
    st.markdown('<div class="section-title">All Feature Distributions</div>', unsafe_allow_html=True)
    grid_cols = 4
    rows = [num_cols[i:i+grid_cols] for i in range(0, len(num_cols), grid_cols)]
    for row in rows:
        cols = st.columns(len(row))
        for col_w, fc in zip(cols, row):
            with col_w:
                fig = px.histogram(dff, x=fc, nbins=30,
                                   color_discrete_sequence=[COLORS["accent"]],
                                   title=fc)
                fig.update_layout(**PLOTLY_TEMPLATE["layout"],
                                  height=200, showlegend=False,
                                  margin=dict(l=0, r=0, t=30, b=0),
                                  title_font=dict(size=12, color=COLORS["gold"]))
                st.plotly_chart(fig, use_container_width=True)

    # Normality test
    st.markdown('<div class="section-title">Normality Test (Shapiro-Wilk, sample n=500)</div>', unsafe_allow_html=True)
    norm_results = []
    sample = dff[num_cols].dropna().sample(min(500, len(dff)), random_state=42)
    for c in num_cols:
        stat, p = stats.shapiro(sample[c])
        norm_results.append({"Feature": c, "W-statistic": round(stat, 4),
                              "p-value": round(p, 4),
                              "Normal?": "✅ Yes" if p > 0.05 else "❌ No"})
    st.dataframe(pd.DataFrame(norm_results).set_index("Feature"),
                 use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — CORRELATIONS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr_cols = num_cols + ["quality", "good"]
    corr_data = dff[corr_cols].select_dtypes(include=np.number).corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale=[
            [0,   "#2d1017"],
            [0.5, "#1e0f1a"],
            [1,   "#c0392b"],
        ],
        zmin=-1, zmax=1,
        text=corr_data.values,
        texttemplate="%{text:.2f}",
        textfont=dict(size=10, color="#f0e6d3"),
    ))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=520,
                      margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Top correlations with quality
    st.markdown('<div class="section-title">Feature Correlations with Quality</div>', unsafe_allow_html=True)
    q_corr = corr_data["quality"].drop("quality").sort_values(key=abs, ascending=False)
    fig_bar = px.bar(
        x=q_corr.values, y=q_corr.index, orientation="h",
        color=q_corr.values,
        color_continuous_scale=[[0, "#2d1017"], [0.5, "#9b3a52"], [1, "#c0392b"]],
        labels={"x": "Correlation", "y": "Feature"},
    )
    fig_bar.update_layout(**PLOTLY_TEMPLATE["layout"], height=340,
                          margin=dict(l=0, r=0, t=10, b=0),
                          coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter matrix
    st.markdown('<div class="section-title">Scatter Matrix — Top 5 Correlated Features</div>', unsafe_allow_html=True)
    top5 = q_corr.abs().nlargest(5).index.tolist()
    fig_sm = px.scatter_matrix(
        dff[top5 + ["color"]].sample(min(800, len(dff)), random_state=1),
        dimensions=top5, color="color",
        color_discrete_map={"red": COLORS["red"], "white": COLORS["white"]},
        opacity=0.4,
    )
    fig_sm.update_traces(marker=dict(size=3))
    fig_sm.update_layout(**PLOTLY_TEMPLATE["layout"], height=500,
                         margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_sm, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — QUALITY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Feature Means by Quality Score</div>', unsafe_allow_html=True)
    sel_feats = st.multiselect(
        "Choose Features", num_cols,
        default=["alcohol", "volatile acidity", "sulphates", "citric acid"],
    )
    if sel_feats:
        mean_by_q = dff.groupby("quality")[sel_feats].mean().reset_index()
        fig = px.line(
            mean_by_q.melt(id_vars="quality", value_vars=sel_feats),
            x="quality", y="value", color="variable",
            markers=True,
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=360,
                          margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="Mean Value",
                          xaxis_title="Quality Score")
        st.plotly_chart(fig, use_container_width=True)

    # Violin plots
    st.markdown('<div class="section-title">Violin Plots — Feature vs Quality</div>', unsafe_allow_html=True)
    viol_feat = st.selectbox("Feature for Violin", num_cols, key="viol",
                             index=num_cols.index("alcohol") if "alcohol" in num_cols else 0)
    fig_v = px.violin(
        dff, x="quality", y=viol_feat, color="color", box=True, points=False,
        color_discrete_map={"red": COLORS["red"], "white": COLORS["white"]},
    )
    fig_v.update_layout(**PLOTLY_TEMPLATE["layout"], height=380,
                        margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_v, use_container_width=True)

    # Good vs Bad comparison
    st.markdown('<div class="section-title">Good vs Not-Good Wine — Feature Comparison</div>', unsafe_allow_html=True)
    good_means = dff.groupby("good")[num_cols].mean().T
    good_means.columns = ["Not Good (0)", "Good (1)"]
    good_means["Difference"] = good_means["Good (1)"] - good_means["Not Good (0)"]
    good_means["Diff %"] = (good_means["Difference"] / good_means["Not Good (0)"] * 100).round(1)

    fig_comp = px.bar(
        good_means.reset_index().rename(columns={"index": "Feature"}),
        x="Feature", y=["Not Good (0)", "Good (1)"],
        barmode="group",
        color_discrete_sequence=[COLORS["red"], COLORS["white"]],
    )
    fig_comp.update_layout(**PLOTLY_TEMPLATE["layout"], height=360,
                           margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_comp, use_container_width=True)

    # Parallel coordinates
    st.markdown('<div class="section-title">Parallel Coordinates — All Features by Quality</div>', unsafe_allow_html=True)
    pc_sample = dff[num_cols + ["quality"]].sample(min(600, len(dff)), random_state=42)
    fig_pc = px.parallel_coordinates(
        pc_sample, color="quality",
        color_continuous_scale=[[0, "#2d1017"], [0.5, "#9b3a52"], [1, "#e8c97c"]],
        dimensions=num_cols,
    )
    fig_pc.update_layout(**PLOTLY_TEMPLATE["layout"], height=440,
                         margin=dict(l=60, r=20, t=30, b=30))
    st.plotly_chart(fig_pc, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICT QUALITY  🤖
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">🤖 Wine Quality Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='insight-box'>Enter the physicochemical properties of a wine sample below. "
        "The Random Forest model will predict whether it is <b>Good Quality</b> (score ≥ 7) "
        "or <b>Not Good Quality</b> (score &lt; 7).</div>",
        unsafe_allow_html=True
    )

    # ── Check if model is available ──────────────────────────────────────────
    if model is None:
        st.warning(
            "⚠️ **model.pkl not found.** Place `model.pkl` in the same folder as this script "
            "and restart the app to enable predictions.\n\n"
            "You can still explore the input sliders below — the prediction will run once the model is loaded."
        )

    st.markdown("")

    # ── Dataset stats for slider defaults / ranges ───────────────────────────
    feat_stats = df[FEATURE_NAMES].agg(["min", "max", "mean"]).T

    # ── Input form — 3 columns ───────────────────────────────────────────────
    st.markdown("##### ✏️ Enter Wine Parameters")

    col1, col2, col3 = st.columns(3)

    def make_slider(col, label, key, fmt="%.3f"):
        lo  = float(feat_stats.loc[label, "min"])
        hi  = float(feat_stats.loc[label, "max"])
        mid = float(feat_stats.loc[label, "mean"])
        step = round((hi - lo) / 200, 5)
        return col.slider(label.title(), min_value=lo, max_value=hi,
                          value=round(mid, 3), step=step,
                          format=fmt, key=key)

    with col1:
        fixed_acidity    = make_slider(col1, "fixed acidity",        "fa")
        volatile_acidity = make_slider(col1, "volatile acidity",     "va")
        citric_acid      = make_slider(col1, "citric acid",          "ca")
        residual_sugar   = make_slider(col1, "residual sugar",       "rs")

    with col2:
        chlorides          = make_slider(col2, "chlorides",              "ch", "%.4f")
        free_sulfur        = make_slider(col2, "free sulfur dioxide",    "fsd", "%.1f")
        total_sulfur       = make_slider(col2, "total sulfur dioxide",   "tsd", "%.1f")

    with col3:
        density   = make_slider(col3, "density",   "den", "%.4f")
        ph        = make_slider(col3, "pH",        "ph")
        sulphates = make_slider(col3, "sulphates", "su")
        alcohol   = make_slider(col3, "alcohol",   "alc")

    # ── Assemble input array ─────────────────────────────────────────────────
    user_input = np.array([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur, total_sulfur, density, ph, sulphates, alcohol
    ]])

    st.markdown("")
    predict_btn = st.button("🔮 Predict Wine Quality", use_container_width=False)

    if predict_btn:
        if model is None:
            st.error("Model not loaded. Please add `model.pkl` to the app folder.")
        else:
            # Scale using dataset statistics (same as training StandardScaler)
            scaler = build_scaler_from_data(df)
            input_scaled = scaler.transform(user_input)

            prediction    = model.predict(input_scaled)[0]
            proba         = model.predict_proba(input_scaled)[0]   # [prob_bad, prob_good]
            confidence    = proba[prediction] * 100

            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-good">
                        <div style="font-size:3rem;">🥂</div>
                        <div class="result-label" style="color:#27ae60;">Good Quality Wine</div>
                        <div class="result-sub">The model predicts this wine has a quality score ≥ 7</div>
                        <div style="color:#27ae60; font-size:1.1rem; margin-top:8px;">
                            Confidence: <b>{confidence:.1f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-bad">
                        <div style="font-size:3rem;">🍷</div>
                        <div class="result-label" style="color:#c0392b;">Not Good Quality</div>
                        <div class="result-sub">The model predicts this wine has a quality score &lt; 7</div>
                        <div style="color:#c0392b; font-size:1.1rem; margin-top:8px;">
                            Confidence: <b>{confidence:.1f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── Probability gauge chart ──────────────────────────────────────
            st.markdown("")
            st.markdown("##### 📊 Prediction Confidence")
            prob_df = pd.DataFrame({
                "Class":       ["Not Good Quality", "Good Quality"],
                "Probability": [proba[0] * 100,     proba[1] * 100],
                "Color":       [COLORS["red"],       "#27ae60"],
            })
            fig_prob = px.bar(
                prob_df, x="Class", y="Probability",
                color="Class",
                color_discrete_map={"Not Good Quality": COLORS["red"], "Good Quality": "#27ae60"},
                text=prob_df["Probability"].apply(lambda v: f"{v:.1f}%"),
                range_y=[0, 100],
            )
            fig_prob.update_traces(textposition="outside", textfont_size=14)
            fig_prob.update_layout(
                **PLOTLY_TEMPLATE["layout"],
                height=320,
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Probability (%)",
                xaxis_title="",
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # ── Input summary ────────────────────────────────────────────────
            with st.expander("📋 View Input Summary"):
                input_df = pd.DataFrame(user_input, columns=FEATURE_NAMES).T
                input_df.columns = ["Your Input"]
                input_df["Dataset Mean"] = feat_stats["mean"].values
                input_df["vs Mean"] = (
                    (input_df["Your Input"] - input_df["Dataset Mean"])
                    / input_df["Dataset Mean"] * 100
                ).round(1).astype(str) + "%"
                st.dataframe(input_df.style.format({"Your Input": "{:.4f}", "Dataset Mean": "{:.4f}"}),
                             use_container_width=True)

    # ── Quick-fill sample wines ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### 🧪 Try a Sample Wine")
    st.caption("Click a preset to auto-fill the sliders with a known wine profile.")

    sample_col1, sample_col2, sample_col3 = st.columns(3)

    SAMPLES = {
        "🟢 Likely Good":  dict(fa=7.5, va=0.5, ca=0.36, rs=6.1, ch=0.071, fsd=17.0, tsd=102.0, den=0.9978, ph=3.35, su=0.8, alc=10.5),
        "🔴 Likely Bad":   dict(fa=8.0, va=0.8, ca=0.1,  rs=2.0, ch=0.09,  fsd=10.0, tsd=50.0,  den=0.9990, ph=3.2,  su=0.5, alc=9.0),
        "⚖️  Borderline":  dict(fa=6.8, va=0.6, ca=0.25, rs=4.0, ch=0.075, fsd=14.0, tsd=78.0,  den=0.9985, ph=3.3,  su=0.65, alc=10.0),
    }

    for (label, vals), col in zip(SAMPLES.items(), [sample_col1, sample_col2, sample_col3]):
        if col.button(label, use_container_width=True):
            st.session_state["fa"]  = vals["fa"]
            st.session_state["va"]  = vals["va"]
            st.session_state["ca"]  = vals["ca"]
            st.session_state["rs"]  = vals["rs"]
            st.session_state["ch"]  = vals["ch"]
            st.session_state["fsd"] = vals["fsd"]
            st.session_state["tsd"] = vals["tsd"]
            st.session_state["den"] = vals["den"]
            st.session_state["ph"]  = vals["ph"]
            st.session_state["su"]  = vals["su"]
            st.session_state["alc"] = vals["alc"]
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — RAW DATA
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">Filtered Dataset</div>', unsafe_allow_html=True)
    st.caption(f"Showing {len(dff):,} rows × {dff.shape[1]} columns")
    st.dataframe(dff, use_container_width=True, height=500)

    col_dl1, col_dl2 = st.columns([1, 4])
    with col_dl1:
        csv_bytes = dff.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download CSV", data=csv_bytes,
            file_name="wine_filtered.csv", mime="text/csv",
        )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#5c2130; font-size:0.75rem;
     margin-top:40px; letter-spacing:0.1em; text-transform:uppercase;'>
    Wine EDA Dashboard · Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
