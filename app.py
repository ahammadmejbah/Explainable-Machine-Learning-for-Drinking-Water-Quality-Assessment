from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(
    page_title="GB Water Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "Datasets" / "gb_water_quality_2000_2026.csv"
EXPECTED_COLUMNS = [
    "Study_ID", "Authors_Source", "Year", "District", "Location",
    "Water_Source_Type", "Sample_Type", "Season", "N_Samples",
    "Temperature_C", "pH", "EC_uS_cm", "Turbidity_NTU", "TDS_mg_L",
    "DO_mg_L", "Hardness_mg_L", "Alkalinity_mg_L", "Ca_mg_L", "Mg_mg_L",
    "Cl_mg_L", "Fe_mg_L", "As_mg_L", "E_coli_CFU_100ml",
    "Total_Coliform_CFU_100ml", "WHO_pH_Pass", "WHO_Turbidity_Pass",
    "WHO_Ecoli_Pass", "Overall_Safe", "Notes",
]
NUMERIC_COLUMNS = [
    "N_Samples", "Temperature_C", "pH", "EC_uS_cm", "Turbidity_NTU",
    "TDS_mg_L", "DO_mg_L", "Hardness_mg_L", "Alkalinity_mg_L", "Ca_mg_L",
    "Mg_mg_L", "Cl_mg_L", "Fe_mg_L", "As_mg_L",
    "E_coli_CFU_100ml", "Total_Coliform_CFU_100ml",
]
FLAG_COLUMNS = ["WHO_pH_Pass", "WHO_Turbidity_Pass", "WHO_Ecoli_Pass", "Overall_Safe"]


def first_number(value: object) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def _flag_start(row: list[str]) -> int | None:
    """Find the four compliance fields even when notes contain unquoted commas."""
    for index in range(20, len(row) - 3):
        values = [item.strip() for item in row[index:index + 4]]
        if all(item in {"", "Yes", "No"} for item in values) and any(values):
            return index
    return None


@st.cache_data(show_spinner=False)
def load_dataset() -> tuple[pd.DataFrame, int]:
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.reader(handle))

    records: list[dict[str, str]] = []
    repaired_rows = 0
    for raw in raw_rows[1:]:
        if not raw or not raw[0].strip():
            continue
        year_index = next(
            (index for index, value in enumerate(raw) if re.fullmatch(r"20\d{2}(?:-20\d{2})?", value.strip())),
            2,
        )
        flag_index = _flag_start(raw)
        if flag_index is None:
            flag_index = max(len(raw) - 5, 24)
        values = [""] * len(EXPECTED_COLUMNS)
        values[0] = raw[0].strip()
        values[1] = ", ".join(item.strip() for item in raw[1:year_index]).strip()
        values[2] = raw[year_index].strip() if year_index < len(raw) else ""
        metadata = raw[year_index + 1:flag_index]
        for offset, value in enumerate(metadata[:20], start=3):
            values[offset] = value.strip()
        flags = [item.strip() for item in raw[flag_index:flag_index + 4]]
        values[24:28] = flags
        values[28] = ", ".join(item.strip() for item in raw[flag_index + 4:]).strip()
        if len(raw) != len(EXPECTED_COLUMNS):
            repaired_rows += 1
        records.append(dict(zip(EXPECTED_COLUMNS, values)))

    frame = pd.DataFrame(records)
    for column in NUMERIC_COLUMNS:
        frame[column] = frame[column].map(first_number)
    frame["Year_Start"] = pd.to_numeric(frame["Year"].str[:4], errors="coerce")
    frame["Risk_Score"] = frame.apply(risk_score, axis=1)
    frame["Risk_Band"] = frame["Risk_Score"].map(risk_band)
    return frame, repaired_rows


def risk_score(row: pd.Series) -> int:
    score = 0
    if row.get("WHO_pH_Pass") == "No":
        score += 25
    if row.get("WHO_Turbidity_Pass") == "No":
        score += 25
    if row.get("WHO_Ecoli_Pass") == "No":
        score += 40
    if row.get("Overall_Safe") == "No":
        score += 10
    if row.get("WHO_Ecoli_Pass") == "" and row.get("Overall_Safe") == "No":
        score += 20
    return min(score, 100)


def risk_band(score: int) -> str:
    if score >= 70:
        return "High risk"
    if score >= 35:
        return "Watch"
    return "Lower risk"


def clean_label(value: object) -> str:
    text = str(value).strip()
    return text if text and text.lower() != "nan" else "Not reported"


def model_pipeline(model: object, scale: bool = False) -> Pipeline:
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def specificity_score(y_true: object, y_pred: object) -> float:
    true_negatives = int(((np.asarray(y_true) == 0) & (np.asarray(y_pred) == 0)).sum())
    false_positives = int(((np.asarray(y_true) == 0) & (np.asarray(y_pred) == 1)).sum())
    return true_negatives / (true_negatives + false_positives) if true_negatives + false_positives else 0.0


@st.cache_data(show_spinner=False)
def run_model_lab(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    feature_columns = [
        column for column in NUMERIC_COLUMNS
        if column != "Total_Coliform_CFU_100ml" and frame[column].notna().sum() >= 5
    ]
    x_data = frame[feature_columns].copy()
    y_data = (frame["Risk_Score"] >= 70).astype(int)
    models = {
        "Logistic regression": model_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42), scale=True),
        "Decision tree": model_pipeline(DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)),
        "Random forest": model_pipeline(RandomForestClassifier(n_estimators=120, max_depth=4, class_weight="balanced", random_state=42)),
        "Extra trees": model_pipeline(ExtraTreesClassifier(n_estimators=120, max_depth=4, class_weight="balanced", random_state=42)),
        "Gradient boosting": model_pipeline(GradientBoostingClassifier(n_estimators=60, max_depth=2, learning_rate=0.05, random_state=42)),
        "AdaBoost": model_pipeline(AdaBoostClassifier(n_estimators=60, learning_rate=0.05, random_state=42)),
        "Histogram gradient boosting": model_pipeline(HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=6, learning_rate=0.05, random_state=42)),
        "K-nearest neighbors": model_pipeline(KNeighborsClassifier(n_neighbors=3, weights="distance"), scale=True),
        "Support vector machine": model_pipeline(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42), scale=True),
        "Linear discriminant analysis": model_pipeline(LinearDiscriminantAnalysis(), scale=True),
        "Gaussian Naive Bayes": model_pipeline(GaussianNB(), scale=True),
    }
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "specificity": make_scorer(specificity_score),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": make_scorer(roc_auc_score, response_method="predict_proba"),
        "pr_auc": make_scorer(average_precision_score, response_method="predict_proba"),
        "mcc": make_scorer(matthews_corrcoef),
    }
    comparison_rows: list[dict[str, object]] = []
    importance_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for name, pipeline in models.items():
        scores = cross_validate(pipeline, x_data, y_data, cv=folds, scoring=scoring, error_score=np.nan)
        comparison_rows.append({"Model": name, **{
            metric: float(np.nanmean(scores[f"test_{metric}"]))
            for metric in scoring
        }, **{
            f"{metric} std": float(np.nanstd(scores[f"test_{metric}"]))
            for metric in scoring
        }})
        pipeline.fit(x_data, y_data)
        permutation = permutation_importance(pipeline, x_data, y_data, scoring="balanced_accuracy", n_repeats=12, random_state=42)
        importance_frames.append(pd.DataFrame({"Feature": feature_columns, "Importance": permutation.importances_mean, "Model": name, "Method": "Permutation importance"}))
        fitted_model = pipeline.named_steps["model"]
        if hasattr(fitted_model, "feature_importances_"):
            importance_frames.append(pd.DataFrame({"Feature": feature_columns, "Importance": fitted_model.feature_importances_[:len(feature_columns)], "Model": name, "Method": "Native tree importance"}))
        if hasattr(fitted_model, "coef_"):
            importance_frames.append(pd.DataFrame({"Feature": feature_columns, "Importance": np.abs(fitted_model.coef_[0][:len(feature_columns)]), "Model": name, "Method": "Coefficient magnitude"}))
        predictions = pipeline.predict_proba(x_data)[:, 1]
        oof_predictions = cross_val_predict(pipeline, x_data, y_data, cv=folds, method="predict_proba")[:, 1]
        prediction_frames.append(pd.DataFrame({
            "Study_ID": frame["Study_ID"],
            "Observed": y_data,
            "Predicted probability": predictions,
            "OOF probability": oof_predictions,
            "Model": name,
        }))
    return (
        pd.DataFrame(comparison_rows),
        pd.concat(importance_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
        feature_columns,
    )


def metric_card(label: str, value: str, detail: str, tone: str = "blue") -> None:
    st.markdown(
        f'<div class="metric-card {tone}"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#14211f; --muted:#62716d; --teal:#087f73; --aqua:#d8f1ec; --orange:#e77c42; --gold:#e9b949; --line:#d9e4e0; --paper:#fbfdfc; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: var(--paper); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    .block-container { max-width: 1380px; padding: 2.4rem 3rem 3rem; }
    .hero { display:flex; justify-content:space-between; gap:2rem; padding: 1.2rem 0 2rem; border-bottom: 1px solid var(--line); margin-bottom: 1.5rem; }
    .hero-copy { max-width: 760px; }
    .eyebrow { color: var(--teal); font-size: .75rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
    .hero h1 { font-size: clamp(2.2rem, 4vw, 4.4rem); line-height: .98; margin: .45rem 0 .8rem; max-width: 760px; }
    .hero p { color: var(--muted); font-size: 1.05rem; max-width: 730px; margin: 0; }
    .hero-stamp { align-self:flex-end; min-width:205px; border-left:3px solid var(--orange); padding: .2rem 0 .2rem 1rem; }
    .hero-stamp strong { display:block; font-family:'Space Grotesk'; font-size:1.35rem; }
    .hero-stamp span { color:var(--muted); font-size:.78rem; }
    .topline { display:flex; align-items:center; justify-content:space-between; border-top:1px solid var(--line); border-bottom:1px solid var(--line); padding:.55rem 0; margin-bottom:1.3rem; color:var(--muted); font-size:.72rem; letter-spacing:.04em; text-transform:uppercase; }
    .topline strong { color:var(--teal); }
    .topline-status { display:flex; align-items:center; gap:.45rem; }
    .status-dot { width:7px; height:7px; border-radius:50%; background:#2caa73; display:inline-block; box-shadow:0 0 0 3px #d9f1e5; }
    .metric-card { background: white; border: 1px solid var(--line); border-top: 4px solid var(--teal); border-radius: 6px; box-shadow: 0 8px 24px rgba(20,33,31,.045); padding: 1.1rem 1.2rem; min-height: 122px; }
    .metric-card.orange { border-top-color: var(--orange); } .metric-card.blue { border-top-color: #2387a7; }
    .metric-label { color: var(--muted); font-size: .78rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
    .metric-value { font-family: 'Space Grotesk'; font-size: 2.15rem; font-weight: 700; margin: .25rem 0; }
    .metric-detail { color: var(--muted); font-size: .8rem; }
    .callout { background: var(--aqua); border: 1px solid #b8dfd7; border-left: 4px solid var(--teal); border-radius: 5px; padding: .9rem 1rem; margin: 1rem 0 1.4rem; color: #1d4c47; }
    .section-kicker { color: var(--muted); font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin: 2rem 0 .2rem; }
    .section-kicker + h2 { margin-top: 0; }
    .panel-note { background:#fff; border:1px solid var(--line); border-radius:6px; padding:1.15rem 1.25rem; min-height:250px; box-shadow: 0 8px 24px rgba(20,33,31,.035); }
    .panel-note h3 { margin-top:0; }
    .toolbar { display:flex; align-items:center; justify-content:space-between; gap:1rem; background:#eef7f4; border:1px solid #cfe6df; border-radius:6px; padding:.7rem .9rem; margin: .5rem 0 1.4rem; }
    .toolbar-copy { color:#365a54; font-size:.82rem; }
    .coverage-strip { color:var(--muted); font-size:.82rem; padding-top:.65rem; }
    .coverage-strip strong { color:var(--ink); }
    .brief { display:grid; grid-template-columns:1.3fr 1fr 1fr; gap:1px; background:var(--line); border:1px solid var(--line); border-radius:6px; overflow:hidden; margin:1.1rem 0 1.8rem; }
    .brief-cell { background:white; padding:1rem 1.15rem; min-height:106px; }
    .brief-cell.primary { background:#0d756b; color:white; }
    .brief-label { color:var(--muted); font-size:.7rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
    .brief-cell.primary .brief-label { color:#bde4dc; }
    .brief-value { font-family:'Space Grotesk'; font-size:1.3rem; font-weight:700; margin-top:.35rem; }
    .brief-detail { color:var(--muted); font-size:.78rem; margin-top:.22rem; }
    .brief-cell.primary .brief-detail { color:#e0f3ef; }
    .stDownloadButton button { border:1px solid var(--teal); color:var(--teal); background:white; font-weight:700; }
    .stDataFrame { border: 1px solid var(--line); }
    div[data-testid="stMetric"] { background: white; border: 1px solid var(--line); padding: .7rem; }
    [data-testid="stSidebar"] { background: #f1f7f5; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h2 { font-size: 1.1rem; }
    .small-note { color: var(--muted); font-size: .8rem; }
    .research-nav { text-align:center; margin: .4rem 0 1.7rem; }
    .research-nav-label { color:var(--muted); font-size:.78rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.6rem; }
    .footer { border-top:1px solid var(--line); margin-top:3rem; padding-top:1rem; color:var(--muted); font-size:.8rem; text-align:center; }
    @media (max-width: 760px) {
        .block-container { padding: 1.4rem 1rem 2rem; }
        .hero { display:block; padding-top:.5rem; }
        .hero-stamp { margin-top:1.3rem; }
        .brief { grid-template-columns:1fr; }
        .topline { display:block; line-height:1.7; }
        .topline-status { margin-top:.25rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    data, repaired_rows = load_dataset()
except FileNotFoundError:
    st.error(f"Dataset not found at {DATA_PATH}")
    st.stop()

st.sidebar.markdown("## GB Water Intelligence")
st.sidebar.caption("Evidence-led water safety analysis")
st.sidebar.markdown('<div class="small-note">MULTI-STUDY COMPILATION<br><strong>2000 — 2026</strong></div>', unsafe_allow_html=True)
st.sidebar.markdown("### Filters")
districts = sorted(data["District"].map(clean_label).unique())
selected_districts = st.sidebar.multiselect("District / study area", districts, default=districts)
risk_options = ["High risk", "Watch", "Lower risk"]
selected_risks = st.sidebar.multiselect("Risk band", risk_options, default=risk_options)
year_min = int(data["Year_Start"].min())
year_max = int(data["Year_Start"].max())
selected_years = st.sidebar.slider("Publication / sampling year", year_min, year_max, (year_min, year_max))

filtered = data[
    data["District"].map(clean_label).isin(selected_districts)
    & data["Risk_Band"].isin(selected_risks)
    & data["Year_Start"].between(*selected_years)
].copy()

st.markdown(
    '<div class="hero"><div class="hero-copy"><div class="eyebrow">Water quality · 2000—2026</div>'
    '<h1>What is making water unsafe?</h1>'
    '<p>Explore contamination patterns, compare study areas, and see which reported indicators drive the risk picture across Gilgit-Baltistan.</p></div>'
    '<div class="hero-stamp"><strong>Gilgit-Baltistan</strong><span>Pakistan · 25 published records</span></div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="topline"><span><strong>FIELD INTELLIGENCE /</strong> Water safety command center</span>'
    '<span class="topline-status"><span class="status-dot"></span> Dataset loaded · 25 source records</span></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="research-nav"><div class="research-nav-label">Research and methods</div></div>',
    unsafe_allow_html=True,
)
research_left, research_center, research_right = st.columns([1, 2, 1])
with research_center:
    research_page = Path(__file__).parent / "pages" / "Research_Paper.py"
    if st.button("📄  Read the Full Research Paper", use_container_width=True, type="primary"):
        st.switch_page(research_page)

if repaired_rows:
    st.markdown(
        f'<div class="callout"><strong>Data quality note.</strong> {repaired_rows} of {len(data)} source rows contain unquoted commas. '
        "The loader keeps every study and repairs the compliance columns, but some free-text fields remain approximate. Use original papers for final claims.</div>",
        unsafe_allow_html=True,
    )

total = len(filtered)
high_risk = int((filtered["Risk_Band"] == "High risk").sum())
microbial_fail = int((filtered["WHO_Ecoli_Pass"] == "No").sum())
reported_measurements = int(filtered[NUMERIC_COLUMNS].notna().sum().sum())
col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Studies in view", str(total), "Published records after filters")
with col2:
    metric_card("High-risk records", str(high_risk), f"{(high_risk / total * 100 if total else 0):.0f}% of current view", "orange")
with col3:
    metric_card("Microbial failures", str(microbial_fail), "Reported E. coli guideline failures", "orange")
with col4:
    metric_card("Measurements", f"{reported_measurements:,}", "Numeric values reported in source")

st.markdown(
    '<div class="toolbar"><div class="toolbar-copy"><strong>Current view ready.</strong> Export the filtered evidence set for review or reporting.</div></div>',
    unsafe_allow_html=True,
)
toolbar_col, coverage_col = st.columns([1, 2])
with toolbar_col:
    st.download_button(
        "Download filtered records",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="gb_water_quality_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
with coverage_col:
    coverage = (reported_measurements / (len(filtered) * len(NUMERIC_COLUMNS)) * 100) if len(filtered) else 0
    st.markdown(
        f'<div class="coverage-strip"><strong>Evidence coverage:</strong> {coverage:.0f}% of numeric fields are reported in the current view. '
        'Blank values are preserved and never interpreted as compliant.</div>',
        unsafe_allow_html=True,
    )

if total:
    district_risk = filtered.assign(District=filtered["District"].map(clean_label)).groupby("District")["Risk_Score"].mean().sort_values(ascending=False)
    leading_area = district_risk.index[0] if not district_risk.empty else "Not reported"
    leading_score = district_risk.iloc[0] if not district_risk.empty else 0
    safe_records = int((filtered["Overall_Safe"] == "Yes").sum())
    brief_headline = "Risk is concentrated in the current view" if high_risk else "No high-risk records in the current view"
else:
    leading_area, leading_score, safe_records, brief_headline = "No records", 0, 0, "Adjust the filters to restore the briefing"
st.markdown(
    f'<div class="brief"><div class="brief-cell primary"><div class="brief-label">Executive brief</div><div class="brief-value">{brief_headline}</div><div class="brief-detail">A live readout of the selected evidence set</div></div>'
    f'<div class="brief-cell"><div class="brief-label">Highest average risk</div><div class="brief-value">{leading_area}</div><div class="brief-detail">Mean rule-based score: {leading_score:.0f} / 100</div></div>'
    f'<div class="brief-cell"><div class="brief-label">Reported safe records</div><div class="brief-value">{safe_records} of {total}</div><div class="brief-detail">Based on the source overall judgement</div></div></div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">01 · Executive readout</div><h2>The signal</h2>', unsafe_allow_html=True)
chart_col, insight_col = st.columns([1.45, 1])
with chart_col:
    risk_counts = filtered["Risk_Band"].value_counts().reindex(risk_options, fill_value=0).rename_axis("Risk").reset_index(name="Records")
    fig = px.bar(risk_counts, x="Risk", y="Records", color="Risk", color_discrete_map={"High risk": "#e77c42", "Watch": "#e9b949", "Lower risk": "#087f73"})
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=20, b=0), showlegend=False, plot_bgcolor="#fbfdfc", paper_bgcolor="#fbfdfc", font_color="#14211f", font_family="DM Sans", yaxis_title="", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
with insight_col:
    st.markdown('<div class="panel-note"><h3>Reading the evidence</h3>', unsafe_allow_html=True)
    st.write("Microbial contamination is the strongest signal in this compilation: a reported zero-tolerance failure immediately moves a record into the high-risk band.")
    st.write("Many rows are study summaries rather than individual samples. Missing numeric values are not treated as safe; they are shown as unreported.")
    st.markdown('<div class="small-note">Risk bands are transparent rule-based explanations, not a clinical or regulatory certification.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-kicker">01B · Explainability</div><h2>What drives the risk?</h2>', unsafe_allow_html=True)
driver_labels = {
    "WHO_Ecoli_Pass": "E. coli failure",
    "WHO_Turbidity_Pass": "Turbidity failure",
    "WHO_pH_Pass": "pH failure",
    "Overall_Safe": "Overall unsafe judgement",
}
driver_rows = [
    {"Driver": label, "Records": int((filtered[column] == "No").sum())}
    for column, label in driver_labels.items()
]
driver_data = pd.DataFrame(driver_rows).sort_values("Records", ascending=True)
driver_col, driver_note_col = st.columns([1.4, 1])
with driver_col:
    fig = px.bar(driver_data, x="Records", y="Driver", orientation="h", color="Driver", color_discrete_sequence=["#e77c42", "#e9b949", "#2387a7", "#087f73"])
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, plot_bgcolor="#fbfdfc", paper_bgcolor="#fbfdfc", font_family="DM Sans", xaxis_title="Reported failures", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
with driver_note_col:
    st.markdown('<div class="panel-note"><h3>Why this matters</h3>', unsafe_allow_html=True)
    st.write("The driver view separates observed failures from the composite risk band. It helps reviewers see whether a record is being flagged by microbial evidence, physical quality, or the original study's overall judgement.")
    st.markdown('<div class="small-note">Counts reflect records, not individual samples. A study-level summary can represent many underlying observations.</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-kicker">02 · Geographic and temporal context</div><h2>Compare places and time</h2>', unsafe_allow_html=True)
left, right = st.columns(2)
with left:
    district_counts = filtered.assign(District=filtered["District"].map(clean_label)).groupby(["District", "Risk_Band"], as_index=False).size().rename(columns={"size": "Records"})
    fig = px.bar(district_counts, x="Records", y="District", color="Risk_Band", orientation="h", color_discrete_map={"High risk": "#e77c42", "Watch": "#e9b949", "Lower risk": "#087f73"})
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor="#fbfdfc", paper_bgcolor="#fbfdfc", legend_title_text="", font_family="DM Sans", xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
with right:
    timeline = filtered.groupby(["Year_Start", "Risk_Band"], as_index=False).size().rename(columns={"size": "Records", "Year_Start": "Year"})
    fig = px.scatter(timeline, x="Year", y="Records", size="Records", color="Risk_Band", hover_name="Risk_Band", color_discrete_map={"High risk": "#e77c42", "Watch": "#e9b949", "Lower risk": "#087f73"})
    fig.update_layout(height=430, margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor="#fbfdfc", paper_bgcolor="#fbfdfc", legend_title_text="", font_family="DM Sans", xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section-kicker">02B · Water sources and seasonality</div><h2>Where and when samples were taken</h2>', unsafe_allow_html=True)
source_col, season_col = st.columns(2)
with source_col:
    source_counts = filtered.assign(Water_Source_Type=filtered["Water_Source_Type"].map(clean_label)).groupby(["Water_Source_Type", "Risk_Band"], as_index=False).size().rename(columns={"size": "Records"})
    fig = px.bar(source_counts, x="Records", y="Water_Source_Type", color="Risk_Band", orientation="h", color_discrete_map={"High risk": "#e77c42", "Watch": "#e9b949", "Lower risk": "#087f73"})
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor="#fbfdfc", paper_bgcolor="#fbfdfc", legend_title_text="", font_family="DM Sans", xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Records by water source type")
with season_col:
    season_data = filtered.assign(Season=filtered["Season"].map(clean_label))
    if season_data["pH"].notna().sum() >= 3:
        fig = px.box(season_data, x="Season", y="pH", color="Season", points="all", color_discrete_sequence=["#087f73", "#e77c42", "#e9b949", "#2387a7"])
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0), showlegend=False, plot_bgcolor="#fbfdfc", paper_bgcolor="#fbfdfc", font_family="DM Sans", xaxis_title="", yaxis_title="pH")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("pH spread by reported season")
    else:
        st.info("Not enough reported pH values to chart seasonal spread.")

source_table = (
    filtered.assign(Water_Source_Type=filtered["Water_Source_Type"].map(clean_label))
    .groupby("Water_Source_Type")
    .agg(Records=("Study_ID", "count"), **{"Mean risk score": ("Risk_Score", "mean")}, **{"High-risk records": ("Risk_Band", lambda values: int((values == "High risk").sum()))})
    .reset_index()
    .sort_values("Records", ascending=False)
)
source_table["Mean risk score"] = source_table["Mean risk score"].round(1)
st.markdown("#### Water source summary table")
st.dataframe(source_table, use_container_width=True, hide_index=True, height=240)

st.markdown('<div class="section-kicker">02C · Distribution and correlation</div><h2>How reported indicators relate</h2>', unsafe_allow_html=True)
stats_col, corr_col = st.columns([1, 1.2])
with stats_col:
    summary_stats = filtered[NUMERIC_COLUMNS].describe().T[["count", "mean", "std", "min", "max"]].round(2)
    summary_stats.index.name = "Indicator"
    summary_stats = summary_stats.reset_index()
    st.markdown("#### Descriptive statistics")
    st.dataframe(summary_stats, use_container_width=True, hide_index=True, height=380)
with corr_col:
    correlation_columns = [column for column in NUMERIC_COLUMNS if filtered[column].notna().sum() >= 3]
    if len(correlation_columns) >= 2:
        correlation = filtered[correlation_columns].corr().round(2)
        fig = px.imshow(correlation, color_continuous_scale=["#e77c42", "#fbfdfc", "#087f73"], zmin=-1, zmax=1, text_auto=True, aspect="auto")
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor="#fbfdfc", paper_bgcolor="#fbfdfc", font_family="DM Sans")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Pairwise correlation of reported numeric indicators")
    else:
        st.info("Not enough overlapping numeric indicators to compute correlations.")

st.markdown('<div class="section-kicker">02D · Contaminant deep dive</div><h2>Where the evidence concentrates</h2>', unsafe_allow_html=True)
top_risk_col, compliance_col = st.columns([1.2, 1])
with top_risk_col:
    top_risk_table = (
        filtered.assign(District=filtered["District"].map(clean_label))
        .sort_values("Risk_Score", ascending=False)
        [["Study_ID", "District", "Year", "Water_Source_Type", "Risk_Score", "Risk_Band"]]
        .head(10)
        .reset_index(drop=True)
    )
    top_risk_table.columns = [column.replace("_", " ") for column in top_risk_table.columns]
    st.markdown("#### Top 10 highest-risk studies")
    st.dataframe(top_risk_table, use_container_width=True, hide_index=True, height=380)
with compliance_col:
    compliance_rows = []
    for column, label in {"WHO_pH_Pass": "pH", "WHO_Turbidity_Pass": "Turbidity", "WHO_Ecoli_Pass": "E. coli", "Overall_Safe": "Overall safe"}.items():
        counts = filtered[column].replace("", "Not reported").value_counts()
        compliance_rows.append({
            "Guideline": label,
            "Pass": int(counts.get("Yes", 0)),
            "Fail": int(counts.get("No", 0)),
            "Not reported": int(counts.get("Not reported", 0)),
        })
    compliance_table = pd.DataFrame(compliance_rows)
    st.markdown("#### WHO compliance summary")
    st.dataframe(compliance_table, use_container_width=True, hide_index=True, height=180)

    completeness = (filtered[NUMERIC_COLUMNS].notna().mean() * 100).round(0).sort_values(ascending=True).reset_index()
    completeness.columns = ["Indicator", "Reported (%)"]
    fig = px.bar(completeness, x="Reported (%)", y="Indicator", orientation="h", range_x=[0, 100], color="Reported (%)", color_continuous_scale=["#e77c42", "#087f73"])
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=15, b=0), coloraxis_showscale=False, plot_bgcolor="#fbfdfc", paper_bgcolor="#fbfdfc", font_family="DM Sans", xaxis_title="Reported (%)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Data completeness by indicator in the current view")

microbial_data = filtered.dropna(subset=["Turbidity_NTU", "E_coli_CFU_100ml"])
if len(microbial_data) >= 3:
    fig = px.scatter(
        microbial_data, x="Turbidity_NTU", y="E_coli_CFU_100ml", color="Risk_Band", hover_name="Study_ID",
        color_discrete_map={"High risk": "#e77c42", "Watch": "#e9b949", "Lower risk": "#087f73"},
    )
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor="#fbfdfc", paper_bgcolor="#fbfdfc", legend_title_text="", font_family="DM Sans", xaxis_title="Turbidity (NTU)", yaxis_title="E. coli (CFU/100ml)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Physicochemical vs. microbial contamination, by risk band")
else:
    st.info("Not enough paired turbidity and E. coli values to chart contamination.")

st.markdown('<div class="section-kicker">03 · Source records</div><h2>Explore the records</h2>', unsafe_allow_html=True)
district_summary = (
    filtered.assign(District=filtered["District"].map(clean_label))
    .groupby("District")
    .agg(Records=("Study_ID", "count"), **{"Mean risk score": ("Risk_Score", "mean")}, **{"Safe records": ("Overall_Safe", lambda values: int((values == "Yes").sum()))})
    .reset_index()
    .sort_values("Mean risk score", ascending=False)
)
district_summary["Mean risk score"] = district_summary["Mean risk score"].round(1)
st.markdown("#### District summary table")
st.dataframe(district_summary, use_container_width=True, hide_index=True, height=240)

display_columns = ["Study_ID", "Year", "District", "Location", "Water_Source_Type", "Risk_Band", "Risk_Score", "WHO_pH_Pass", "WHO_Turbidity_Pass", "WHO_Ecoli_Pass", "Notes"]
table = filtered[display_columns].copy()
table.columns = [column.replace("_", " ") for column in table.columns]
st.markdown("#### Full record table")
st.dataframe(table, use_container_width=True, hide_index=True, height=410)

st.markdown('<div class="section-kicker">04 · Explainable AI lab</div><h2>Compare models and explanations</h2>', unsafe_allow_html=True)
st.markdown(
    '<div class="callout"><strong>Exploratory modeling only.</strong> The compilation contains 25 aggregated study records and just 3 high-risk records. '
    'These results compare methods and generate hypotheses; they are not a validated water-safety predictor.</div>',
    unsafe_allow_html=True,
)
if len(data) >= 6 and int((data["Risk_Score"] >= 70).sum()) >= 3:
    comparison, importance, predictions, model_features = run_model_lab(data)
    model_names = comparison["Model"].tolist()
    metric_labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "precision": "Precision",
        "recall": "Recall / sensitivity",
        "specificity": "Specificity",
        "f1": "F1 score",
        "roc_auc": "ROC AUC",
        "pr_auc": "PR AUC",
        "mcc": "Matthews correlation",
    }
    selected_metric = st.selectbox("Performance metric", list(metric_labels), format_func=metric_labels.get, index=1)
    selected_model = st.selectbox("Model for explanation", model_names, index=2)
    model_col, explanation_col = st.columns([1.35, 1])
    with model_col:
        chart_data = comparison[["Model", selected_metric]].rename(columns={selected_metric: "Score"}).sort_values("Score")
        score_range = [-1, 1] if selected_metric == "mcc" else [0, 1]
        fig = px.bar(chart_data, x="Score", y="Model", orientation="h", color="Score", range_x=score_range, color_continuous_scale=["#d8f1ec", "#087f73"])
        fig.update_layout(height=370, margin=dict(l=0, r=0, t=15, b=0), coloraxis_showscale=False, plot_bgcolor="#fbfdfc", paper_bgcolor="#fbfdfc", font_family="DM Sans", xaxis_title=f"3-fold mean {metric_labels[selected_metric]}", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with explanation_col:
        st.markdown('<div class="panel-note"><h3>Model comparison</h3>', unsafe_allow_html=True)
        st.write("The scorecard compares linear, tree, distance-based, kernel, and probabilistic classifiers using the same folds. No single metric is sufficient: recall captures missed high-risk records, specificity captures false alarms, and PR AUC is useful when high-risk labels are rare.")
        st.markdown(f'<div class="small-note">Features used: {", ".join(model_features)}. Missing values are median-imputed with missingness indicators.</div></div>', unsafe_allow_html=True)
    model_families = {
        "Logistic regression": "Linear",
        "Decision tree": "Tree",
        "Random forest": "Ensemble",
        "Extra trees": "Ensemble",
        "Gradient boosting": "Boosting",
        "AdaBoost": "Boosting",
        "Histogram gradient boosting": "Boosting",
        "K-nearest neighbors": "Distance",
        "Support vector machine": "Kernel",
        "Linear discriminant analysis": "Discriminant",
        "Gaussian Naive Bayes": "Probabilistic",
    }
    methods_by_model = importance.groupby("Model")["Method"].apply(lambda values: ", ".join(values.drop_duplicates())).to_dict()
    scorecard = comparison.copy()
    scorecard["Rank"] = scorecard[selected_metric].rank(method="min", ascending=False).astype(int)
    scorecard["Family"] = scorecard["Model"].map(model_families)
    scorecard["Explanation methods"] = scorecard["Model"].map(methods_by_model)
    scorecard["Selected metric"] = scorecard[selected_metric].map(lambda value: f"{value:.1%}")
    scorecard["Metric variability"] = scorecard[f"{selected_metric} std"].map(lambda value: f"± {value:.1%}")
    for metric in metric_labels:
        scorecard[metric_labels[metric]] = scorecard[metric].map(lambda value: f"{value:.1%}")
    scorecard = scorecard[["Rank", "Model", "Family", "Selected metric", "Metric variability", *metric_labels.values(), "Explanation methods"]].sort_values(["Rank", "Model"])
    st.markdown("#### Model scorecard")
    st.dataframe(scorecard, use_container_width=True, hide_index=True, height=410)

    st.markdown("#### Out-of-fold validation: confusion matrix and ROC curve")
    model_predictions = predictions[predictions["Model"] == selected_model]
    observed_labels = model_predictions["Observed"].to_numpy()
    oof_probability = model_predictions["OOF probability"].to_numpy()
    oof_predicted = (oof_probability >= 0.5).astype(int)
    confusion_col, roc_col = st.columns(2)
    with confusion_col:
        matrix = confusion_matrix(observed_labels, oof_predicted, labels=[0, 1])
        confusion_table = pd.DataFrame(matrix, index=["Actual: not high-risk", "Actual: high-risk"], columns=["Predicted: not high-risk", "Predicted: high-risk"]).reset_index(names="")
        st.dataframe(confusion_table, use_container_width=True, hide_index=True, height=140)
        st.caption(f"{selected_model} · out-of-fold predictions at a 0.5 probability threshold")
    with roc_col:
        if len(np.unique(observed_labels)) == 2:
            false_positive_rate, true_positive_rate, _ = roc_curve(observed_labels, oof_probability)
            roc_data = pd.DataFrame({"False positive rate": false_positive_rate, "True positive rate": true_positive_rate})
            fig = px.line(roc_data, x="False positive rate", y="True positive rate")
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="#c7d3ce", dash="dash"))
            fig.update_traces(line_color="#087f73")
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="#fbfdfc", paper_bgcolor="#fbfdfc", font_family="DM Sans")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ROC curve needs both classes present in the current selection.")

    available_methods = importance[importance["Model"] == selected_model]["Method"].drop_duplicates().tolist()
    selected_method = st.selectbox("Explanation method", available_methods)
    importance_view = importance[(importance["Model"] == selected_model) & (importance["Method"] == selected_method)].sort_values("Importance", ascending=True)
    importance_col, local_col = st.columns([1.35, 1])
    with importance_col:
        fig = px.bar(importance_view, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale=["#d8f1ec", "#087f73"])
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=15, b=0), coloraxis_showscale=False, plot_bgcolor="#fbfdfc", paper_bgcolor="#fbfdfc", font_family="DM Sans", xaxis_title=selected_method, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with local_col:
        st.markdown('<div class="panel-note"><h3>Local case explanation</h3>', unsafe_allow_html=True)
        selected_study = st.selectbox("Study record", predictions["Study_ID"].tolist())
        selected_prediction = predictions[(predictions["Model"] == selected_model) & (predictions["Study_ID"] == selected_study)].iloc[0]
        probability = float(selected_prediction["Predicted probability"])
        observed = "High risk" if int(selected_prediction["Observed"]) else "Not high risk"
        st.metric("Predicted high-risk probability", f"{probability:.0%}")
        st.write(f"Observed label: **{observed}**. This probability is fitted on the complete small dataset and should be treated as a case-study explanation, not a validated forecast.")
        st.markdown(f'<div class="small-note">Global explanation: {selected_method}. Local explanation: selected-record probability and observed label.</div></div>', unsafe_allow_html=True)
else:
    st.warning("Not enough labeled records for the exploratory model lab.")

with st.expander("Method and data dictionary"):
    st.write("The dashboard uses the provided multi-study compilation. Risk Score is an interpretable heuristic: pH failure = 25, turbidity failure = 25, E. coli failure = 40, and an overall unsafe judgement = 10 points, capped at 100.")
    st.write("WHO reference points in the supplied dictionary include pH 6.5–8.5, turbidity below 5 NTU, and zero E. coli. Aggregated values, ranges, and narrative findings are kept as reported rather than imputed.")
    st.write("Source: `Datasets/gb_water_quality_2000_2026.csv` and `Datasets/data_dictionary.csv`. Cite the original study listed in `Authors_Source` for publication use.")

st.markdown('<div class="footer">© 2026 Mejbah Ahammad | Lead AI Instructor &amp; Research Scientist Portfolio</div>', unsafe_allow_html=True)