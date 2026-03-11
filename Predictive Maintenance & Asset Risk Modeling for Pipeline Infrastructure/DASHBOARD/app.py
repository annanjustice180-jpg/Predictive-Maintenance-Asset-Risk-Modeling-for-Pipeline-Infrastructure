import pandas as pd
import numpy as np
import streamlit as st


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "Pipeline Predictive Maintenance Dashboard"
DEFAULT_CSV = "risk_snapshot.csv"

# These thresholds match the operational tuning you used earlier
DEFAULT_IMMINENT_THR = 0.10
DEFAULT_EARLY_THR = 0.05

# Metrics from your model evaluation (hard-coded for credibility & consistency)
IMMINENT_METRICS = {
    "Recall": "91.7%",
    "Precision": "73.3%",
    "PR-AUC": "0.82",
    "Inspection rate @ 0.10": "0.57%",
}

EARLY_WARNING_METRICS = {
    "PR-AUC": "0.20 (baseline ≈ 0.0017)",
    "Avg lead time": "7.6 months",
    "Median lead time": "8 months",
    "Inspection rate @ 0.05": "0.49%",
}


# ============================================================
# HELPERS
# ============================================================
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic required fields for a useful dashboard
    if "Asset_ID" not in df.columns:
        raise ValueError("CSV must include 'Asset_ID'.")

    # Normalize common text columns (safe)
    for col in ["Risk_Tier", "Material", "Grade"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Enforce numeric probability columns if present
    for col in ["Imminent_Failure_Prob", "Early_Warning_Prob"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Month is optional, but improves snapshot logic
    if "Month" in df.columns:
        df["Month"] = pd.to_numeric(df["Month"], errors="coerce")

    return df


def compute_risk_tier(df: pd.DataFrame, imminent_thr: float, early_thr: float) -> pd.Series:
    """
    Tier logic:
      - HIGH: Imminent prob >= imminent_thr
      - MEDIUM: Early warning prob >= early_thr
      - LOW: otherwise
    If probability columns are missing, returns "UNSPECIFIED".
    """
    if "Imminent_Failure_Prob" not in df.columns or "Early_Warning_Prob" not in df.columns:
        return pd.Series(["UNSPECIFIED"] * len(df), index=df.index)

    imminent = df["Imminent_Failure_Prob"].fillna(0.0)
    early = df["Early_Warning_Prob"].fillna(0.0)

    tier = np.where(
        imminent >= imminent_thr,
        "HIGH RISK (Imminent)",
        np.where(early >= early_thr, "MEDIUM RISK (Early Trend)", "LOW RISK"),
    )
    return pd.Series(tier, index=df.index)


def latest_per_asset(df: pd.DataFrame) -> pd.DataFrame:
    """
    If Month exists, keep the latest record per Asset_ID.
    Otherwise, keep the first occurrence per Asset_ID.
    """
    if "Month" in df.columns and df["Month"].notna().any():
        return (
            df.sort_values(["Asset_ID", "Month"])
              .groupby("Asset_ID", as_index=False)
              .tail(1)
              .reset_index(drop=True)
        )
    return df.drop_duplicates(subset=["Asset_ID"]).reset_index(drop=True)


def histogram_df(series: pd.Series, bins: int = 20) -> pd.DataFrame:
    """
    Returns a dataframe with string bin labels to avoid Altair schema issues.
    """
    s = series.dropna().astype(float)
    if s.empty:
        return pd.DataFrame({"Bin": [], "Count": []})

    cut = pd.cut(s, bins=bins)
    counts = cut.value_counts().sort_index()

    out = counts.reset_index()
    out.columns = ["Bin", "Count"]
    out["Bin"] = out["Bin"].astype(str)
    return out


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Inspection prioritization using imminent failure risk and early warning trend signals.")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Data")
csv_path = st.sidebar.text_input("Risk snapshot CSV", value=DEFAULT_CSV)

df_raw = load_csv(csv_path)

st.sidebar.divider()
st.sidebar.header("Controls")

imminent_thr = st.sidebar.slider(
    "Imminent threshold",
    min_value=0.0, max_value=1.0, value=float(DEFAULT_IMMINENT_THR), step=0.01
)
early_thr = st.sidebar.slider(
    "Early warning threshold",
    min_value=0.0, max_value=1.0, value=float(DEFAULT_EARLY_THR), step=0.01
)

use_latest = st.sidebar.checkbox("Use latest record per asset", value=True)
recompute_tiers = st.sidebar.checkbox("Compute Risk Tier from thresholds", value=("Risk_Tier" not in df_raw.columns))

# Prepare working data
df = df_raw.copy()

if recompute_tiers:
    df["Risk_Tier"] = compute_risk_tier(df, imminent_thr, early_thr)
elif "Risk_Tier" not in df.columns:
    df["Risk_Tier"] = "UNSPECIFIED"

if use_latest:
    df = latest_per_asset(df)

# Tier filter
tiers = sorted(df["Risk_Tier"].dropna().unique().tolist())
selected_tiers = st.sidebar.multiselect("Filter by Risk Tier", options=tiers, default=tiers)
if selected_tiers:
    df_view = df[df["Risk_Tier"].isin(selected_tiers)].copy()
else:
    df_view = df.copy()

# Sorting
sort_options = [c for c in ["Imminent_Failure_Prob", "Early_Warning_Prob", "Asset_ID"] if c in df_view.columns]
sort_by = st.sidebar.selectbox("Sort by", options=sort_options if sort_options else ["Asset_ID"])
sort_asc = st.sidebar.checkbox("Ascending", value=False)

if sort_by in df_view.columns:
    df_view = df_view.sort_values(sort_by, ascending=sort_asc).reset_index(drop=True)

# ----------------------------
# KPI Row
# ----------------------------
k1, k2, k3, k4 = st.columns(4)

assets = df_view["Asset_ID"].nunique()
k1.metric("Assets in view", f"{assets:,}")

if "Imminent_Failure_Prob" in df_view.columns:
    k2.metric("Avg imminent risk", f"{df_view['Imminent_Failure_Prob'].mean():.3f}")
    k3.metric("Max imminent risk", f"{df_view['Imminent_Failure_Prob'].max():.3f}")
else:
    k2.metric("Avg imminent risk", "N/A")
    k3.metric("Max imminent risk", "N/A")

high_count = int((df_view["Risk_Tier"] == "HIGH RISK (Imminent)").sum())
k4.metric("High-risk assets", f"{high_count:,}")

st.divider()

# ----------------------------
# Risk Tier + Inspection Load
# ----------------------------
left, right = st.columns([1, 2])

with left:
    st.subheader("Risk Tier Distribution")
    tier_counts = df_view["Risk_Tier"].value_counts()
    st.bar_chart(tier_counts)

    st.subheader("Inspection Load")
    if "Imminent_Failure_Prob" in df_view.columns:
        immin_rate = (df_view["Imminent_Failure_Prob"].fillna(0.0) >= imminent_thr).mean() * 100
        st.write(f"Imminent alerts (≥ {imminent_thr:.2f}): **{immin_rate:.3f}%**")
    else:
        st.write("Imminent alerts: N/A")

    if "Early_Warning_Prob" in df_view.columns:
        early_rate = (df_view["Early_Warning_Prob"].fillna(0.0) >= early_thr).mean() * 100
        st.write(f"Early warnings (≥ {early_thr:.2f}): **{early_rate:.3f}%**")
    else:
        st.write("Early warnings: N/A")

with right:
    st.subheader("Top Assets by Risk")
    top_n = st.slider("Top N", min_value=5, max_value=50, value=20, step=1)

    sort_cols = []
    if "Imminent_Failure_Prob" in df_view.columns:
        sort_cols.append("Imminent_Failure_Prob")
    if "Early_Warning_Prob" in df_view.columns:
        sort_cols.append("Early_Warning_Prob")
    if not sort_cols:
        sort_cols = ["Asset_ID"]

    top = df_view.sort_values(sort_cols, ascending=False).head(top_n).copy()

    show_cols = [c for c in [
        "Asset_ID", "Month", "Risk_Tier",
        "Imminent_Failure_Prob", "Early_Warning_Prob",
        "Material", "Grade",
        "Pipe_Size_mm", "Thickness_mm",
        "Max_Pressure_psi", "Temperature_C", "Corrosion_Impact_Percent"
    ] if c in top.columns]

    st.dataframe(top[show_cols], use_container_width=True)

st.divider()

# ----------------------------
# Distributions (fixed)
# ----------------------------
d1, d2 = st.columns(2)

with d1:
    st.subheader("Imminent Risk Distribution")
    if "Imminent_Failure_Prob" in df_view.columns:
        h = histogram_df(df_view["Imminent_Failure_Prob"], bins=20)
        if len(h) == 0:
            st.info("No values available.")
        else:
            st.bar_chart(h.set_index("Bin"))
    else:
        st.info("Imminent_Failure_Prob not available in the CSV.")

with d2:
    st.subheader("Early Warning Distribution")
    if "Early_Warning_Prob" in df_view.columns:
        h = histogram_df(df_view["Early_Warning_Prob"], bins=20)
        if len(h) == 0:
            st.info("No values available.")
        else:
            st.bar_chart(h.set_index("Bin"))
    else:
        st.info("Early_Warning_Prob not available in the CSV.")

st.divider()

# ----------------------------
# Table + Download
# ----------------------------
st.subheader("Risk Snapshot Table")
st.dataframe(df_view, use_container_width=True, height=420)

csv_bytes = df_view.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download filtered snapshot (CSV)",
    data=csv_bytes,
    file_name="risk_snapshot_filtered.csv",
    mime="text/csv",
)

# ----------------------------
# Executive Report (metrics-backed)
# ----------------------------
st.divider()
st.header("Executive Report")

cA, cB = st.columns(2)

with cA:
    st.markdown("### Imminent Failure Model (12-month horizon)")
    for k, v in IMMINENT_METRICS.items():
        st.write(f"- **{k}**: {v}")

with cB:
    st.markdown("### Early Warning Model (trend-based)")
    for k, v in EARLY_WARNING_METRICS.items():
        st.write(f"- **{k}**: {v}")

st.markdown("""
### Operational Interpretation
- **HIGH RISK (Imminent)**: prioritize immediate inspection / intervention.
- **MEDIUM RISK (Early Trend)**: schedule targeted monitoring and maintenance planning.
- **LOW RISK**: routine monitoring.

### Notes
- Thresholds can be adjusted in the sidebar to match inspection capacity and risk tolerance.
- Recommended validation approach: asset-level splits and avoidance of post-failure data in training.
""")
