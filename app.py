"""
GP Practice List Size Dashboard
================================
Tracks growth/shrinkage of GP Practices in Northern Ireland
using quarterly CSV files from Open Data NI.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import io
import re
from datetime import datetime

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GP Practice List Size Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h2 { font-size: 2rem; margin: 0; }
    .metric-card p  { font-size: 0.9rem; margin: 4px 0 0; opacity: 0.85; }

    .metric-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-red {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .metric-blue {
        background: linear-gradient(135deg, #2980b9 0%, #6dd5fa 100%);
    }
    .metric-purple {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2c3e50;
        border-left: 4px solid #667eea;
        padding-left: 10px;
        margin: 20px 0 10px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    div[data-testid="stSidebarContent"] { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper: parse quarter date from filename
# ─────────────────────────────────────────────
MONTH_MAP = {
    "january": 1, "jan": 1,
    "april": 4, "apr": 4,
    "july": 7, "jul": 7,
    "october": 10, "oct": 10,
}

def parse_quarter_from_filename(filename: str):
    """
    Attempt to extract a quarter date (datetime) from a CSV filename.
    Handles patterns like:
      GP_Practice_July_2024.csv
      gp-practice-list-sizes-october-2023.csv
      GPPracticeListSize_2024Q1.csv   → fallback: None
    Returns a datetime or None.
    """
    name = os.path.basename(filename).lower()
    # Try month + year pattern
    for month_name, month_num in MONTH_MAP.items():
        pattern = rf"{month_name}[\s_\-]*(\d{{4}})"
        match = re.search(pattern, name)
        if match:
            year = int(match.group(1))
            return datetime(year, month_num, 1)
    # Try year + Q pattern  e.g. 2024q1, 2024_q2
    match = re.search(r"(\d{4})[\s_\-]*q(\d)", name)
    if match:
        year = int(match.group(1))
        q = int(match.group(2))
        month = (q - 1) * 3 + 1
        return datetime(year, month, 1)
    return None


def quarter_label(dt: datetime) -> str:
    """Format datetime as 'Q1 2024' style label."""
    q = (dt.month - 1) // 3 + 1
    return f"Q{q} {dt.year}"


# ─────────────────────────────────────────────
# Helper: detect column names flexibly
# ─────────────────────────────────────────────
def detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name (case-insensitive) that matches a candidate."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")


def read_csv_robust(filepath: str) -> pd.DataFrame:
    """Try multiple encodings to read a CSV file, stripping trailing empty columns."""
    for encoding in ["utf-8", "latin-1", "cp1252", "utf-8-sig"]:
        try:
            df = pd.read_csv(filepath, encoding=encoding, dtype=str)
            df.columns = [c.strip() for c in df.columns]
            # Drop unnamed/empty trailing columns (caused by trailing commas in CSV)
            df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]
            return df
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError(f"Could not read file with any supported encoding: {filepath}")


def parse_one_file(filepath: str) -> tuple:
    """
    Parse a single CSV file into a clean DataFrame.
    Returns (clean_df, error_string).
    One of them will be None.
    """
    filename = os.path.basename(filepath)
    try:
        df = read_csv_robust(filepath)

        # Show actual columns in debug output (stored, shown later)
        actual_cols = list(df.columns)

        code_col = detect_column(df, [
            "pracno", "prac_no", "prac no",
            "practice code", "practicecode", "practice_code",
            "gp code", "gpcode", "code", "prac_code",
            "practice id", "practiceid", "id"
        ])
        name_col = detect_column(df, [
            "practicename", "practice name", "practice_name",
            "gp practice name", "name", "gp_name", "practice"
        ])
        size_col = detect_column(df, [
            "registered_patients", "registered patients",
            "list size", "listsize", "list_size",
            "patients", "no of patients",
            "number of patients", "total patients", "total",
            "no. of patients", "numberofpatients"
        ])
        lcg_col = detect_column(df, [
            "lcg", "local commissioning group", "lcg name",
            "lcgname", "trust", "health trust", "hsct",
            "area", "hscni", "hsc trust"
        ])

        if not code_col or not size_col:
            return None, (
                f"**{filename}**\n"
                f"- Practice Code column found: `{code_col or 'NOT FOUND'}`\n"
                f"- List Size column found: `{size_col or 'NOT FOUND'}`\n"
                f"- Actual columns in file: `{actual_cols}`"
            )

        quarter_dt = parse_quarter_from_filename(filename)

        date_col = detect_column(df, ["date", "quarter", "period", "quarter date", "quarterdate"])
        if quarter_dt is None and date_col:
            sample = df[date_col].dropna().iloc[0] if not df[date_col].dropna().empty else None
            if sample:
                try:
                    quarter_dt = pd.to_datetime(sample, dayfirst=True).to_pydatetime()
                except Exception:
                    pass

        addr1_col = detect_column(df, ["address1", "address 1", "addr1", "practice address"])

        clean = pd.DataFrame()
        clean["PracticeCode"] = df[code_col].str.strip()
        clean["PracticeName"] = df[name_col].str.strip() if name_col else df[code_col].str.strip()
        clean["PracticeAddress"] = df[addr1_col].str.strip().str.title() if addr1_col else ""
        clean["ListSize"] = pd.to_numeric(
            df[size_col].str.replace(",", "").str.replace(" ", "").str.strip(),
            errors="coerce"
        )
        clean["LCG"] = df[lcg_col].str.strip() if lcg_col else "Unknown"
        clean["QuarterDate"] = quarter_dt
        clean["SourceFile"] = filename

        return clean, None

    except Exception as e:
        return None, f"**{filename}** — unexpected error: {e}"


@st.cache_data(show_spinner=False)
def load_from_folder(folder: str) -> tuple:
    """
    Load and combine all CSV files found in the local data/ folder.
    Returns (combined_df, list_of_error_strings, list_of_all_column_snapshots).
    """
    csv_files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".csv")
    ])
    if not csv_files:
        return pd.DataFrame(), [], []

    frames = []
    errors = []
    col_snapshots = []  # For diagnostics: [(filename, [cols])]

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        try:
            df = read_csv_robust(filepath)
            col_snapshots.append((filename, list(df.columns)))
        except Exception:
            col_snapshots.append((filename, ["UNREADABLE"]))

        clean, error = parse_one_file(filepath)
        if error:
            errors.append(error)
        else:
            frames.append(clean)

    if not frames:
        return pd.DataFrame(), errors, col_snapshots

    frames = [f.dropna(axis=1, how="all") for f in frames]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["ListSize"])
    combined["ListSize"] = combined["ListSize"].astype(int)

    # Identify and warn about files where quarter date could not be parsed
    undated_files = (
        combined[combined["QuarterDate"].isna()]["SourceFile"]
        .unique().tolist()
    )
    if undated_files:
        errors.append(
            "⚠️ **Quarter date not detected** for the following file(s) — "
            "these rows have been excluded from the dashboard. "
            "Rename the files to include the month and year "
            "(e.g. `gp-practice-reference-file-january-2023.csv`) to include them:\n"
            + "\n".join(f"- `{f}`" for f in undated_files)
        )

    # Drop rows with no quarter date — exclude from all charts
    combined = combined.dropna(subset=["QuarterDate"])

    combined = combined.sort_values(["PracticeCode", "QuarterDate"]).reset_index(drop=True)
    combined["QuarterLabel"] = combined["QuarterDate"].apply(quarter_label)
    return combined, errors, col_snapshots


@st.cache_data(show_spinner=False)
def load_and_combine(uploaded_files) -> pd.DataFrame:
    """
    Load multiple uploaded CSV files, tag each with its quarter, and combine.
    Returns a cleaned, sorted DataFrame.
    """
    frames = []
    errors = []

    for uf in uploaded_files:
        try:
            df = pd.read_csv(uf, encoding="utf-8", dtype=str)
            # Strip whitespace from column names
            df.columns = [c.strip() for c in df.columns]

            # Detect key columns
            code_col = detect_column(df, [
                "pracno", "prac_no", "prac no",
                "practice code", "practicecode", "practice_code",
                "gp code", "gpcode", "code", "prac_code"
            ])
            name_col = detect_column(df, [
                "practicename", "practice name", "practice_name",
                "gp practice name", "name", "gp_name"
            ])
            size_col = detect_column(df, [
                "registered_patients", "registered patients",
                "list size", "listsize", "list_size",
                "patients", "no of patients",
                "number of patients", "total patients"
            ])
            lcg_col = detect_column(df, [
                "lcg", "local commissioning group", "lcg name",
                "lcgname", "trust", "health trust", "hsct"
            ])

            if not code_col or not size_col:
                errors.append(f"⚠️ **{uf.name}** — could not identify Practice Code or List Size columns. Columns found: {list(df.columns)}")
                continue

            # Parse quarter date from filename
            quarter_dt = parse_quarter_from_filename(uf.name)

            # Also check if there's a date/quarter column inside the file
            date_col = detect_column(df, ["date", "quarter", "period", "quarter date"])
            if quarter_dt is None and date_col:
                sample = df[date_col].dropna().iloc[0] if not df[date_col].dropna().empty else None
                if sample:
                    try:
                        quarter_dt = pd.to_datetime(sample, dayfirst=True).to_pydatetime()
                    except Exception:
                        pass

            # Build clean frame
            addr1_col = detect_column(df, ["address1", "address 1", "addr1", "practice address"])
            clean = pd.DataFrame()
            clean["PracticeCode"] = df[code_col].str.strip()
            clean["PracticeName"] = df[name_col].str.strip() if name_col else clean["PracticeCode"]
            clean["PracticeAddress"] = df[addr1_col].str.strip().str.title() if addr1_col else ""
            clean["ListSize"] = pd.to_numeric(df[size_col].str.replace(",", "").str.strip(), errors="coerce")
            if lcg_col:
                clean["LCG"] = df[lcg_col].str.strip()
            else:
                clean["LCG"] = "Unknown"

            # Quarter metadata
            clean["QuarterDate"] = quarter_dt
            clean["SourceFile"] = uf.name

            frames.append(clean)

        except Exception as e:
            errors.append(f"⚠️ **{uf.name}** — error reading file: {e}")

    if errors:
        for err in errors:
            st.sidebar.warning(err)

    if not frames:
        return pd.DataFrame()

    frames = [f.dropna(axis=1, how="all") for f in frames]
    combined = pd.concat(frames, ignore_index=True)

    # Drop rows with no list size
    combined = combined.dropna(subset=["ListSize"])
    combined["ListSize"] = combined["ListSize"].astype(int)

    # Sort
    combined = combined.sort_values(["PracticeCode", "QuarterDate"]).reset_index(drop=True)

    # Add quarter label
    combined["QuarterLabel"] = combined["QuarterDate"].apply(
        lambda d: quarter_label(d) if pd.notna(d) else combined["SourceFile"]
    )

    return combined


@st.cache_data(show_spinner=False)
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add quarter-on-quarter change columns per practice."""
    df = df.copy().sort_values(["PracticeCode", "QuarterDate"])
    df["PrevListSize"] = df.groupby("PracticeCode")["ListSize"].shift(1)
    df["Change"]       = df["ListSize"] - df["PrevListSize"]
    df["ChangePct"]    = (df["Change"] / df["PrevListSize"] * 100).round(2)
    return df


def generate_sample_data() -> pd.DataFrame:
    """Create synthetic data so users can try the dashboard without uploading files."""
    np.random.seed(42)
    practices = {
        "P001": ("Antrim Health Centre", "Northern"),
        "P002": ("Ballymena Medical Practice", "Northern"),
        "P003": ("Belfast City Practice", "Belfast"),
        "P004": ("Derry Central GP", "Western"),
        "P005": ("Lisburn Road Surgery", "South Eastern"),
        "P006": ("Newry Medical Centre", "Southern"),
        "P007": ("Omagh Health Practice", "Western"),
        "P008": ("Bangor Surgery", "South Eastern"),
        "P009": ("Coleraine Medical", "Northern"),
        "P010": ("Armagh Primary Care", "Southern"),
        "P011": ("Downpatrick Practice", "South Eastern"),
        "P012": ("Enniskillen Surgery", "Western"),
    }
    quarters = [
        datetime(2022, 1, 1), datetime(2022, 4, 1),
        datetime(2022, 7, 1), datetime(2022, 10, 1),
        datetime(2023, 1, 1), datetime(2023, 4, 1),
        datetime(2023, 7, 1), datetime(2023, 10, 1),
        datetime(2024, 1, 1), datetime(2024, 4, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
    ]

    rows = []
    for code, (name, lcg) in practices.items():
        base = np.random.randint(2000, 12000)
        trend = np.random.choice([-1, 0, 1, 2], p=[0.1, 0.2, 0.5, 0.2])
        size = base
        for q in quarters:
            noise = np.random.randint(-150, 250)
            size = max(500, size + trend * 80 + noise)
            rows.append({
                "PracticeCode": code,
                "PracticeName": name,
                "LCG": lcg,
                "ListSize": int(size),
                "QuarterDate": q,
                "QuarterLabel": quarter_label(q),
                "SourceFile": f"sample_{q.strftime('%b_%Y')}.csv",
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Metric card HTML helper
# ─────────────────────────────────────────────
def metric_card(title, value, subtitle="", colour="purple"):
    st.markdown(f"""
    <div class="metric-card metric-{colour}">
        <h2>{value}</h2>
        <p><strong>{title}</strong></p>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

# Check what's in the data/ folder before rendering sidebar
folder_csv_files = []
if os.path.isdir(DATA_FOLDER):
    folder_csv_files = sorted([f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".csv")])

with st.sidebar:
    st.markdown("## 🏥 GP Dashboard")
    st.markdown("---")

    st.markdown("### 📂 Data Source")

    # Show data/ folder status
    if folder_csv_files:
        st.success(f"✅ **data/ folder** — {len(folder_csv_files)} CSV file(s) found")
        with st.expander("View files in data/ folder"):
            for f in folder_csv_files:
                st.markdown(f"- `{f}`")
        st.markdown("Files are loaded automatically from your `data/` folder.")
    else:
        st.info(
            "No CSVs found in `data/` folder. Either add files there, "
            "or upload them manually below."
        )

    st.markdown("**Or upload files manually:**")
    uploaded_files = st.file_uploader(
        "Select CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Download quarterly files from Open Data NI and upload them all at once."
    )

    # Demo mode only if nothing else is available
    has_real_data = bool(folder_csv_files) or bool(uploaded_files)
    use_sample = st.checkbox("Use sample data (demo mode)", value=not has_real_data)

    st.markdown("---")
    st.markdown("### ⚙️ Filters")

    # Placeholders — populated after data loads
    lcg_filter = st.empty()
    threshold_filter = st.empty()

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "Data sourced from **Open Data NI** — a quarterly reference file of "
        "active GP Practices and their registered patient list sizes in Northern Ireland."
    )


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
load_errors = []
col_snapshots = []

with st.spinner("Loading data..."):
    if use_sample:
        raw_df = generate_sample_data()
        st.info("🎭 **Demo mode** — showing synthetic sample data. Add real CSV files to your `data/` folder to analyse actual data.", icon="ℹ️")
    elif uploaded_files:
        raw_df = load_and_combine(uploaded_files)
        st.success(f"✅ Loaded {len(uploaded_files)} manually uploaded file(s).")
    elif folder_csv_files:
        raw_df, load_errors, col_snapshots = load_from_folder(DATA_FOLDER)
        if not raw_df.empty:
            quarters_loaded = raw_df["QuarterDate"].nunique()
            skipped = len(folder_csv_files) - quarters_loaded
            if skipped > 0:
                st.success(f"✅ Auto-loaded **{quarters_loaded} of {len(folder_csv_files)} file(s)** from `data/` folder.")
            else:
                st.success(f"✅ Auto-loaded **{len(folder_csv_files)} file(s)** from `data/` folder.")

    # Show any undated-file warnings in the sidebar
    if load_errors:
        with st.sidebar:
            st.markdown("---")
            st.markdown("### ⚠️ File Warnings")
            for err in load_errors:
                st.warning(err)
    else:
        raw_df = pd.DataFrame()

if raw_df.empty:
    st.error("❌ No data could be loaded. See the diagnostics below to fix the issue.")

    if col_snapshots:
        st.markdown("### 🔍 Column Diagnostics")
        st.markdown(
            "The app couldn't match the column names in your CSV files. "
            "Here are the **actual column names** found in each file — "
            "use these to identify what the correct names are:"
        )
        for fname, cols in col_snapshots[:3]:  # Show first 3 files
            with st.expander(f"📄 {fname}"):
                st.code(", ".join(cols))

        if load_errors:
            st.markdown("### ⚠️ Errors per file")
            for err in load_errors[:5]:
                st.warning(err)

        st.markdown("---")
        st.info(
            "**To fix this:** Copy one of the column names exactly from the diagnostics above "
            "and share it here so the app can be updated to recognise your file format."
        )

    st.stop()

df = compute_metrics(raw_df)


# ─────────────────────────────────────────────
# SIDEBAR FILTERS (now data is loaded)
# ─────────────────────────────────────────────
all_lcgs = sorted(df["LCG"].dropna().unique().tolist())
selected_lcgs = lcg_filter.multiselect(
    "Local Commissioning Group",
    options=all_lcgs,
    default=all_lcgs,
    help="Filter practices by LCG / Health Trust area"
)

change_threshold = threshold_filter.slider(
    "Growth alert threshold (%)",
    min_value=1, max_value=30, value=5,
    help="Flag practices that grew or shrank by more than this % in a quarter"
)

# Apply LCG filter
filtered_df = df[df["LCG"].isin(selected_lcgs)] if selected_lcgs else df

quarters_sorted = sorted(filtered_df["QuarterDate"].dropna().unique())
latest_quarter  = max(quarters_sorted) if quarters_sorted else None
prev_quarter    = quarters_sorted[-2] if len(quarters_sorted) >= 2 else None


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🏥 GP Practice List Size Dashboard")
st.markdown(
    f"**Northern Ireland** · Quarterly data · "
    f"{'Demo mode' if use_sample else (f'{len(uploaded_files)} file(s) uploaded' if uploaded_files else f'{len(folder_csv_files)} file(s) from data/ folder')} · "
    f"{len(quarters_sorted)} quarter(s) available"
)
st.markdown("---")


# ─────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────
latest_df = filtered_df[filtered_df["QuarterDate"] == latest_quarter] if latest_quarter else pd.DataFrame()
prev_df   = filtered_df[filtered_df["QuarterDate"] == prev_quarter]   if prev_quarter   else pd.DataFrame()

total_patients  = int(latest_df["ListSize"].sum()) if not latest_df.empty else 0
total_practices = int(latest_df["PracticeCode"].nunique()) if not latest_df.empty else 0

# Quarter-on-quarter total change
prev_total = int(prev_df["ListSize"].sum()) if not prev_df.empty else 0
qoq_change = total_patients - prev_total
qoq_pct    = round(qoq_change / prev_total * 100, 1) if prev_total else 0

growing = filtered_df[filtered_df["ChangePct"] > change_threshold]["PracticeCode"].nunique()
shrinking = filtered_df[filtered_df["ChangePct"] < -change_threshold]["PracticeCode"].nunique()

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Total Registered Patients", f"{total_patients:,}",
                f"Latest quarter: {quarter_label(latest_quarter) if latest_quarter else 'N/A'}", "blue")
with col2:
    metric_card("Active Practices", f"{total_practices:,}",
                f"Across {len(selected_lcgs)} LCG(s)", "purple")
with col3:
    arrow = "▲" if qoq_change >= 0 else "▼"
    metric_card("QoQ Patient Change", f"{arrow} {abs(qoq_change):,}",
                f"{qoq_pct:+.1f}% vs previous quarter",
                "green" if qoq_change >= 0 else "red")
with col4:
    metric_card("Practices Flagged", f"↑{growing} / ↓{shrinking}",
                f"Growing / shrinking >{change_threshold}% in any quarter", "purple")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overall Trends",
    "🏆 Top 10 by Year",
    "🏥 Practice Detail",
    "🗺️ LCG Breakdown",
    "🚨 Movers & Alerts",
    "📋 Raw Data",
])


# ══════════════════════════════════════════════
# TAB 1 — Overall Trends
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Total Registered Patients Over Time</div>', unsafe_allow_html=True)

    # Aggregate by quarter
    quarterly_totals = (
        filtered_df.groupby(["QuarterDate", "QuarterLabel"])
        .agg(TotalPatients=("ListSize", "sum"), ActivePractices=("PracticeCode", "nunique"))
        .reset_index()
        .sort_values("QuarterDate")
    )

    # Calculate a tight y-axis range so small changes are visible
    # (default zero-based axis makes ~2% variation look completely flat)
    _min_patients = quarterly_totals["TotalPatients"].min()
    _max_patients = quarterly_totals["TotalPatients"].max()
    _padding = (_max_patients - _min_patients) * 0.3 or _max_patients * 0.005
    _y_min = _min_patients - _padding
    _y_max = _max_patients + _padding

    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(
        go.Scatter(
            x=quarterly_totals["QuarterLabel"],
            y=quarterly_totals["TotalPatients"],
            name="Total Patients",
            fill="toself",
            line=dict(color="#667eea", width=3),
            fillcolor="rgba(102,126,234,0.15)",
        ),
        secondary_y=False,
    )
    fig1.add_trace(
        go.Scatter(
            x=quarterly_totals["QuarterLabel"],
            y=quarterly_totals["ActivePractices"],
            name="Active Practices",
            line=dict(color="#f39c12", width=2, dash="dot"),
            mode="lines+markers",
            marker=dict(size=7),
        ),
        secondary_y=True,
    )
    fig1.update_layout(
        height=400,
        xaxis_title="Quarter",
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig1.update_yaxes(
        title_text="Total Registered Patients",
        secondary_y=False,
        gridcolor="#f0f0f0",
        range=[_y_min, _y_max],
        tickformat=",d",
    )
    fig1.update_yaxes(title_text="Number of Active Practices", secondary_y=True)
    st.plotly_chart(fig1, width="stretch")

    # QoQ change bar chart
    st.markdown('<div class="section-header">Quarter-on-Quarter Patient Change</div>', unsafe_allow_html=True)
    quarterly_totals["QoQChange"] = quarterly_totals["TotalPatients"].diff()
    quarterly_totals["Colour"] = quarterly_totals["QoQChange"].apply(
        lambda x: "#27ae60" if x >= 0 else "#e74c3c"
    )

    fig2 = go.Figure(go.Bar(
        x=quarterly_totals["QuarterLabel"],
        y=quarterly_totals["QoQChange"],
        marker_color=quarterly_totals["Colour"],
        text=quarterly_totals["QoQChange"].apply(lambda x: f"{x:+,.0f}" if pd.notna(x) else ""),
        textposition="outside",
    ))
    fig2.update_layout(
        height=320,
        xaxis_title="Quarter",
        yaxis_title="Patient Change",
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(gridcolor="#f0f0f0"),
    )
    st.plotly_chart(fig2, width="stretch")


# ══════════════════════════════════════════════
# TAB 2 — Top 10 by Year
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Top 10 Practices by Registered Patients — Per Year</div>', unsafe_allow_html=True)

    # Use the last available quarter of each year as the year's representative snapshot
    top10_df = filtered_df.copy()
    top10_df["Year"] = top10_df["QuarterDate"].apply(lambda d: d.year)

    # Pick the latest quarter per year as the year snapshot
    latest_per_year = (
        top10_df.groupby("Year")["QuarterDate"].max().reset_index()
        .rename(columns={"QuarterDate": "LatestQuarter"})
    )
    snapshot_df = top10_df.merge(latest_per_year, left_on=["Year", "QuarterDate"],
                                  right_on=["Year", "LatestQuarter"], how="inner")

    available_years = sorted(snapshot_df["Year"].unique(), reverse=True)

    if not available_years:
        st.warning("No yearly data available.")
    else:
        # Year selector — default to most recent
        selected_year = st.selectbox(
            "Select Year",
            options=available_years,
            index=0,
            key="top10_year",
        )

        year_df = (
            snapshot_df[snapshot_df["Year"] == selected_year]
            .sort_values("ListSize", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        year_df.index = year_df.index + 1  # rank from 1

        # Label: "Known as" if available, else PracticeName
        year_df["Label"] = year_df.apply(
            lambda r: r["PracticeAddress"] if (
                "PracticeAddress" in r and r["PracticeAddress"] and str(r["PracticeAddress"]).strip()
            ) else r["PracticeName"],
            axis=1,
        )
        year_df["FullLabel"] = year_df.apply(
            lambda r: f"{r['Label']} ({r['PracticeCode']})", axis=1
        )

        # ── Horizontal bar chart ──
        fig_top = go.Figure(go.Bar(
            x=year_df["ListSize"],
            y=year_df["FullLabel"],
            orientation="h",
            marker=dict(
                color=year_df["ListSize"],
                colorscale="Blues",
                showscale=False,
            ),
            text=year_df["ListSize"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            customdata=year_df[["LCG", "PracticeName"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Registered Patients: %{x:,}<br>"
                "LCG: %{customdata[0]}<br>"
                "GP Partner: %{customdata[1]}<extra></extra>"
            ),
        ))

        # Tight x-axis so bars aren't squashed
        _x_max = year_df["ListSize"].max()
        fig_top.update_layout(
            height=420,
            xaxis=dict(
                title="Registered Patients",
                range=[0, _x_max * 1.15],
                gridcolor="#f0f0f0",
            ),
            yaxis=dict(
                autorange="reversed",  # rank 1 at the top
                tickfont=dict(size=12),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=10, r=80),
        )
        st.plotly_chart(fig_top, width="stretch")

        # ── Biggest net change for the selected year ──
        st.markdown('<div class="section-header">Biggest Net Change in Selected Year</div>', unsafe_allow_html=True)
        st.caption(
            f"Compares each practice's list size at the **start vs end** of {selected_year}. "
            "Shows the 10 biggest growers and 10 biggest shrinkers by absolute patient count."
        )

        # Get first and last quarter snapshots within the selected year
        year_quarters = sorted(top10_df[top10_df["Year"] == selected_year]["QuarterDate"].unique())
        first_q_year  = year_quarters[0]  if len(year_quarters) >= 1 else None
        last_q_year   = year_quarters[-1] if len(year_quarters) >= 1 else None

        if first_q_year and last_q_year and first_q_year != last_q_year:
            start_snap = top10_df[top10_df["QuarterDate"] == first_q_year][["PracticeCode", "PracticeName", "LCG", "ListSize"]].rename(columns={"ListSize": "StartSize"})
            end_snap   = top10_df[top10_df["QuarterDate"] == last_q_year][["PracticeCode", "ListSize"]].rename(columns={"ListSize": "EndSize"})
            net_df = start_snap.merge(end_snap, on="PracticeCode", how="inner")
            net_df["NetChange"]    = net_df["EndSize"] - net_df["StartSize"]
            net_df["NetChangePct"] = (net_df["NetChange"] / net_df["StartSize"] * 100).round(1)
            net_df["ShortName"]    = net_df["PracticeName"].str.slice(0, 35)

            top_growers_yr   = net_df.nlargest(10, "NetChange")
            top_shrinkers_yr = net_df.nsmallest(10, "NetChange")

            col_ng, col_ns = st.columns(2)

            with col_ng:
                st.markdown("##### 🟢 Top 10 Growers")
                fig_ng = go.Figure(go.Bar(
                    x=top_growers_yr["NetChange"],
                    y=top_growers_yr["ShortName"],
                    orientation="h",
                    marker_color="#27ae60",
                    text=top_growers_yr.apply(
                        lambda r: f"+{r['NetChange']:,} ({r['NetChangePct']:+.1f}%)", axis=1
                    ),
                    textposition="outside",
                    customdata=top_growers_yr[["StartSize", "EndSize", "LCG"]],
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Net Change: %{x:+,}<br>"
                        "Start: %{customdata[0]:,} → End: %{customdata[1]:,}<br>"
                        "LCG: %{customdata[2]}<extra></extra>"
                    ),
                ))
                fig_ng.update_layout(
                    height=400,
                    xaxis=dict(title="Net Patient Change", gridcolor="#f0f0f0",
                               range=[0, top_growers_yr["NetChange"].max() * 1.25]),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                    plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(l=10, r=80),
                )
                st.plotly_chart(fig_ng, width="stretch")

            with col_ns:
                st.markdown("##### 🔴 Top 10 Shrinkers")
                fig_ns = go.Figure(go.Bar(
                    x=top_shrinkers_yr["NetChange"],
                    y=top_shrinkers_yr["ShortName"],
                    orientation="h",
                    marker_color="#e74c3c",
                    text=top_shrinkers_yr.apply(
                        lambda r: f"{r['NetChange']:,} ({r['NetChangePct']:+.1f}%)", axis=1
                    ),
                    textposition="outside",
                    customdata=top_shrinkers_yr[["StartSize", "EndSize", "LCG"]],
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Net Change: %{x:,}<br>"
                        "Start: %{customdata[0]:,} → End: %{customdata[1]:,}<br>"
                        "LCG: %{customdata[2]}<extra></extra>"
                    ),
                ))
                fig_ns.update_layout(
                    height=400,
                    xaxis=dict(title="Net Patient Change", gridcolor="#f0f0f0",
                               range=[top_shrinkers_yr["NetChange"].min() * 1.25, 0]),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                    plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(l=10, r=80),
                )
                st.plotly_chart(fig_ns, width="stretch")

            # Combined waterfall-style dot plot for overall picture
            st.markdown('<div class="section-header">All Practices — Net Change Overview</div>', unsafe_allow_html=True)
            net_all = net_df.sort_values("NetChange", ascending=False).reset_index(drop=True)
            net_all["Colour"] = net_all["NetChange"].apply(lambda x: "#27ae60" if x >= 0 else "#e74c3c")

            fig_all = go.Figure(go.Bar(
                x=net_all.index,
                y=net_all["NetChange"],
                marker_color=net_all["Colour"],
                customdata=net_all[["PracticeName", "NetChange", "NetChangePct", "LCG"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Net Change: %{customdata[1]:+,}<br>"
                    "Change %: %{customdata[2]:+.1f}%<br>"
                    "LCG: %{customdata[3]}<extra></extra>"
                ),
            ))
            fig_all.add_hline(y=0, line_color="grey", line_width=1)
            fig_all.update_layout(
                height=320,
                xaxis=dict(title=f"All practices ranked by net change in {selected_year}", showticklabels=False),
                yaxis=dict(title="Net Patient Change", gridcolor="#f0f0f0"),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig_all, width="stretch")

        else:
            st.info(f"Only one quarter of data available for {selected_year} — net change requires at least two quarters in the year.")

        # ── Summary table ──
        st.markdown('<div class="section-header">Full Rankings Table</div>', unsafe_allow_html=True)

        table_df = year_df[["Label", "PracticeName", "PracticeCode", "LCG", "ListSize"]].copy()
        table_df.insert(0, "Rank", range(1, len(table_df) + 1))
        table_df.columns = ["Rank", "Practice Name", "GP Partner", "Code", "LCG", "Registered Patients"]
        table_df["Registered Patients"] = table_df["Registered Patients"].apply(lambda x: f"{x:,}")
        st.dataframe(table_df, width="stretch", hide_index=True)

        # ── Year-on-year comparison: same top 10 across all years ──
        st.markdown('<div class="section-header">How the Selected Year\'s Top 10 Compares Over Time</div>', unsafe_allow_html=True)
        top10_codes = year_df["PracticeCode"].tolist()

        comparison_df = (
            snapshot_df[snapshot_df["PracticeCode"].isin(top10_codes)]
            .sort_values("QuarterDate")
            .copy()
        )
        comparison_df["Label"] = comparison_df.apply(
            lambda r: r["PracticeAddress"] if (
                "PracticeAddress" in r and r["PracticeAddress"] and str(r["PracticeAddress"]).strip()
            ) else r["PracticeName"],
            axis=1,
        )

        fig_compare = px.line(
            comparison_df,
            x="Year",
            y="ListSize",
            color="Label",
            markers=True,
            labels={"ListSize": "Registered Patients", "Year": "Year", "Label": "Practice"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_compare.update_layout(
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(gridcolor="#f0f0f0", tickformat=",d"),
            xaxis=dict(tickmode="linear", dtick=1),
            legend=dict(orientation="h", y=-0.3, font=dict(size=11)),
            hovermode="x unified",
        )
        st.plotly_chart(fig_compare, width="stretch")


# ══════════════════════════════════════════════
# TAB 3 — Practice Detail
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Individual Practice Trend</div>', unsafe_allow_html=True)

    # Sort practice codes numerically where possible (avoids '10' < '2' string sort)
    def _sort_key(code):
        try:
            return (0, int(code))
        except (ValueError, TypeError):
            return (1, str(code))

    all_practices = sorted(filtered_df["PracticeCode"].unique().tolist(), key=_sort_key)

    # Use the most recent name for each practice (in case name changed over time)
    practice_names = (
        filtered_df.sort_values("QuarterDate")
        .groupby("PracticeCode")["PracticeName"]
        .last()
        .to_dict()
    )
    practice_options = [f"{c} — {practice_names.get(c, c)}" for c in all_practices]

    selected_practice_str = st.selectbox(
        "Select a GP Practice",
        options=practice_options,
        index=0,
    )
    selected_code = selected_practice_str.split(" — ")[0]

    practice_df = filtered_df[filtered_df["PracticeCode"] == selected_code].sort_values("QuarterDate")

    if not practice_df.empty:
        latest_size = int(practice_df["ListSize"].iloc[-1])
        first_size  = int(practice_df["ListSize"].iloc[0])
        overall_change = latest_size - first_size
        overall_pct    = round(overall_change / first_size * 100, 1) if first_size else 0

        # Derive info from practice_df AFTER filtering so it always matches the selection
        selected_name = practice_df["PracticeName"].iloc[-1]  # use latest entry for name
        selected_lcg  = practice_df["LCG"].iloc[-1]

        col_info = st.columns([2, 1])
        selected_address = practice_df["PracticeAddress"].iloc[-1] if "PracticeAddress" in practice_df.columns else ""

        with col_info[1]:
            st.markdown(f"**Practice:** {selected_name}")
            if selected_address:
                st.markdown(f"**Known as:** {selected_address}")
            st.markdown(f"**Code:** {selected_code}")
            st.markdown(f"**LCG:** {selected_lcg}")
            st.markdown(f"**Latest list size:** {latest_size:,}")
            st.markdown(f"**Overall change:** {overall_change:+,} ({overall_pct:+.1f}%)")

        # Line chart
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=practice_df["QuarterLabel"],
            y=practice_df["ListSize"],
            mode="lines+markers+text",
            line=dict(color="#667eea", width=3),
            marker=dict(size=9, color="#667eea"),
            text=practice_df["ListSize"].apply(lambda x: f"{x:,}"),
            textposition="top center",
            name="List Size",
        ))
        # Add trend line
        if len(practice_df) > 2:
            x_idx = np.arange(len(practice_df))
            z = np.polyfit(x_idx, practice_df["ListSize"], 1)
            trend_vals = np.polyval(z, x_idx)
            fig3.add_trace(go.Scatter(
                x=practice_df["QuarterLabel"],
                y=trend_vals,
                mode="lines",
                line=dict(color="#e74c3c", dash="dash", width=2),
                name="Trend",
            ))

        fig3.update_layout(
            height=380,
            xaxis_title="Quarter",
            yaxis_title="Registered Patients",
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(gridcolor="#f0f0f0"),
            legend=dict(orientation="h", y=1.1),
            hovermode="x unified",
        )
        st.plotly_chart(fig3, width="stretch")

        # QoQ change bar for this practice
        st.markdown('<div class="section-header">Quarter-on-Quarter Change</div>', unsafe_allow_html=True)

        # Warn if this practice is missing from some quarters (opened/closed/merged)
        total_quarters = filtered_df["QuarterDate"].nunique()
        practice_quarters = practice_df["QuarterDate"].nunique()
        if practice_quarters < total_quarters:
            st.info(
                f"ℹ️ This practice only appears in **{practice_quarters} of {total_quarters}** quarters — "
                "it may have opened, closed, or merged during this period."
            )

        change_df = practice_df.dropna(subset=["Change"])
        if not change_df.empty:
            fig4 = go.Figure(go.Bar(
                x=change_df["QuarterLabel"],
                y=change_df["Change"],
                marker_color=change_df["Change"].apply(lambda x: "#27ae60" if x >= 0 else "#e74c3c"),
                text=change_df["Change"].apply(lambda x: f"{x:+,.0f}"),
                textposition="outside",
                customdata=change_df["ChangePct"],
                hovertemplate="<b>%{x}</b><br>Change: %{y:+,.0f}<br>Change %: %{customdata:.1f}%<extra></extra>",
            ))
            fig4.update_layout(
                height=280,
                xaxis_title="Quarter",
                yaxis_title="Patient Change",
                plot_bgcolor="white",
                paper_bgcolor="white",
                yaxis=dict(gridcolor="#f0f0f0"),
            )
            st.plotly_chart(fig4, width="stretch")

        # Data table for this practice
        with st.expander("View data table"):
            display_cols = ["QuarterLabel", "ListSize", "Change", "ChangePct", "LCG"]
            disp = practice_df[display_cols].copy()
            disp.columns = ["Quarter", "List Size", "Change", "Change %", "LCG"]
            disp["Change"] = disp["Change"].apply(lambda x: f"{x:+,.0f}" if pd.notna(x) else "—")
            disp["Change %"] = disp["Change %"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
            disp["List Size"] = disp["List Size"].apply(lambda x: f"{x:,}")
            st.dataframe(disp, width="stretch", hide_index=True)


# ══════════════════════════════════════════════
# TAB 4 — LCG Breakdown
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Patient Distribution by Local Commissioning Group</div>', unsafe_allow_html=True)

    lcg_quarterly = (
        filtered_df.groupby(["QuarterDate", "QuarterLabel", "LCG"])
        .agg(TotalPatients=("ListSize", "sum"), Practices=("PracticeCode", "nunique"))
        .reset_index()
        .sort_values("QuarterDate")
    )

    col_x, col_y = st.columns(2)
    with col_x:
        # Stacked area chart
        fig5 = px.area(
            lcg_quarterly,
            x="QuarterLabel",
            y="TotalPatients",
            color="LCG",
            title="Total Patients by LCG Over Time",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig5.update_layout(
            height=380, xaxis_title="Quarter", yaxis_title="Registered Patients",
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#f0f0f0"),
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig5, width="stretch")

    with col_y:
        # Latest quarter pie
        lcg_latest = lcg_quarterly[lcg_quarterly["QuarterDate"] == latest_quarter]
        if not lcg_latest.empty:
            fig6 = px.pie(
                lcg_latest,
                values="TotalPatients",
                names="LCG",
                title=f"Patient Share by LCG ({quarter_label(latest_quarter) if latest_quarter else 'Latest'})",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            fig6.update_layout(height=380, paper_bgcolor="white")
            st.plotly_chart(fig6, width="stretch")

    # LCG growth comparison
    st.markdown('<div class="section-header">LCG Growth Rate Comparison</div>', unsafe_allow_html=True)
    if latest_quarter and prev_quarter:
        lcg_comp = lcg_quarterly[lcg_quarterly["QuarterDate"].isin([latest_quarter, prev_quarter])]
        lcg_pivot = lcg_comp.pivot_table(index="LCG", columns="QuarterDate", values="TotalPatients")
        if lcg_pivot.shape[1] >= 2:
            lcg_pivot["Change"] = lcg_pivot.iloc[:, 1] - lcg_pivot.iloc[:, 0]
            lcg_pivot["ChangePct"] = (lcg_pivot["Change"] / lcg_pivot.iloc[:, 0] * 100).round(2)
            lcg_pivot = lcg_pivot.reset_index()

            fig7 = go.Figure(go.Bar(
                x=lcg_pivot["LCG"],
                y=lcg_pivot["ChangePct"],
                marker_color=lcg_pivot["ChangePct"].apply(lambda x: "#27ae60" if x >= 0 else "#e74c3c"),
                text=lcg_pivot["ChangePct"].apply(lambda x: f"{x:+.1f}%"),
                textposition="outside",
            ))
            fig7.update_layout(
                height=320,
                xaxis_title="Local Commissioning Group",
                yaxis_title="Patient Change (%)",
                plot_bgcolor="white",
                paper_bgcolor="white",
                yaxis=dict(gridcolor="#f0f0f0"),
            )
            st.plotly_chart(fig7, width="stretch")

    # Practice count by LCG
    st.markdown('<div class="section-header">Number of Active Practices by LCG</div>', unsafe_allow_html=True)
    if latest_quarter:
        prac_count = (
            filtered_df[filtered_df["QuarterDate"] == latest_quarter]
            .groupby("LCG")["PracticeCode"].nunique()
            .reset_index()
            .rename(columns={"PracticeCode": "Practices"})
            .sort_values("Practices", ascending=True)
        )
        fig8 = px.bar(
            prac_count, x="Practices", y="LCG", orientation="h",
            color="Practices",
            color_continuous_scale="Blues",
            title=f"Active Practices per LCG ({quarter_label(latest_quarter) if latest_quarter else 'Latest'})",
        )
        fig8.update_layout(height=350, plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
        st.plotly_chart(fig8, width="stretch")


# ══════════════════════════════════════════════
# TAB 5 — Movers & Alerts
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Biggest Movers (Latest Quarter)</div>', unsafe_allow_html=True)

    latest_changes = (
        filtered_df[filtered_df["QuarterDate"] == latest_quarter]
        .dropna(subset=["ChangePct"])
        .sort_values("ChangePct", ascending=False)
    ) if latest_quarter else pd.DataFrame()

    if not latest_changes.empty:
        col_g, col_s = st.columns(2)

        top_n = 10
        top_growers  = latest_changes.head(top_n)
        top_shrinkers = latest_changes.tail(top_n).sort_values("ChangePct")

        with col_g:
            st.markdown("#### 🟢 Top Growers")
            fig9 = go.Figure(go.Bar(
                y=top_growers["PracticeName"],
                x=top_growers["ChangePct"],
                orientation="h",
                marker_color="#27ae60",
                text=top_growers["ChangePct"].apply(lambda x: f"{x:+.1f}%"),
                textposition="auto",
            ))
            fig9.update_layout(
                height=400, xaxis_title="Change (%)",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#f0f0f0"),
                margin=dict(l=10),
            )
            st.plotly_chart(fig9, width="stretch")

        with col_s:
            st.markdown("#### 🔴 Biggest Shrinkers")
            fig10 = go.Figure(go.Bar(
                y=top_shrinkers["PracticeName"],
                x=top_shrinkers["ChangePct"],
                orientation="h",
                marker_color="#e74c3c",
                text=top_shrinkers["ChangePct"].apply(lambda x: f"{x:+.1f}%"),
                textposition="auto",
            ))
            fig10.update_layout(
                height=400, xaxis_title="Change (%)",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#f0f0f0"),
                margin=dict(l=10),
            )
            st.plotly_chart(fig10, width="stretch")

    # Alerts table
    st.markdown(f'<div class="section-header">⚠️ Practices Exceeding ±{change_threshold}% Change (All Quarters)</div>', unsafe_allow_html=True)

    alerts = filtered_df[
        filtered_df["ChangePct"].abs() > change_threshold
    ].copy().dropna(subset=["ChangePct"])

    if not alerts.empty:
        alerts["Alert"] = alerts["ChangePct"].apply(
            lambda x: f"🟢 +{x:.1f}%" if x > 0 else f"🔴 {x:.1f}%"
        )
        alert_display = alerts[[
            "QuarterLabel", "PracticeCode", "PracticeName", "LCG",
            "ListSize", "Change", "Alert"
        ]].sort_values("QuarterLabel", ascending=False)

        alert_display.columns = [
            "Quarter", "Code", "Practice Name", "LCG",
            "List Size", "Change", "Change %"
        ]
        alert_display["List Size"] = alert_display["List Size"].apply(lambda x: f"{x:,}")
        alert_display["Change"]    = alert_display["Change"].apply(
            lambda x: f"{x:+,.0f}" if pd.notna(x) else "—"
        )
        st.dataframe(alert_display, width="stretch", hide_index=True, height=400)

        # Download
        csv_buf = io.BytesIO()
        alerts.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download Alerts as CSV",
            data=csv_buf.getvalue(),
            file_name="gp_practice_alerts.csv",
            mime="text/csv",
        )
    else:
        st.success(f"No practices exceeded the ±{change_threshold}% threshold.")

    # Scatter: list size vs change
    st.markdown('<div class="section-header">List Size vs Growth Rate (Latest Quarter)</div>', unsafe_allow_html=True)
    if not latest_changes.empty:
        fig11 = px.scatter(
            latest_changes,
            x="ListSize",
            y="ChangePct",
            color="LCG",
            hover_name="PracticeName",
            hover_data={"PracticeCode": True, "Change": True},
            size="ListSize",
            size_max=30,
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"ListSize": "Registered Patients", "ChangePct": "Change (%)"},
        )
        fig11.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
        fig11.add_hline(y=change_threshold,  line_dash="dot", line_color="green", opacity=0.4)
        fig11.add_hline(y=-change_threshold, line_dash="dot", line_color="red",   opacity=0.4)
        fig11.update_layout(
            height=420,
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(gridcolor="#f0f0f0"),
            xaxis=dict(gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig11, width="stretch")

    # ── All-time scatter ──
    st.markdown('<div class="section-header">List Size vs Growth Rate — All Time</div>', unsafe_allow_html=True)
    st.caption(
        "Each bubble is one practice. X-axis = latest list size. "
        "Y-axis = overall % growth from first to last recorded quarter across all data. "
        "Hover to see practice name, Known As, code, and full figures."
    )

    # Build all-time summary per practice, pulling in KnownAs if available
    _alltime = filtered_df.sort_values("QuarterDate")

    # Check whether a KnownAs / address column exists in the data
    _has_known_as = "PracticeAddress" in filtered_df.columns

    alltime_agg = (
        _alltime.groupby("PracticeCode")
        .agg(
            PracticeName  = ("PracticeName",    "last"),
            LCG           = ("LCG",             "last"),
            FirstSize     = ("ListSize",         "first"),
            LastSize      = ("ListSize",         "last"),
            FirstQuarter  = ("QuarterLabel",     "first"),
            LastQuarter   = ("QuarterLabel",     "last"),
            **( {"KnownAs": ("PracticeAddress", "last")} if _has_known_as else {} )
        )
        .reset_index()
    )

    if not _has_known_as:
        alltime_agg["KnownAs"] = ""

    alltime_agg["KnownAs"]   = alltime_agg["KnownAs"].fillna("").astype(str).str.strip()
    alltime_agg["DisplayName"] = alltime_agg.apply(
        lambda r: r["KnownAs"] if r["KnownAs"] else r["PracticeName"], axis=1
    )
    alltime_agg["NetChange"]  = alltime_agg["LastSize"] - alltime_agg["FirstSize"]
    alltime_agg["GrowthPct"]  = (alltime_agg["NetChange"] / alltime_agg["FirstSize"] * 100).round(1)

    at_median_size   = alltime_agg["LastSize"].median()
    at_median_growth = alltime_agg["GrowthPct"].median()

    fig_at = px.scatter(
        alltime_agg,
        x="LastSize",
        y="GrowthPct",
        size="LastSize",
        color="LCG",
        hover_name="DisplayName",
        custom_data=[
            "PracticeCode",   # 0
            "PracticeName",   # 1
            "KnownAs",        # 2
            "FirstSize",      # 3
            "LastSize",       # 4
            "NetChange",      # 5
            "GrowthPct",      # 6
            "FirstQuarter",   # 7
            "LastQuarter",    # 8
            "LCG",            # 9
        ],
        size_max=45,
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={
            "LastSize":  "Latest List Size (Registered Patients)",
            "GrowthPct": "Overall Growth (%)",
            "LCG":       "LCG",
        },
    )

    fig_at.update_traces(
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "<i>Known As: %{customdata[2]}</i><br>"
            "Code: %{customdata[0]}<br>"
            "LCG: %{customdata[9]}<br>"
            "───────────────────<br>"
            "Latest list size: %{customdata[4]:,}<br>"
            "Overall growth: %{customdata[6]:+.1f}%<br>"
            "Net change: %{customdata[5]:+,} patients<br>"
            "Period: %{customdata[7]} → %{customdata[8]}<extra></extra>"
        )
    )

    # Quadrant reference lines
    fig_at.add_vline(x=at_median_size,   line_dash="dot",   line_color="#aaa", line_width=1)
    fig_at.add_hline(y=at_median_growth, line_dash="dot",   line_color="#aaa", line_width=1)
    fig_at.add_hline(y=0,                line_dash="solid",  line_color="#e74c3c", line_width=1, opacity=0.4)

    _x_max = alltime_agg["LastSize"].max()
    _y_max = alltime_agg["GrowthPct"].max()
    _y_min = alltime_agg["GrowthPct"].min()

    for _label, _x, _y, _color, _anchor in [
        ("Large & Growing",   _x_max * 0.98, _y_max * 0.92, "#27ae60", "right"),
        ("Small & Growing",   _x_max * 0.02, _y_max * 0.92, "#2980b9", "left"),
        ("Large & Shrinking", _x_max * 0.98, _y_min * 0.92, "#e74c3c", "right"),
        ("Small & Shrinking", _x_max * 0.02, _y_min * 0.92, "#e67e22", "left"),
    ]:
        fig_at.add_annotation(
            x=_x, y=_y, text=_label,
            showarrow=False,
            font=dict(size=10, color=_color),
            xanchor=_anchor,
            opacity=0.6,
        )

    fig_at.update_layout(
        height=540,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0", tickformat=",d"),
        yaxis=dict(gridcolor="#f0f0f0", ticksuffix="%"),
        legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
    )
    st.plotly_chart(fig_at, width="stretch")


# ══════════════════════════════════════════════
# TAB 6 — Raw Data
# ══════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Full Dataset</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        q_options = ["All"] + [quarter_label(q) for q in sorted(quarters_sorted)]
        q_select = st.selectbox("Filter by Quarter", q_options)
    with col_f2:
        practice_search = st.text_input("Search practice name / code", "")
    with col_f3:
        sort_col = st.selectbox("Sort by", ["QuarterDate", "ListSize", "ChangePct", "PracticeName"])

    display_raw = filtered_df.copy()
    if q_select != "All":
        display_raw = display_raw[display_raw["QuarterLabel"] == q_select]
    if practice_search:
        mask = (
            display_raw["PracticeName"].str.contains(practice_search, case=False, na=False) |
            display_raw["PracticeCode"].str.contains(practice_search, case=False, na=False)
        )
        display_raw = display_raw[mask]

    display_raw = display_raw.sort_values(sort_col, ascending=(sort_col == "PracticeName"))

    show_cols = ["QuarterLabel", "PracticeCode", "PracticeName", "LCG", "ListSize", "Change", "ChangePct"]
    st.dataframe(
        display_raw[show_cols].rename(columns={
            "QuarterLabel": "Quarter", "PracticeCode": "Code",
            "PracticeName": "Practice", "ListSize": "List Size",
            "ChangePct": "Change %"
        }),
        width="stretch",
        hide_index=True,
        height=500,
    )

    st.markdown(f"**{len(display_raw):,} rows shown**")

    csv_out = io.BytesIO()
    display_raw[show_cols].to_csv(csv_out, index=False)
    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=csv_out.getvalue(),
        file_name="gp_practice_data_filtered.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>Data source: "
    "<a href='https://www.opendatani.gov.uk/dataset/gp-practice-list-sizes' target='_blank'>"
    "Open Data NI — GP Practice List Sizes</a> · "
    "Built with Streamlit & Plotly</small>",
    unsafe_allow_html=True,
)
