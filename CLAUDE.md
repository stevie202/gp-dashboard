# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Streamlit data analytics dashboard for tracking GP Practice list size trends in Northern Ireland, using quarterly CSV data from [Open Data NI](https://www.opendatani.gov.uk/dataset/gp-practice-list-sizes).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app (opens at http://localhost:8501)
streamlit run app.py
```

There are no tests, linters, or build steps configured.

## Architecture

The entire application is a **single file: `app.py`** (~1,526 lines). There is no modularization. The file follows a linear structure:

1. **Imports & CSS** (lines 1–76) — Streamlit page config and injected custom CSS for metric cards and styling
2. **Helper functions** (lines 80–456):
   - `parse_quarter_from_filename()` — extracts quarter date from CSV filename
   - `detect_column()` — fuzzy column name matching (supports 30+ variants for practice code, name, list size, LCG)
   - `read_csv_robust()` — tries UTF-8, Latin-1, CP1252, UTF-8-BOM encodings
   - `parse_one_file()` — normalizes a single CSV into the standard schema
   - `load_from_folder()` / `load_and_combine()` — load from `./data/` folder or uploaded files (both `@st.cache_data`)
   - `compute_metrics()` — adds `PrevListSize`, `Change`, `ChangePct` columns via per-practice quarter-over-quarter diff
   - `generate_sample_data()` — creates synthetic demo data (12 hardcoded practices, 5 LCGs)
3. **Sidebar** (lines 461–511) — file upload, LCG multi-select filter, alert threshold slider
4. **Data loading** (lines 515–571) — priority: `./data/` folder → uploaded files → sample data
5. **Dashboard tabs** (lines 573–1526) — 6 tabs: Overall Trends, Top 10 by Year, Practice Detail, LCG Breakdown, Movers & Alerts, Raw Data

## Normalized Data Schema

All data is standardized into this DataFrame shape before rendering:

| Column | Type | Description |
|--------|------|-------------|
| `PracticeCode` | str | e.g. `'P001'` |
| `PracticeName` | str | Practice display name |
| `ListSize` | int | Registered patient count |
| `LCG` | str | Local Commissioning Group / Trust |
| `QuarterDate` | datetime | Parsed from filename |
| `QuarterLabel` | str | `'Q1 2024'` format |
| `SourceFile` | str | Original CSV filename |
| `Change` | int | QoQ patient count delta (added by `compute_metrics`) |
| `ChangePct` | float | QoQ % change (added by `compute_metrics`) |

## Key Data Notes

- **Quarter date is parsed from the filename**, not from file contents. Files must include month+year in the name (e.g. `GP_Practice_July_2024.csv`).
- Practices 473 and 475 are geographically in South Eastern LCG but managed by Southern Trust.
- Practice mergers appear as one practice dropping to 0 and a new one appearing — both will trigger alerts.
- Rows where `ListSize` cannot be parsed as a number are silently dropped.
