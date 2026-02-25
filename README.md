# 🏥 GP Practice List Size Dashboard

A Streamlit app for tracking growth and shrinkage of GP Practices in Northern Ireland,
using quarterly CSV files from [Open Data NI](https://www.opendatani.gov.uk/dataset/gp-practice-list-sizes).

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

---

## 📂 Loading Your Data

1. Download quarterly CSV files from [Open Data NI — GP Practice List Sizes](https://www.opendatani.gov.uk/dataset/gp-practice-list-sizes)
2. In the sidebar, click **Browse files** and select all your quarterly CSVs at once
3. Uncheck **"Use sample data"** if it's ticked

### CSV Format Expected
The app auto-detects column names, but your CSVs should contain columns similar to:

| Column | Examples of accepted names |
|--------|--------------------------|
| Practice Code | `Practice Code`, `PracticeCode`, `GP Code` |
| Practice Name | `Practice Name`, `PracticeName`, `Name` |
| List Size | `List Size`, `ListSize`, `Registered Patients`, `Patients` |
| LCG | `LCG`, `Local Commissioning Group`, `Trust`, `HSCT` |

**Quarter date** is read from the filename. Supported patterns:
- `GP_Practice_July_2024.csv`
- `gp-practice-list-sizes-october-2023.csv`
- `GPPractice_2024Q1.csv`

---

## 📊 Dashboard Features

| Tab | What it shows |
|-----|--------------|
| **Overall Trends** | Total patients & active practices over time; QoQ change bar chart |
| **Practice Detail** | Line chart + trend line for any individual practice; QoQ breakdown |
| **LCG Breakdown** | Stacked area by Local Commissioning Group; pie chart; practice counts |
| **Movers & Alerts** | Top growers/shrinkers; configurable alert threshold; scatter plot |
| **Raw Data** | Searchable, filterable full dataset with CSV download |

### Key Controls (Sidebar)
- **LCG filter** — narrow to specific Health Trust areas
- **Growth alert threshold** — flag practices that changed by more than X% in a quarter

---

## 🔧 Troubleshooting

**"Could not identify Practice Code or List Size columns"**
→ Check the column names in your CSV match the expected patterns above. You can rename columns before uploading.

**Quarter date shows as None / missing**
→ Rename your files to include the month and year, e.g. `GP_Practice_April_2024.csv`

**App is slow with many files**
→ Streamlit caches data after the first load. Subsequent filter changes will be fast.

---

## 📝 Notes on the Data

- Practices **473 and 475** are geographically in South Eastern LCG but managed by Southern Trust — this is noted in the Open Data NI metadata.
- **Practice mergers** appear as one practice dropping to zero and a new one appearing with a larger list size — these will be flagged as alerts.
- The app filters out rows where List Size cannot be parsed as a number.
