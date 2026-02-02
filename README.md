# 🌍 SP Model Calibration & Optimization

This repository contains a **local Streamlit application** for calibrating **Stochastic Frontier Models (SFA)** in **R**, optimizing **public expenditures** in **Python**, and visualizing results interactively.

⚠️ This project is designed to run **locally only** (no hosting, no server, no database).

---

##  Main Features

- Step-by-step **Streamlit wizard**
- **SFA calibration in R** (Translog)
- Cost or production frontier depending on `TargetDirection`
- Optimization:
  - With or without GDP constraint
  - Single country or multi-country
  - Multi-year horizon
- Interactive charts and diagnostics
- Centralized configuration via `config.py`

---

##  Project Structure

```
├── Step_1_Upload_Data.py
├── pages/
│   ├── Step_2_Model_Specification.py
│   ├── Step_3_Optimization.py
│   ├── Step_4_Charts.py
├── Data/
│   ├── INPUT_SP1_ALL_exploration_subset(in).csv
│   ├── optimized_output.csv
│   ├── coef/
│   │   └── sfa_coefficients_*.csv
│   ├── summary/
│   └── descriptive/
├── optimization.py
├── optimization_gdp.py
├── data_loader.py
├── calibrate.R
├── config.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🖥️ System Requirements

### Python
- Python **≥ 3.9** (recommended: 3.10)

```bash
python --version
```

### R
- R **≥ 4.2**

```bash
R --version
```

### Rscript
Must be accessible from the command line:

```bash
Rscript --version
```

If not, edit `config.py`:

```python
RSCRIPT_EXECUTABLE = "C:/Program Files/R/R-4.3.2/bin/Rscript.exe"
```

---

## 📦 Python Dependencies

All Python dependencies are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 🧪 Required R Packages

Automatically installed by `calibrate.R` if missing:

- frontier
- optparse
- plm
- dplyr
- openxlsx

---

## ⚙️ Configuration (IMPORTANT)

All paths and parameters are centralized in:

```text
config.py
```

Key variables:

```python
PROJECT_ROOT
DATA_DIR
COEF_DIR
INDICATOR_METADATA_FILE
CALIBRATION_SCRIPT
DEFAULT_START_YEAR
DEFAULT_END_YEAR
```

⚠️ If the project is moved to another folder, **only `PROJECT_ROOT` must be updated**.

---

## ▶️ Running the Application

From the project root:

```bash
python -m streamlit run Step_1_Upload_Data.py
```

---

## 🧭 Application Workflow

### Step 1 — Upload Data
- Upload CSV or Excel file
- Data stored locally

### Step 2 — Filter Dataset
- Optional filtering (country, region, etc.)

### Step 3 — Calibration (R)
- Configure indicators, expenditures, controls
- Choose model specification
- Run SFA calibration
- Outputs written to:
  ```
  Data/coef/
  Data/summary/
  Data/descriptive/
  ```

### Step 4 — Optimization (Python)
- Choose:
  - With GDP constraint or without
  - Single country or all countries
  - Time horizon
- Progress bar with ETA
- Results written directly into dataset

### Step 5 — Results & Diagnostics
- Total expenditure evolution
- Reallocation analysis
- Indicators vs targets
- Constraint status

---

## 📂 Outputs

```
Data/
├── coef/
│   └── sfa_coefficients_X*.csv
├── summary/
│   └── sfa_summary_X*.txt
├── descriptive/
├── optimized_output.csv
```

---

## ⚠️ Common Issues

### Rscript not found
➡️ Fix `RSCRIPT_EXECUTABLE` in `config.py`

### No coefficients found
➡️ Calibration must be executed before optimization



