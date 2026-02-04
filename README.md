# 🌍 SP Model Calibration & Optimization

This repository contains a **local Streamlit application** for calibrating **Stochastic Frontier Models (SFA)** in **R**, optimizing **public expenditures** in **Python**, and visualizing results interactively.

⚠️ This project is designed to run **locally only** (no hosting, no server, no database).

---

## 🚀 Main Features

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

## 📁 Project Structure

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

- frontier
- optparse
- plm
- dplyr
- openxlsx

---

## ⚠️ Common Issue: Installing the `frontier` R Package (IMPORTANT)

On **Windows**, installing the `frontier` package may fail due to **administrative privilege requirements**.

### Recommended fix (manual installation)

1. **Run R or RStudio as Administrator**
2. Install the package manually:

```r
install.packages("frontier", repos = "https://cloud.r-project.org")
```
---

## ▶️ Running the Application

From the project root:

```bash
python -m streamlit run Step_1_Upload_Data.py
```

---

## 📊 How to Read the Charts

### 📈 Total Expenditure (All Years)
- Shows total expenditure over time
- Toggle between **absolute** and **per capita**
- 🔴 Red dots indicate at least one negative expenditure 

### 📊 Optimized Expenditure Allocation
- Distribution of optimized expenditure 
- 🔵 Bars: positive values
- 🔻 Red triangles: negative values

### 🎯 Optimized Value vs Target
- ○ Target
- ● Optimized value
- ▬ Allowed range based on `TargetDirection`
- Values outside the band indicate constraint violations

### 🚦 Constraint Status Table
- Formal validation of indicator constraints

---

## 📂 Outputs

```
Data/
├── coef/
├── summary/
├── descriptive/
├── optimized_output.csv
```

---


