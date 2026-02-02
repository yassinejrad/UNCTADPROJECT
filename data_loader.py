import os
import re
import pandas as pd

# =====================================================
#  DISCOVER SFA COEFFICIENT FILES
# =====================================================

def discover_indicators(coef_dir):
    indicators = {}
    for f in os.listdir(coef_dir):
        if f.startswith("sfa_coefficients_") and f.endswith(".csv"):
            ind = f.replace("sfa_coefficients_", "").replace(".csv", "")
            indicators[ind] = os.path.join(coef_dir, f)
    return indicators


# =====================================================
#  PARSE SFA COEFFICIENT FILE (FRONTIER + INEFF)
# =====================================================

def parse_sfa_coefficients(path):
    df = pd.read_csv(path)

    coef = {
        # -------- FRONTIER --------
        "frontier": {
            "intercept": 0.0,
            "linear": {},
            "quadratic": {},
            "cross": {},
            "controls": {}
        },

        # -------- INEFFICIENCY --------
        "inefficiency": {
            "intercept": 0.0,
            "cv": {},
            "country": {}
        },

        # -------- OPTIMIZATION VARS --------
        "opt_vars": []
    }

    for _, row in df.iterrows():
        name = str(row["Name"]).strip()
        value = float(row["Value"])

        # =========================
        # FRONTIER PART
        # =========================

        if name == "(Intercept)":
            coef["frontier"]["intercept"] = value

        elif re.match(r"I\s*\(\s*log\(.+\)\^2\s*\)", name):
            v = re.findall(r"log\((.+?)\)", name)[0]
            coef["frontier"]["quadratic"][v] = value

        elif "*" in name and "log(" in name:
            v1, v2 = re.findall(r"log\((.+?)\)", name)
            coef["frontier"]["cross"][(v1, v2)] = value

        elif name.startswith("log(CV_"):
            v = re.findall(r"log\((.+?)\)", name)[0]
            coef["frontier"]["controls"][v] = value

        elif name.startswith("log("):
            v = re.findall(r"log\((.+?)\)", name)[0]
            coef["frontier"]["linear"][v] = value

        # =========================
        # INEFFICIENCY PART (Z_)
        # =========================

        elif name.startswith("Z_"):

            z = name.replace("Z_", "")

            # ---- inefficiency intercept
            if z == "(Intercept)":
                coef["inefficiency"]["intercept"] += value

            # ---- inefficiency control variables
            elif z.startswith("CV_"):
                coef["inefficiency"]["cv"][z] = value

            # ---- country fixed effects
            elif z.startswith("factor(Country_name)"):
                country = z.replace("factor(Country_name)", "").strip()
                coef["inefficiency"]["country"][country] = value

    coef["opt_vars"] = sorted(coef["frontier"]["linear"].keys())
    return coef


# =====================================================
#  LOAD HISTORICAL DATA 
# =====================================================

def load_historical_data_by_iso3(
    data_file,
    iso3,
    exp_year,
    cv_year,
    opt_vars
):
    df = pd.read_csv(data_file)

    if "iso3" not in df.columns or "years" not in df.columns:
        raise ValueError("❌ Data file must contain 'iso3' and 'years'")

    df_country = df[df["iso3"] == iso3]

    if df_country.empty:
        raise ValueError(f"❌ No data for iso3={iso3}")

    # ---------- expenditures (exp_year)
    row_exp = df_country[df_country["years"] == exp_year].iloc[0]

    inputs = {}
    missing = []

    for v in opt_vars:
        if v in row_exp and pd.notna(row_exp[v]) and row_exp[v] > 0:
            inputs[v] = float(row_exp[v])
        else:
            inputs[v] = None
            missing.append(v)

    # ---------- controls (cv_year)
    row_cv = df_country[df_country["years"] == cv_year].iloc[0]

    controls = {
        c: float(row_cv[c])
        for c in df.columns
        if c.startswith("CV_") and pd.notna(row_cv[c])
    }

    return controls, inputs, missing


# =====================================================
#  BUILD STARTING POINT
# =====================================================

def build_starting_point(opt_vars, hist_inputs, default=1.0):
    return [
        hist_inputs[v] if hist_inputs[v] is not None and hist_inputs[v] > 0
        else default
        for v in opt_vars
    ]


# =====================================================
#  LOAD METADATA (INDICATORS)
# =====================================================

def load_indicator_metadata(path):
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252")

    meta = {}

    def safe_float(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return None

    for _, r in df.iterrows():
        ind = str(r.get("ExplorationCodeNew")).strip()

        direction = r.get("TargetDirection")
        if pd.isna(direction):
            direction = None
        else:
            direction = str(direction).strip()

        meta[ind] = {
            "target": safe_float(r.get("Target")),
            "lb": safe_float(r.get("Indicator_lower_value_boundary")),
            "ub": safe_float(r.get("Indicator_upper_value_boundary")),
            "direction": direction
        }

    return meta


# =====================================================
#  ISO3 → Country_name 
# =====================================================

def iso3_to_country_name(metadata_path, iso3):
    df = pd.read_csv(metadata_path)

    if "iso3" not in df.columns or "Country_name" not in df.columns:
        raise ValueError(
            "❌ Metadata must contain 'iso3' and 'Country_name'"
        )

    row = df[df["iso3"] == iso3]

    if row.empty:
        raise ValueError(f"❌ ISO3 '{iso3}' not found in metadata")

    return str(row.iloc[0]["Country_name"]).strip()
