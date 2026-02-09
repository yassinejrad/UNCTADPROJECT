import numpy as np
import autograd.numpy as anp
from autograd import jacobian
from scipy.optimize import minimize, NonlinearConstraint
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

from data_loader import (
    discover_indicators,
    parse_sfa_coefficients,
    load_historical_data_by_iso3,
    build_starting_point,
    load_indicator_metadata,
    iso3_to_country_name
)

# ======================================================
# SP-BALANCED (MIN–MAX) OPTIMIZATION 
# ======================================================

def optimize_country_year_minimax(
    iso3,
    year,
    cv_year,
    data_file,
    coef_dir,
    metadata_file,
    previous_expenditures=None,
    anchor_mode="rolling",      # rolling | fixed | base_skip
    reference_year=None,
    start_year=None,
    normalize_indicators=True,
    verbose=True
):

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    def log(msg):
        if verbose:
            print(msg)

    log("\n" + "=" * 80)
    log(f" SP-BALANCED OPTIMIZATION — {iso3} | {year} → {year + 1}")
    log("=" * 80)

    # --------------------------------------------------
    # Country
    # --------------------------------------------------
    country_name = iso3_to_country_name(data_file, iso3)
    log(f"Country : {country_name}")

    # --------------------------------------------------
    # Load SFA coefficients
    # --------------------------------------------------
    indicator_files = discover_indicators(coef_dir)
    indicators = {}
    all_opt_vars = set()

    for ind, path in indicator_files.items():
        coef = parse_sfa_coefficients(path)
        indicators[ind] = coef
        all_opt_vars.update(coef["opt_vars"])

    OPT_VARS = sorted(all_opt_vars)
    n = len(OPT_VARS)
    log(f"Optimization variables ({n}): {OPT_VARS}")

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------
    meta_all = load_indicator_metadata(metadata_file)
    for ind in indicators:
        indicators[ind]["meta"] = meta_all.get(ind, {})

    # --------------------------------------------------
    #  CONTROLS 
    # --------------------------------------------------
    CONTROL_VALUES, _, _ = load_historical_data_by_iso3(
        data_file, iso3, year, cv_year, OPT_VARS
    )

    # --------------------------------------------------
    #  HISTORICAL ANCHORING (X0 ONLY)
    # --------------------------------------------------
    if anchor_mode == "rolling" and previous_expenditures is not None:
        X0 = np.array([previous_expenditures[v] for v in OPT_VARS])
        log("X0 source : OPTIMIZED (t−1)")

    elif anchor_mode == "fixed":
        if reference_year is None:
            raise ValueError("reference_year required for fixed mode")

        _, HIST_INPUTS, _ = load_historical_data_by_iso3(
            data_file, iso3, reference_year, cv_year, OPT_VARS
        )
        X0 = np.array(build_starting_point(OPT_VARS, HIST_INPUTS))
        log(f"X0 source : FIXED HISTORICAL YEAR ({reference_year})")

    elif anchor_mode == "base_skip":
        if reference_year is None or start_year is None:
            raise ValueError("reference_year and start_year required for base_skip")

        if previous_expenditures is None or year == start_year:
            _, HIST_INPUTS, _ = load_historical_data_by_iso3(
                data_file, iso3, reference_year, cv_year, OPT_VARS
            )
            X0 = np.array(build_starting_point(OPT_VARS, HIST_INPUTS))
            log(f"X0 source : BASE YEAR ({reference_year})")
        else:
            X0 = np.array([previous_expenditures[v] for v in OPT_VARS])
            log("X0 source : OPTIMIZED (t−1)")

    else:
        _, HIST_INPUTS, _ = load_historical_data_by_iso3(
            data_file, iso3, year, cv_year, OPT_VARS
        )
        X0 = np.array(build_starting_point(OPT_VARS, HIST_INPUTS))
        log("X0 source : HISTORICAL DATA")

    HIST_TOTAL = np.sum(X0)
    log(f"Initial total expenditure = {HIST_TOTAL:.4f}")

    # --------------------------------------------------
    # GDP growth
    # --------------------------------------------------
    CONTROL_VALUES_PREV, _, _ = load_historical_data_by_iso3(
        data_file, iso3, year, year, OPT_VARS
    )

    gdp_t = CONTROL_VALUES.get("CV_GDP")
    gdp_t_minus_1 = CONTROL_VALUES_PREV.get("CV_GDP")

    gdp_growth = None
    if gdp_t and gdp_t_minus_1 and gdp_t_minus_1 > 0:
        gdp_growth = (gdp_t - gdp_t_minus_1) / gdp_t_minus_1
    else:
        log("❌ GDP growth could not be computed")

    # --------------------------------------------------
    # Inefficiency
    # --------------------------------------------------
    def inefficiency(coef):
        u = 0.0
        for cv, c in coef.get("inefficiency", {}).get("cv", {}).items():
            if cv in CONTROL_VALUES and CONTROL_VALUES[cv] > 0:
                u += c * np.log(CONTROL_VALUES[cv])

        fe = coef.get("inefficiency", {}).get("country", {}).get(country_name)
        if fe is not None:
            u += fe

        return max(u, 0.0)

    # --------------------------------------------------
    # Translog
    # --------------------------------------------------
    def build_translog(coef, meta):
        idx = {v: i for i, v in enumerate(OPT_VARS)}
        u = inefficiency(coef)
        direction = meta.get("direction")

        def f(X):
            X = anp.maximum(X, 1e-12)
            lnX = anp.log(X)
            y = coef["frontier"]["intercept"]

            for v, c in coef["frontier"]["linear"].items():
                y += c * lnX[idx[v]]
            for v, c in coef["frontier"]["quadratic"].items():
                y += c * lnX[idx[v]] ** 2
            for (v1, v2), c in coef["frontier"]["cross"].items():
                y += c * lnX[idx[v1]] * lnX[idx[v2]]

            # controls 
            for cv, c in coef["frontier"]["controls"].items():
                if cv in CONTROL_VALUES and CONTROL_VALUES[cv] > 0:
                    y += c * anp.log(CONTROL_VALUES[cv])

            if direction == "lowerOrEqual":
                y += u
            elif direction == "upperOrEqual":
                y -= u

            return y

        return f, jacobian(f)

    translog_models = {
        ind: build_translog(indicators[ind], indicators[ind]["meta"])
        for ind in indicators
    }

    # --------------------------------------------------
    # Aligned & normalized indicators
    # --------------------------------------------------
    aligned_indicators = []

    for ind, (f, _) in translog_models.items():
        meta = indicators[ind]["meta"]
        direction = meta.get("direction")
        baseline = f(X0)

        def Ik(X, f=f, direction=direction, baseline=baseline):
            val = f(X)
            if normalize_indicators:
                val = val / baseline
            return val if direction == "upperOrEqual" else -val

        aligned_indicators.append(Ik)

    # --------------------------------------------------
    # Extended variable Z = [X, t]
    # --------------------------------------------------
    X0_ext = np.concatenate([X0, [0.0]])

    def objective(Z):
        return -Z[-1]  # maximize t

    constraints = []

    # --------------------------------------------------
    # Min–max constraints: Ik(X) ≥ t
    # --------------------------------------------------
    for Ik in aligned_indicators:
        constraints.append(
            NonlinearConstraint(
                lambda Z, Ik=Ik: Ik(Z[:-1]) - Z[-1],
                lb=0,
                ub=np.inf
            )
        )

    # --------------------------------------------------
    # Indicator bounds
    # --------------------------------------------------
    for ind, (f, _) in translog_models.items():
        meta = indicators[ind]["meta"]
        lb = meta.get("lb")
        ub = meta.get("ub")

        lb_log = np.log(lb) if lb and lb > 0 else -np.inf
        ub_log = np.log(ub) if ub and ub > 0 else np.inf

        if np.isfinite(lb_log) or np.isfinite(ub_log):
            constraints.append(
                NonlinearConstraint(
                    lambda Z, f=f: f(Z[:-1]),
                    lb=lb_log,
                    ub=ub_log
                )
            )

    # --------------------------------------------------
    # GDP macro constraint
    # --------------------------------------------------
    if gdp_growth is not None:
        gdp_factor = 1 + gdp_growth
        lb = min(HIST_TOTAL, HIST_TOTAL * gdp_factor)
        ub = max(HIST_TOTAL, HIST_TOTAL * gdp_factor)

        log("Macro corridor:")
        log(f"  LB = {lb:.4f}")
        log(f"  UB = {ub:.4f}")

        constraints.append(
            NonlinearConstraint(
                lambda Z: np.sum(Z[:-1]),
                lb=lb,
                ub=ub
            )
        )

    # --------------------------------------------------
    # Bounds
    # --------------------------------------------------
    bounds = [(1e-6, None)] * n + [(None, None)]

    # --------------------------------------------------
    # Solve
    # --------------------------------------------------
    log("\n🚀 Starting SDG-balanced solver...")
    res = minimize(
        objective,
        X0_ext,
        method="trust-constr",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000}
    )

    log(f"Solver success : {res.success}")
    log(f"Solver message : {res.message}")
    log(f"Balanced indicator level (t*) = {res.x[-1]:.6f}")
    log(f"Total expenditure = {np.sum(res.x[:-1]):.4f}")
    log("=" * 80)

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    return {
        "iso3": iso3,
        "year": year + 1,
        "expenditures": {
            OPT_VARS[i]: float(res.x[i]) for i in range(n)
        },
        "worst_indicator": float(res.x[-1]),
        "indicators": {
            ind: float(np.exp(f(res.x[:-1])))
            for ind, (f, _) in translog_models.items()
        }
    }
