import os
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

# =====================================================
# MAIN OPTIMIZATION FUNCTION
# =====================================================
HISTORICAL_YEAR = 2022

def optimize_country_year(
    iso3,
    year,
    cv_year,
    data_file,
    coef_dir,
    metadata_file
):
    

    # -----------------------------
    # Country mapping
    # -----------------------------
    country_name = iso3_to_country_name(data_file, iso3)

    # -----------------------------
    # Load coefficients
    # -----------------------------
    indicator_files = discover_indicators(coef_dir)
    indicators = {}
    all_opt_vars = set()

    for ind, path in indicator_files.items():
        coef = parse_sfa_coefficients(path)
        indicators[ind] = coef
        all_opt_vars.update(coef["opt_vars"])

    OPT_VARS = sorted(all_opt_vars)
    N_VARS = len(OPT_VARS)

    # -----------------------------
    # Load historical data
    # -----------------------------
    CONTROL_VALUES, HIST_INPUTS, _ = load_historical_data_by_iso3(
        data_file, iso3, year, cv_year, OPT_VARS
    )

    X0 = np.array(build_starting_point(OPT_VARS, HIST_INPUTS))
    HIST_TOTAL = np.sum(X0)

    # -----------------------------
    # Metadata
    # -----------------------------
    meta_all = load_indicator_metadata(metadata_file)
    for ind in indicators:
        indicators[ind]["meta"] = meta_all.get(ind, {})

    # -----------------------------
    # Inefficiency
    # -----------------------------
    def compute_inefficiency(coef):
        ineff = coef.get("inefficiency", {})
        u = 0.0

        for cv, c in ineff.get("cv", {}).items():
            if cv in CONTROL_VALUES and CONTROL_VALUES[cv] > 0:
                u += c * np.log(CONTROL_VALUES[cv])

        country_fe = ineff.get("country", {}).get(country_name)
        if country_fe is not None:
            u += country_fe

        return max(u, 0.0)

    # -----------------------------
    # Build translog
    # -----------------------------
    def build_translog(coef, meta):
        idx = {v: i for i, v in enumerate(OPT_VARS)}
        u = compute_inefficiency(coef)
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

            for cv, c in coef["frontier"]["controls"].items():
                y += c * anp.log(CONTROL_VALUES[cv])

            # inefficiency
            if direction == "lowerOrEqual":      # cost
                y = y + u
            elif direction == "upperOrEqual":    # production
                y = y - u

            return y

        return f, jacobian(f)

    translog_models = {
        ind: build_translog(indicators[ind], indicators[ind]["meta"])
        for ind in indicators
    }

    # -----------------------------
    # Constraints
    # -----------------------------
    constraints = []

    def indicator_bounds(meta):
        lb, ub = -np.inf, np.inf

        if meta.get("lb") and meta["lb"] > 0:
            lb = np.log(meta["lb"])
        if meta.get("ub") and meta["ub"] > 0:
            ub = np.log(meta["ub"])

        t = meta.get("target")
        d = meta.get("direction")
        if t and t > 0 and d:
            tlog = np.log(t)
            if d == "lowerOrEqual":
                ub = min(ub, tlog)
            elif d == "upperOrEqual":
                lb = max(lb, tlog)
            elif d == "equal":
                lb = ub = tlog

        return lb, ub

    for ind, (f, jac) in translog_models.items():
        lb, ub = indicator_bounds(indicators[ind]["meta"])
        if np.isfinite(lb) or np.isfinite(ub):
            constraints.append(
                NonlinearConstraint(f, lb, ub, jac=jac)
            )

    # -----------------------------
    # Objective
    # -----------------------------
    def objective(X):
        X = np.maximum(X, 1e-12)
        return (np.log(np.sum(X)) - np.log(HIST_TOTAL)) ** 2
    
    def objective_sum(X):
        return np.sum(X)

    bounds = [(1e-6, None)] * N_VARS

    # -----------------------------
    # Solve
    # -----------------------------
    res = minimize(
        objective_sum,
        X0,
        method="trust-constr",
        bounds=bounds,
        constraints=constraints,
        options={"verbose": 0, "maxiter": 1000}
    )

    # -----------------------------
    # Collect results
    # -----------------------------
    expenditures = {OPT_VARS[i]: res.x[i] for i in range(N_VARS)}
    indicators_out = {
        ind: float(np.exp(f(res.x)))
        for ind, (f, _) in translog_models.items()
    }

    return {
        "iso3": iso3,
        "year": year + 1,
        "expenditures": expenditures,
        "indicators": indicators_out
    }
