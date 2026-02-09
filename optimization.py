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
# COST MINIMIZATION 
# =====================================================

def optimize_country_year(
    iso3,
    year,
    cv_year,
    data_file,
    coef_dir,
    metadata_file,
    previous_expenditures=None,
    anchor_mode="rolling",      
    reference_year=None,
    start_year=None
):

    # --------------------------------------------------
    # Country
    # --------------------------------------------------
    country_name = iso3_to_country_name(data_file, iso3)

    # --------------------------------------------------
    # Load coefficients
    # --------------------------------------------------
    indicator_files = discover_indicators(coef_dir)
    indicators = {}
    all_opt_vars = set()

    for ind, path in indicator_files.items():
        coef = parse_sfa_coefficients(path)
        indicators[ind] = coef
        all_opt_vars.update(coef["opt_vars"])

    OPT_VARS = sorted(all_opt_vars)
    N_VARS = len(OPT_VARS)

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

    elif anchor_mode == "fixed":
        if reference_year is None:
            raise ValueError("reference_year required for fixed mode")

        _, HIST_INPUTS, _ = load_historical_data_by_iso3(
            data_file, iso3, reference_year, cv_year, OPT_VARS
        )
        X0 = np.array(build_starting_point(OPT_VARS, HIST_INPUTS))

    elif anchor_mode == "base_skip":
        if reference_year is None or start_year is None:
            raise ValueError("reference_year and start_year required for base_skip")

        if previous_expenditures is None or year == start_year:
            _, HIST_INPUTS, _ = load_historical_data_by_iso3(
                data_file, iso3, reference_year, cv_year, OPT_VARS
            )
            X0 = np.array(build_starting_point(OPT_VARS, HIST_INPUTS))
        else:
            X0 = np.array([previous_expenditures[v] for v in OPT_VARS])

    else:
        _, HIST_INPUTS, _ = load_historical_data_by_iso3(
            data_file, iso3, year, cv_year, OPT_VARS
        )
        X0 = np.array(build_starting_point(OPT_VARS, HIST_INPUTS))

    # --------------------------------------------------
    # Inefficiency
    # --------------------------------------------------
    def compute_inefficiency(coef):
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
    # Constraints
    # --------------------------------------------------
    constraints = []

    def indicator_bounds(meta):
        lb, ub = -np.inf, np.inf
        if meta.get("lb") and meta["lb"] > 0:
            lb = np.log(meta["lb"])
        if meta.get("ub") and meta["ub"] > 0:
            ub = np.log(meta["ub"])

        t, d = meta.get("target"), meta.get("direction")
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

    # --------------------------------------------------
    # Solve
    # --------------------------------------------------
    res = minimize(
        lambda X: np.sum(X),
        X0,
        method="trust-constr",
        bounds=[(1e-6, None)] * N_VARS,
        constraints=constraints,
        options={"maxiter": 1000}
    )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    return {
        "iso3": iso3,
        "year": year + 1,
        "expenditures": {
            OPT_VARS[i]: float(res.x[i]) for i in range(N_VARS)
        },
        "indicators": {
            ind: float(np.exp(f(res.x)))
            for ind, (f, _) in translog_models.items()
        }
    }
