import numpy as np
import autograd.numpy as anp
from autograd import jacobian
from scipy.optimize import minimize, NonlinearConstraint
from data_loader import *


# ======================================================
# SDG-BALANCED (MIN–MAX) OPTIMIZATION WITH GDP CONSTRAINT
# + PHYSICAL INDICATOR BOUNDS
# ======================================================
def optimize_country_year_minimax(
    iso3,
    year,
    cv_year,
    data_file,
    coef_dir,
    metadata_file,
    previous_expenditures=None,
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
    log(f"🌍 SDG-BALANCED OPTIMIZATION — {iso3} | {year} → {year+1}")
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
    # Initial point X0
    # --------------------------------------------------
    if previous_expenditures is not None:
        X0 = np.array([previous_expenditures[v] for v in OPT_VARS])
        HIST_TOTAL = np.sum(X0)
        log("X0 source : OPTIMIZED (t−1)")
    else:
        CONTROL_VALUES, HIST_INPUTS, _ = load_historical_data_by_iso3(
            data_file, iso3, year, cv_year, OPT_VARS
        )
        X0 = np.array(build_starting_point(OPT_VARS, HIST_INPUTS))
        HIST_TOTAL = np.sum(X0)
        log("X0 source : HISTORICAL DATA")

    log(f"Initial total expenditure = {HIST_TOTAL:.4f}")

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------
    meta_all = load_indicator_metadata(metadata_file)
    for ind in indicators:
        indicators[ind]["meta"] = meta_all.get(ind, {})

    # --------------------------------------------------
    # GDP growth
    # --------------------------------------------------
    CONTROL_VALUES, _, _ = load_historical_data_by_iso3(
        data_file, iso3, year, cv_year, OPT_VARS
    )
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
    # Inefficiency term
    # --------------------------------------------------
    def inefficiency(coef):
        u = 0.0
        for cv, c in coef["inefficiency"]["cv"].items():
            if cv in CONTROL_VALUES and CONTROL_VALUES[cv] > 0:
                u += c * np.log(CONTROL_VALUES[cv])
        fe = coef["inefficiency"]["country"].get(country_name)
        if fe is not None:
            u += fe
        return max(u, 0.0)

    # --------------------------------------------------
    # Translog construction
    # --------------------------------------------------
    def build_translog(coef, meta):
        idx = {v: i for i, v in enumerate(OPT_VARS)}
        u = inefficiency(coef)
        direction = meta.get("direction")

        def f(X):
            lnX = anp.log(anp.maximum(X, 1e-12))
            y = coef["frontier"]["intercept"]

            for v, c in coef["frontier"]["linear"].items():
                y += c * lnX[idx[v]]
            for v, c in coef["frontier"]["quadratic"].items():
                y += c * lnX[idx[v]] ** 2
            for (v1, v2), c in coef["frontier"]["cross"].items():
                y += c * lnX[idx[v1]] * lnX[idx[v2]]
            for cv, c in coef["frontier"]["controls"].items():
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
    # Aligned & normalized indicators Ik(X)
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
        return -Z[-1]   # maximize t

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
    # PHYSICAL INDICATOR BOUNDS (domain constraints)
    # --------------------------------------------------
    for ind, (f, jac) in translog_models.items():
        meta = indicators[ind]["meta"]

        lb = meta.get("lb")        # physical lower bound
        ub = meta.get("ub")        # physical upper bound

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
    gdp_factor = 1 + gdp_growth
    lb = min(HIST_TOTAL, HIST_TOTAL * gdp_factor)
    ub = max(HIST_TOTAL, HIST_TOTAL * gdp_factor)
    log("Macro corridor:")
    log(f"  LB = {lb:.4f}")
    log(f"  UB = {ub:.4f}")
    if gdp_growth is not None:
        constraints.append(
            NonlinearConstraint(
                lambda Z: np.sum(Z[:-1]),
                lb=lb,
                ub=ub 
            )
        )

    # --------------------------------------------------
    # Bounds on X and t
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
            OPT_VARS[i]: res.x[i] for i in range(n)
        },
        "worst_indicator": float(res.x[-1]),
        "indicators": {
            ind: float(np.exp(f(res.x[:-1])))
            for ind, (f, _) in translog_models.items()
        }
    }
