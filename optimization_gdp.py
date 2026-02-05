import numpy as np
import autograd.numpy as anp
from autograd import jacobian
from scipy.optimize import minimize, NonlinearConstraint
from data_loader import *

def optimize_country_year(
    iso3,
    year,
    cv_year,
    data_file,
    coef_dir,
    metadata_file,
    previous_expenditures=None,  
    verbose=True
):

    def log(msg):
        if verbose:
            print(msg)

    log("\n" + "=" * 80)
    log(f"🌍 OPTIMIZATION START — {iso3} | {year} → {year+1}")
    log("=" * 80)

    # --------------------------------------------------
    # Country
    # --------------------------------------------------
    country_name = iso3_to_country_name(data_file, iso3)
    log(f"Country name : {country_name}")

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
    log(f"Optimization variables ({len(OPT_VARS)}): {OPT_VARS}")

    # --------------------------------------------------
    # Initial point X0
    # --------------------------------------------------
    if previous_expenditures is not None:
        X0 = np.array([previous_expenditures[v] for v in OPT_VARS])
        HIST_TOTAL = np.sum(X0)
        log("X0 source : OPTIMIZED (t-1)")
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

    log("\n📈 GDP CHECK")
    log(f"GDP(t-1) [year {year}]     = {gdp_t_minus_1}")
    log(f"GDP(t)   [year {cv_year}] = {gdp_t}")

    gdp_growth = None
    if gdp_t and gdp_t_minus_1 and gdp_t_minus_1 > 0:
        gdp_growth = (gdp_t - gdp_t_minus_1) / gdp_t_minus_1
        log(f"GDP growth rate = {100 * gdp_growth:.2f}%")

        
    else:
        log("❌ GDP growth could NOT be computed")

    # --------------------------------------------------
    # Inefficiency
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
    # Translog
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
            constraints.append(NonlinearConstraint(f, lb, ub, jac=jac))

    if gdp_growth is not None:
        gdp_factor = 1 + gdp_growth
        lb = min(HIST_TOTAL, HIST_TOTAL * gdp_factor)
        ub = max(HIST_TOTAL, HIST_TOTAL * gdp_factor)
        log("Macro corridor:")
        log(f"  LB = {lb:.4f}")
        log(f"  UB = {ub:.4f}")
        constraints.append(
            NonlinearConstraint(
                lambda X: np.sum(X),
                lb=lb,
                ub=ub
            )
        )
        log("✅ GDP macro constraint ACTIVATED")

    # --------------------------------------------------
    # Objective
    # --------------------------------------------------
    def objective(X):
        return np.sum(X)

    # --------------------------------------------------
    # Solve
    # --------------------------------------------------
    log("\n🚀 Starting solver...")
    res = minimize(
        objective,
        X0,
        method="trust-constr",
        bounds=[(1e-6, None)] * len(X0),
        constraints=constraints,
        options={"maxiter": 1000}
    )

    log(f"Solver success : {res.success}")
    log(f"Solver message : {res.message}")
    log(f"Optimized total expenditure = {np.sum(res.x):.4f}")
    log("=" * 80)

    return {
        "iso3": iso3,
        "year": year + 1,
        "expenditures": {OPT_VARS[i]: res.x[i] for i in range(len(OPT_VARS))},
        "indicators": {ind: float(np.exp(f(res.x))) for ind, (f, _) in translog_models.items()}
    }
