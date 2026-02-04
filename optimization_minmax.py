import numpy as np
import autograd.numpy as anp
from autograd import jacobian
from scipy.optimize import minimize, NonlinearConstraint

from data_loader import (
    discover_indicators,
    parse_sfa_coefficients,
    load_historical_data_by_iso3,
    build_starting_point,
    load_indicator_metadata,
    iso3_to_country_name
)

# =====================================================
# MINIMAX OPTIMIZATION
# =====================================================

def optimize_country_year_minimax(
    iso3,
    year,
    cv_year,
    data_file,
    coef_dir,
    metadata_file,
    previous_expenditures=None
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
    N = len(OPT_VARS)

    # --------------------------------------------------
    # Load historical data
    # --------------------------------------------------
    CONTROL_VALUES, HIST_INPUTS, _ = load_historical_data_by_iso3(
        data_file, iso3, year, cv_year, OPT_VARS
    )

    if previous_expenditures is not None:
        X0 = np.array([previous_expenditures[v] for v in OPT_VARS])
        HIST_TOTAL = np.sum(X0)
    else:
        X0 = np.array(build_starting_point(OPT_VARS, HIST_INPUTS))
        HIST_TOTAL = np.sum(X0)

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------
    meta_all = load_indicator_metadata(metadata_file)
    for ind in indicators:
        indicators[ind]["meta"] = meta_all.get(ind, {})

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
    # Build translog indicators
    # --------------------------------------------------
    def build_indicator(coef, meta):
        idx = {v: i for i, v in enumerate(OPT_VARS)}
        u = inefficiency(coef)
        direction = meta.get("direction")

        def I(X):
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
            if direction == "lowerOrEqual":
                y += u
                return -y        
            else:
                y -= u
                return y

        return I, jacobian(I)

    indicators_signed = {
        ind: build_indicator(indicators[ind], indicators[ind]["meta"])
        for ind in indicators
    }

    # --------------------------------------------------
    # VARIABLES : [X , t]
    # --------------------------------------------------
    def unpack(z):
        return z[:-1], z[-1]

    z0 = np.concatenate([X0, [0.0]])

    # --------------------------------------------------
    # OBJECTIVE: minimize -t  (≡ maximize t)
    # --------------------------------------------------
    def objective(z):
        _, t = unpack(z)
        return -t

    # --------------------------------------------------
    # CONSTRAINTS
    # --------------------------------------------------
    constraints = []

    # Indicator constraints: I_k(X) ≥ t
    for I, dI in indicators_signed.values():

        def c_fun(z, I=I):
            X, t = unpack(z)
            return I(X) - t

        constraints.append(
            NonlinearConstraint(c_fun, 0.0, np.inf)
        )

    # Macro GDP constraint
    constraints.append(
        NonlinearConstraint(
            lambda z: np.sum(unpack(z)[0]),
            lb=HIST_TOTAL,
            ub=np.inf
        )
    )

    bounds = [(1e-6, None)] * N + [(None, None)]

    # --------------------------------------------------
    # Solve
    # --------------------------------------------------
    res = minimize(
        objective,
        z0,
        method="trust-constr",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000}
    )

    X_opt, t_opt = unpack(res.x)

    return {
        "iso3": iso3,
        "year": year + 1,
        "min_indicator_value": t_opt,
        "expenditures": {OPT_VARS[i]: X_opt[i] for i in range(N)},
        "indicators": {
            ind: float(np.exp(I(X_opt))) if indicators[ind]["meta"]["direction"] != "lowerOrEqual"
            else float(np.exp(-I(X_opt)))
            for ind, (I, _) in indicators_signed.items()
        }
    }
