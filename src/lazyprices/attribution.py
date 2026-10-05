"""Factor attribution: is the long-short return new, or known factors in disguise?

Regress the strategy's monthly return on standard factor returns. The intercept
(alpha) is the part the factors cannot explain. Standard errors are Newey-West (HAC),
which stay valid when monthly returns are autocorrelated or change in volatility.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from .backtest import HAC_LAGS

MODELS = {
    "CAPM": ["MKT_RF"],
    "FF3": ["MKT_RF", "SMB", "HML"],
    "FF5": ["MKT_RF", "SMB", "HML", "RMW", "CMA"],
    "FF5+MOM": ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"],
}


def factor_regression(
    returns: pd.Series, factors: pd.DataFrame, model: str = "FF5+MOM", subtract_rf: bool = False,
    lags: int = HAC_LAGS,
) -> dict:
    """OLS of returns on the model's factors with HAC standard errors.

    subtract_rf: True for a long-only portfolio (regress its excess return). A
    long-short spread is already self-financing, so nothing is subtracted.
    """
    cols = MODELS[model]
    df = pd.concat([returns.rename("y"), factors[[*cols, "RF"]]], axis=1, join="inner").dropna()
    if len(df) < len(cols) + lags + 2:
        return {"model": model, "months": len(df)}
    y = df["y"] - df["RF"] if subtract_rf else df["y"]
    fit = sm.OLS(y, sm.add_constant(df[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {
        "model": model,
        "months": int(fit.nobs),
        "alpha_monthly_bps": fit.params["const"] * 1e4,
        "alpha_ann": fit.params["const"] * 12,
        "alpha_t": fit.tvalues["const"],
        "alpha_p": fit.pvalues["const"],
        "r2": fit.rsquared,
    }
    for c in cols:
        out[f"beta_{c}"] = fit.params[c]
        out[f"t_{c}"] = fit.tvalues[c]
    return out
