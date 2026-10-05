"""Runs the full research design and writes the result tables.

Pre-registered design (fixed before any return data was looked at, see METHODOLOGY.md):
- Primary signals: full-document cosine similarity (the paper's headline construction)
  and Item 1A (Risk Factors) cosine similarity (the paper's strongest section).
- Primary specification: quintile sort, equal-weighted, 12-month hold, Q5 minus Q1,
  alpha against Fama-French five factors plus momentum with Newey-West errors.
- Everything else (other measures, sections, holding periods) is robustness, reported
  in full so that nothing is cherry-picked.
"""

from __future__ import annotations

import pandas as pd

from . import attribution, backtest, config, factors, prices
from .similarity import MEASURES, SECTION_NAMES

PRIMARY_SIGNALS = ["full_cosine", "item1a_cosine"]
SEMANTIC_SIGNALS = ["item1a_semantic", "item7_semantic"]

# The paper's sample ends in 2014. Returns after that are out of sample for the
# original result, which is the cleanest test of whether the effect is still there.
SAMPLES = {
    "Full sample": (None, None),
    "Through 2014 (inside the paper's sample)": (None, "2014-12-31"),
    "2015 onward (out of sample)": ("2015-01-01", None),
}

PORTFOLIOS = config.PROCESSED / "portfolio_returns.parquet"
PERFORMANCE = config.PROCESSED / "performance.csv"
ATTRIBUTION = config.PROCESSED / "attribution.csv"
ROBUSTNESS = config.PROCESSED / "robustness.csv"
QUINTILES = config.PROCESSED / "quintile_means.csv"


def signal_columns(sim: pd.DataFrame) -> list[str]:
    classic = [f"{s}_{m}" for s in SECTION_NAMES for m in MEASURES]
    return [c for c in classic + SEMANTIC_SIGNALS if c in sim.columns and sim[c].notna().sum() > 0]


def _slice(series: pd.Series | pd.DataFrame, start: str | None, end: str | None):
    return series.loc[start:end]


def run(hold_months: int = backtest.DEFAULT_HOLD_MONTHS, cost_bps: float = backtest.DEFAULT_COST_BPS) -> dict:
    sim = pd.read_parquet(config.SIMILARITY)
    returns = prices.load_monthly_returns()
    fac = factors.load_factors()

    ports, perf_rows, attr_rows, robust_rows, quint_rows = [], [], [], [], []
    for column in signal_columns(sim):
        port = backtest.run_sort(sim, column, returns, hold_months, cost_bps)
        ports.append(port.assign(signal=column).reset_index())
        for sample, (start, end) in SAMPLES.items():
            p = _slice(port, start, end)
            base = {"signal": column, "sample": sample}
            for leg in ("LS", "LS_net"):
                perf_rows.append({**base, "series": leg, **backtest.performance(p[leg])})
            quint_rows.append(
                {**base, **{q: p[q].mean() * 1e4 for q in ("Q1", "Q2", "Q3", "Q4", "Q5")},
                 "avg_names": p["n_names"].mean(),
                 "avg_monthly_turnover": (p["turnover_long"] + p["turnover_short"]).mean() / 2}
            )
            for model in attribution.MODELS:
                attr_rows.append({**base, "series": "LS", **attribution.factor_regression(p["LS"], fac, model)})
            attr_rows.append({**base, "series": "LS_net", **attribution.factor_regression(p["LS_net"], fac, "FF5+MOM")})
            for leg in ("Q1", "Q5"):
                attr_rows.append(
                    {**base, "series": leg, **attribution.factor_regression(p[leg], fac, "FF5+MOM", subtract_rf=True)}
                )
        if column in PRIMARY_SIGNALS:
            for hold in (3, 6, 12):
                alt = backtest.run_sort(sim, column, returns, hold, cost_bps)
                res = attribution.factor_regression(alt["LS"], fac, "FF5+MOM")
                robust_rows.append(
                    {"signal": column, "hold_months": hold, **backtest.performance(alt["LS"]),
                     "alpha_monthly_bps": res.get("alpha_monthly_bps"), "alpha_t": res.get("alpha_t"),
                     "avg_names": alt["n_names"].mean()}
                )

    out = {
        "portfolios": pd.concat(ports, ignore_index=True),
        "performance": pd.DataFrame(perf_rows),
        "attribution": pd.DataFrame(attr_rows),
        "robustness": pd.DataFrame(robust_rows),
        "quintiles": pd.DataFrame(quint_rows),
    }
    out["portfolios"].to_parquet(PORTFOLIOS, index=False)
    out["performance"].to_csv(PERFORMANCE, index=False)
    out["attribution"].to_csv(ATTRIBUTION, index=False)
    out["robustness"].to_csv(ROBUSTNESS, index=False)
    out["quintiles"].to_csv(QUINTILES, index=False)
    return out
