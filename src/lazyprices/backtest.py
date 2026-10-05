"""Portfolio sorts on the similarity signal, with strict point-in-time timing.

Timeline for one filing (this is the part that makes or breaks a backtest):

    filing published      formation date            return earned
    (filing_date)   -->   first month-end that  --> the following
                          is at least one           calendar month
                          business day later

A 10-K filed on 15 March is used to form the portfolio at the 31 March close and earns
April's return. A 10-K filed on the last business day of March is NOT used until the
April close, because it may have been published after that day's market close.

Each month, companies with a live signal are sorted into quintiles. Q1 holds the
lowest similarity (the "changers"), Q5 the highest ("non-changers"). The paper's
prediction is Q5 minus Q1 > 0. Portfolios are equal-weighted and rebalanced monthly.
A signal stays live for `hold_months` after formation or until the next 10-K.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

N_QUANTILES = 5
DEFAULT_HOLD_MONTHS = 12
MIN_NAMES = 50  # fewer live signals than this in a month: no portfolio that month
DEFAULT_COST_BPS = 10.0  # cost per dollar traded, one way (large-cap US equities)
HAC_LAGS = 6


def formation_month_end(filing_date: pd.Series) -> pd.Series:
    """First month-end at which a filing may be traded on (see module docstring)."""
    filing_date = pd.to_datetime(filing_date)
    business_month_end = filing_date + pd.offsets.BMonthEnd(0)  # on or after the filing
    cutoff = business_month_end - pd.offsets.BDay(1)
    usable_this_month = filing_date <= cutoff
    this_month = business_month_end + pd.offsets.MonthEnd(0)
    next_month = this_month + pd.offsets.MonthEnd(1)
    return this_month.where(usable_this_month, next_month)


def signal_panel(
    sim: pd.DataFrame, column: str, months: pd.DatetimeIndex, tickers: list[str], hold_months: int
) -> pd.DataFrame:
    """Month-end x ticker matrix holding each company's live signal value."""
    values = np.full((len(months), len(tickers)), np.nan)
    col_pos = {t: i for i, t in enumerate(tickers)}
    rows = sim.dropna(subset=[column]).sort_values("filing_date")
    rows = rows[rows["ticker"].isin(col_pos)]
    formation = formation_month_end(rows["filing_date"])
    for ticker, start, value in zip(rows["ticker"], formation, rows[column], strict=True):
        pos = months.searchsorted(start)  # first month-end >= formation date
        if pos < len(months):
            # A newer filing overwrites the older signal from its own formation month on.
            values[pos : pos + hold_months, col_pos[ticker]] = value
    return pd.DataFrame(values, index=months, columns=tickers)


def assign_quantiles(panel: pd.DataFrame, n: int = N_QUANTILES, min_names: int = MIN_NAMES) -> pd.DataFrame:
    """Cross-sectional quantile (1..n) per month; months with too few names are empty."""
    ranks = panel.rank(axis=1, pct=True, method="first")
    quantiles = np.ceil(ranks * n).clip(1, n)
    too_thin = panel.notna().sum(axis=1) < min_names
    quantiles.loc[too_thin] = np.nan
    return quantiles


def _traded_fraction(new_w: pd.Series, old_w: pd.Series | None, month_ret: pd.Series) -> float:
    """Sum of absolute weight changes when moving from last month's drifted portfolio."""
    if old_w is None or old_w.empty:
        return float(new_w.abs().sum())  # initial buy-in
    drifted = old_w * (1.0 + month_ret.reindex(old_w.index).fillna(0.0))
    drifted = drifted / drifted.sum()
    names = new_w.index.union(drifted.index)
    return float((new_w.reindex(names, fill_value=0.0) - drifted.reindex(names, fill_value=0.0)).abs().sum())


def quantile_returns(
    quantiles: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = DEFAULT_COST_BPS
) -> pd.DataFrame:
    """Equal-weighted quantile returns, long-short spread, turnover and net returns.

    Row t holds the return earned during month t by the portfolio formed at the end of
    month t-1. The signal matrix is never aligned with same-month returns.
    """
    months = quantiles.index
    n = int(np.nanmax(quantiles.to_numpy())) if quantiles.notna().any().any() else N_QUANTILES
    records = []
    prev_weights: dict[int, pd.Series | None] = {1: None, n: None}
    for i in range(len(months) - 1):
        formed, earned = months[i], months[i + 1]
        q_row = quantiles.loc[formed]
        r_next = returns.loc[earned] if earned in returns.index else None
        if r_next is None or q_row.notna().sum() == 0:
            prev_weights = {1: None, n: None}
            continue
        rec = {"date": earned, "n_names": int(q_row.notna().sum())}
        r_formed = returns.loc[formed] if formed in returns.index else pd.Series(dtype=float)
        for q in range(1, n + 1):
            members = q_row.index[(q_row == q) & r_next.notna()]
            rec[f"Q{q}"] = float(r_next[members].mean()) if len(members) else np.nan
            if q in (1, n):
                weights = pd.Series(1.0 / len(members), index=members) if len(members) else pd.Series(dtype=float)
                rec["turnover_short" if q == 1 else "turnover_long"] = _traded_fraction(
                    weights, prev_weights[q], r_formed
                )
                prev_weights[q] = weights
        records.append(rec)
    out = pd.DataFrame(records).set_index("date")
    out["LS"] = out[f"Q{n}"] - out["Q1"]
    out["LS_net"] = out["LS"] - (out["turnover_long"] + out["turnover_short"]) * cost_bps / 1e4
    return out


def newey_west_tstat(series: pd.Series, lags: int = HAC_LAGS) -> float:
    """t-statistic of the mean, robust to autocorrelation and heteroskedasticity."""
    y = series.dropna()
    if len(y) < lags + 2:
        return float("nan")
    fit = sm.OLS(y.to_numpy(), np.ones(len(y))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.tvalues[0])


def performance(series: pd.Series, lags: int = HAC_LAGS) -> dict:
    r = series.dropna()
    if len(r) < 12:
        return {"months": len(r)}
    wealth = (1.0 + r).cumprod()
    return {
        "months": len(r),
        "mean_monthly_bps": r.mean() * 1e4,
        "ann_return": r.mean() * 12,
        "ann_vol": r.std(ddof=1) * np.sqrt(12),
        "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(12),
        "t_stat": newey_west_tstat(r, lags),
        "max_drawdown": float((wealth / wealth.cummax() - 1.0).min()),
        "hit_rate": float((r > 0).mean()),
    }


def run_sort(
    sim: pd.DataFrame,
    column: str,
    returns: pd.DataFrame,
    hold_months: int = DEFAULT_HOLD_MONTHS,
    cost_bps: float = DEFAULT_COST_BPS,
    min_names: int = MIN_NAMES,
) -> pd.DataFrame:
    panel = signal_panel(sim, column, returns.index, list(returns.columns), hold_months)
    return quantile_returns(assign_quantiles(panel, min_names=min_names), returns, cost_bps)
