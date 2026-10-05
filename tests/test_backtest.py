import numpy as np
import pandas as pd

from lazyprices import attribution, backtest


def test_formation_date_never_uses_a_filing_before_it_is_public():
    filed = pd.Series(pd.to_datetime(["2024-03-15", "2024-03-28", "2024-03-29", "2024-03-30", "2024-02-29"]))
    got = list(backtest.formation_month_end(filed))
    assert got == [
        pd.Timestamp("2024-03-31"),  # mid-month: usable at the March close
        pd.Timestamp("2024-03-31"),  # one business day before the last one: usable
        pd.Timestamp("2024-04-30"),  # last business day of March: wait for April
        pd.Timestamp("2024-04-30"),  # weekend after month-end
        pd.Timestamp("2024-03-31"),  # last business day of February: wait for March
    ]


def _toy(n_tickers=100, n_months=36, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    returns = pd.DataFrame(rng.normal(0.0, 0.05, (n_months, n_tickers)), index=months, columns=tickers)
    return months, tickers, returns


def test_signal_earns_next_month_return_not_same_month():
    months, tickers, returns = _toy()
    # One filing per ticker on 2020-03-16 with signal = ticker number.
    sim = pd.DataFrame({"ticker": tickers, "filing_date": pd.Timestamp("2020-03-16"), "sig": np.arange(len(tickers), dtype=float)})
    # Plant a huge return for the high-signal names in the filing month itself (March)
    # and a different, known return in April. A look-ahead bug would pick up March.
    top = tickers[80:]
    returns.loc["2020-03-31", top] = 5.0
    returns.loc["2020-04-30", :] = 0.0
    returns.loc["2020-04-30", top] = 0.10
    out = backtest.run_sort(sim, "sig", returns, hold_months=12, cost_bps=0.0)
    assert out.index[0] == pd.Timestamp("2020-04-30")  # first return month is April
    assert np.isclose(out.loc["2020-04-30", "Q5"], 0.10) and np.isclose(out.loc["2020-04-30", "Q1"], 0.0)
    assert np.isclose(out.loc["2020-04-30", "LS"], 0.10)
    # hold_months=12: formed at 12 month-ends (Mar 2020..Feb 2021), earning Apr 2020..Mar 2021.
    assert out.index[-1] == pd.Timestamp("2021-03-31") and len(out) == 12


def test_newer_filing_replaces_older_signal_and_quantiles_are_balanced():
    months, tickers, returns = _toy()
    old = pd.DataFrame({"ticker": tickers, "filing_date": pd.Timestamp("2020-03-16"), "sig": np.arange(100.0)})
    new = pd.DataFrame({"ticker": tickers, "filing_date": pd.Timestamp("2021-03-15"), "sig": np.arange(100.0)[::-1]})
    sim = pd.concat([old, new])
    panel = backtest.signal_panel(sim, "sig", months, tickers, hold_months=12)
    assert panel.loc["2021-02-28", "T000"] == 0.0 and panel.loc["2021-03-31", "T000"] == 99.0
    q = backtest.assign_quantiles(panel)
    assert (q.loc["2020-06-30"].value_counts() == 20).all()
    assert q.loc["2020-01-31"].isna().all()  # no signal before the first filing


def test_thin_months_are_skipped():
    months, tickers, returns = _toy()
    sim = pd.DataFrame({"ticker": tickers[:10], "filing_date": pd.Timestamp("2020-03-16"), "sig": np.arange(10.0)})
    panel = backtest.signal_panel(sim, "sig", months, tickers, hold_months=12)
    assert backtest.assign_quantiles(panel, min_names=50).isna().all().all()


def test_turnover_and_costs():
    months, tickers, returns = _toy()
    returns.loc[:, :] = 0.0  # no drift, so holding the same names means zero trading
    sim = pd.DataFrame({"ticker": tickers, "filing_date": pd.Timestamp("2020-03-16"), "sig": np.arange(100.0)})
    out = backtest.run_sort(sim, "sig", returns, hold_months=12, cost_bps=10.0)
    first, later = out.iloc[0], out.iloc[1]
    assert np.isclose(first["turnover_long"], 1.0) and np.isclose(first["turnover_short"], 1.0)
    assert np.isclose(first["LS_net"], first["LS"] - 2 * 10 / 1e4)  # buy-in of both legs
    assert np.isclose(later["turnover_long"], 0.0) and np.isclose(later["LS_net"], later["LS"])


def test_factor_regression_recovers_known_alpha_and_beta():
    rng = np.random.default_rng(1)
    months = pd.date_range("2005-01-31", periods=240, freq="ME")
    f = pd.DataFrame(rng.normal(0, 0.03, (240, 6)), index=months, columns=["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"])
    f["RF"] = 0.002
    y = 0.004 + 0.5 * f["MKT_RF"] - 0.3 * f["HML"] + rng.normal(0, 0.002, 240)
    res = attribution.factor_regression(y, f, "FF5+MOM")
    assert abs(res["alpha_monthly_bps"] - 40) < 5 and abs(res["beta_MKT_RF"] - 0.5) < 0.02
    assert abs(res["beta_HML"] + 0.3) < 0.02 and res["alpha_t"] > 5
    # A long-only portfolio is regressed in excess of the risk-free rate.
    res_long = attribution.factor_regression(y + f["RF"], f, "FF5+MOM", subtract_rf=True)
    assert abs(res_long["alpha_monthly_bps"] - 40) < 5


def test_performance_stats():
    r = pd.Series([0.01, -0.02, 0.03] * 8)
    stats = backtest.performance(r)
    assert stats["months"] == 24 and np.isclose(stats["ann_return"], r.mean() * 12)
    assert np.isclose(stats["hit_rate"], 2 / 3) and stats["max_drawdown"] < 0
