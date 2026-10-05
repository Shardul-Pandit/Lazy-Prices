"""Monthly total returns from Yahoo Finance (split- and dividend-adjusted closes).

Free data has limits that matter for a backtest and are documented, not hidden:
- Only currently listed tickers are available, which is the survivorship bias.
- Adjusted closes are revised after the fact when dividends are paid, so tiny
  differences from a point-in-time vendor (CRSP) are expected.
"""

from __future__ import annotations

import time

import pandas as pd
from tqdm import tqdm

from . import config

CHUNK = 40
PRICE_START = "2004-12-01"


def _download_chunk(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf

    last_error: Exception | None = None
    for attempt in range(4):
        try:
            data = yf.download(
                tickers, start=start, auto_adjust=True, progress=False, threads=True, group_by="column"
            )
            close = data["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(tickers[0])
            close = close.dropna(how="all")
            if not close.empty:
                return close
        except Exception as exc:  # yfinance raises many types when Yahoo throttles
            last_error = exc
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Yahoo download failed for {tickers[:3]}...: {last_error}")


def download_daily(universe: pd.DataFrame, start: str = PRICE_START, force: bool = False) -> pd.DataFrame:
    """Daily adjusted closes, cached per chunk so a failed run resumes where it stopped."""
    config.PRICES_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    yahoo = list(universe["yahoo_ticker"])
    frames = []
    for i in tqdm(range(0, len(yahoo), CHUNK), desc="Prices"):
        path = config.PRICES_DAILY_DIR / f"chunk_{i // CHUNK:03d}.parquet"
        if path.exists() and not force:
            frames.append(pd.read_parquet(path))
            continue
        close = _download_chunk(yahoo[i : i + CHUNK], start)
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close.to_parquet(path)
        frames.append(close)
        time.sleep(1.0)
    wide = pd.concat(frames, axis=1).sort_index()
    back = dict(zip(universe["yahoo_ticker"], universe["ticker"], strict=True))
    return wide.rename(columns=back)


def to_monthly_returns(daily_close: pd.DataFrame) -> pd.DataFrame:
    """Month-end to month-end simple returns. An unfinished last month is dropped."""
    daily_close = daily_close.sort_index()
    monthly = daily_close.resample("ME").last()
    last_day = daily_close.index.max()
    last_business_day = last_day + pd.offsets.BMonthEnd(0)
    if last_day < last_business_day:  # the final month has not ended yet
        monthly = monthly.iloc[:-1]
    returns = monthly.pct_change(fill_method=None)
    return returns.iloc[1:].dropna(how="all", axis=1)


def build_monthly_returns(universe: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    returns = to_monthly_returns(download_daily(universe, force=force))
    long = (
        returns.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="ret")
        .dropna(subset=["ret"])
    )
    config.MONTHLY_RETURNS.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(config.MONTHLY_RETURNS, index=False)
    missing = sorted(set(universe["ticker"]) - set(returns.columns))
    if missing:
        print(f"No price history for {len(missing)} tickers: {', '.join(missing[:15])}")
    return returns


def load_monthly_returns() -> pd.DataFrame:
    long = pd.read_parquet(config.MONTHLY_RETURNS)
    return long.pivot(index="date", columns="ticker", values="ret").sort_index()
