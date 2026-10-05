"""Stock universe: S&P 500 constituents (snapshot vendored in data/reference).

Known limitation, stated up front: this is the index membership as of the snapshot
date, not point-in-time membership. Firms that were delisted, acquired or dropped from
the index before the snapshot are missing, so the backtest has survivorship bias.
See docs/METHODOLOGY.md for how that is handled and what it can and cannot affect.
"""

from __future__ import annotations

import pandas as pd

from . import config


def load_universe(tickers: list[str] | None = None) -> pd.DataFrame:
    """Return one row per company: ticker, yahoo_ticker, name, sector, cik.

    Companies with two listed share classes (GOOGL/GOOG, FOXA/FOX, NWSA/NWS) file a
    single 10-K under one CIK, so they are collapsed to the first listed class.
    """
    df = pd.read_csv(config.UNIVERSE_CSV)
    df = df.rename(
        columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector", "CIK": "cik"}
    )[["ticker", "name", "sector", "cik"]]
    df["cik"] = df["cik"].astype(int)
    df = df.drop_duplicates("cik", keep="first").reset_index(drop=True)
    # Yahoo Finance writes share classes with a dash (BRK-B), S&P with a dot (BRK.B).
    df["yahoo_ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    if tickers:
        wanted = {t.strip().upper() for t in tickers}
        df = df[df["ticker"].isin(wanted)].reset_index(drop=True)
        missing = wanted - set(df["ticker"])
        if missing:
            raise ValueError(f"Tickers not in the universe file: {sorted(missing)}")
    return df
