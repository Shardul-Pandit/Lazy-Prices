"""Fama-French five factors plus momentum, from the Ken French Data Library.

The library files are zipped CSVs with free-text headers and a monthly block followed
by an annual block. Values are in percent. This module reads only the monthly block.
"""

from __future__ import annotations

import io
import re
import zipfile

import pandas as pd
import requests

from . import config

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {"ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip", "mom": "F-F_Momentum_Factor_CSV.zip"}
FACTOR_COLUMNS = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]

_MONTH_ROW = re.compile(r"^\s*(\d{6})\s*,")


def parse_french_csv(text: str) -> pd.DataFrame:
    """Monthly block of a Ken French CSV as decimals, indexed by month-end date."""
    lines = text.splitlines()
    first = next((i for i, line in enumerate(lines) if _MONTH_ROW.match(line)), None)
    if first is None or first == 0:
        raise ValueError("No monthly data block found in Ken French file")
    header = [h.strip() for h in lines[first - 1].split(",")][1:]
    rows = []
    for line in lines[first:]:
        if not _MONTH_ROW.match(line):
            break  # the monthly block ends at the first non-monthly line
        parts = [p.strip() for p in line.split(",")]
        rows.append([parts[0]] + [float(p) for p in parts[1 : len(header) + 1]])
    df = pd.DataFrame(rows, columns=["yyyymm", *header])
    df.index = pd.to_datetime(df.pop("yyyymm"), format="%Y%m") + pd.offsets.MonthEnd(0)
    df.index.name = "date"
    df = df.where(df > -99)  # -99.99 / -999 are the library's missing-value codes
    return df / 100.0


def _read_zip(name: str) -> str:
    local = config.REFERENCE / name
    if local.exists():  # manual fallback: drop the zip into data/reference
        payload = local.read_bytes()
    else:
        resp = requests.get(BASE + name, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        payload = resp.content
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        return zf.read(zf.namelist()[0]).decode("latin-1")


def build_factors() -> pd.DataFrame:
    ff5 = parse_french_csv(_read_zip(FILES["ff5"])).rename(columns={"Mkt-RF": "MKT_RF"})
    mom = parse_french_csv(_read_zip(FILES["mom"]))
    mom.columns = ["MOM"]
    factors = ff5.join(mom, how="inner")[[*FACTOR_COLUMNS, "RF"]]
    config.FACTORS.parent.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(config.FACTORS)
    return factors


def load_factors() -> pd.DataFrame:
    return pd.read_parquet(config.FACTORS)
