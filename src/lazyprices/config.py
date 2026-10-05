"""Paths and constants shared by every stage of the pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root: src/lazyprices/config.py -> two levels up. Can be overridden for tests.
ROOT = Path(os.environ.get("LAZYPRICES_ROOT", Path(__file__).resolve().parents[2]))
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
REFERENCE = DATA / "reference"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

UNIVERSE_CSV = REFERENCE / "sp500_constituents.csv"
RAW_FILINGS = RAW / "filings"  # {cik}/{accession}.html.gz
FILING_INDEX = INTERIM / "filing_index.parquet"
TEXT_DIR = INTERIM / "text"  # {cik}/{accession}.json.gz
PARSE_QA = PROCESSED / "parse_qa.parquet"
SIMILARITY = PROCESSED / "similarity.parquet"
PRICES_DAILY_DIR = INTERIM / "prices_daily"
MONTHLY_RETURNS = PROCESSED / "monthly_returns.parquet"
FACTORS = PROCESSED / "factors.parquet"

# First filing date pulled from EDGAR. Item 1A (Risk Factors) only became a required
# 10-K item for fiscal years ending after 2005-12-01, so earlier filings add little.
DEFAULT_START_DATE = "2005-01-01"

# SEC fair-access policy caps automated clients at 10 requests/second. Stay under it.
SEC_REQUESTS_PER_SECOND = 8.0


def sec_user_agent() -> str:
    """Return the declared User-Agent, or fail with instructions if it is missing."""
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua or len(ua.split()) < 2:
        raise RuntimeError(
            "SEC_USER_AGENT is not set. SEC EDGAR requires a name and contact email on "
            'every request. Put a line like SEC_USER_AGENT="Jane Doe jane@example.com" '
            "in the .env file at the project root."
        )
    return ua


def ensure_dirs() -> None:
    for p in (RAW_FILINGS, INTERIM, TEXT_DIR, PROCESSED, FIGURES, PRICES_DAILY_DIR):
        p.mkdir(parents=True, exist_ok=True)
