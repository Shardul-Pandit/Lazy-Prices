"""SEC EDGAR access: a polite HTTP client, the 10-K filing index, and raw downloads.

Design notes
- EDGAR requires a declared User-Agent (name + email) and caps clients at 10 req/s.
  Every request goes through one shared rate limiter, including across threads.
- The index comes from the official submissions API (data.sec.gov), which gives the
  filing date, period of report and primary document for every filing a company made.
- Raw filings are cached gzipped on disk. Re-running only fetches what is missing, and
  the parser can be improved later without downloading anything again.
"""

from __future__ import annotations

import gzip
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from . import config

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"

# Original annual reports only. 10-K405 is the pre-2003 variant of the 10-K.
# Amendments (10-K/A) are excluded: they are often partial and would be compared
# against a full document. Transition-period reports (10-KT) cover odd-length periods.
ANNUAL_FORMS = ("10-K", "10-K405")

_INDEX_COLUMNS = [
    "accessionNumber",
    "filingDate",
    "reportDate",
    "acceptanceDateTime",
    "form",
    "primaryDocument",
]


class SecError(RuntimeError):
    """A request to EDGAR failed after retries."""


class NotFound(SecError):
    """EDGAR returned 404 for the requested resource."""


class RateLimiter:
    """Spaces calls at least 1/rate seconds apart across all threads."""

    def __init__(self, rate_per_second: float):
        self.min_interval = 1.0 / rate_per_second
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._next_slot > now:
                time.sleep(self._next_slot - now)
                now = time.monotonic()
            self._next_slot = max(now, self._next_slot) + self.min_interval

    def pause(self, seconds: float) -> None:
        """Push the next allowed request into the future for every thread."""
        with self._lock:
            self._next_slot = max(self._next_slot, time.monotonic() + seconds)


class SecClient:
    """Thin wrapper around requests with EDGAR's access rules built in."""

    def __init__(
        self,
        user_agent: str | None = None,
        rate_per_second: float = config.SEC_REQUESTS_PER_SECOND,
        session: requests.Session | None = None,
    ):
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent or config.sec_user_agent(),
                "Accept-Encoding": "gzip, deflate",
            }
        )
        self.limiter = RateLimiter(rate_per_second)

    def get(self, url: str, max_tries: int = 6) -> bytes:
        last_error: Exception | None = None
        for attempt in range(max_tries):
            self.limiter.wait()
            try:
                resp = self.session.get(url, timeout=60)
            except requests.RequestException as exc:  # network hiccup: back off and retry
                last_error = exc
                time.sleep(min(2**attempt, 30))
                continue
            status = resp.status_code
            if status == 200:
                return resp.content
            if status == 404:
                raise NotFound(url)
            last_error = SecError(f"HTTP {status} for {url}")
            if status in (403, 429):
                # EDGAR answers with 403/429 when it thinks a client is too fast.
                # Slow every thread down, not just this one.
                self.limiter.pause(min(20 * (attempt + 1), 120))
            elif status >= 500:
                time.sleep(min(2**attempt, 30))
            else:
                break
        raise SecError(f"Gave up on {url}: {last_error}")

    def get_json(self, url: str) -> dict:
        return json.loads(self.get(url))


def _block_to_frame(block: dict) -> pd.DataFrame:
    """The submissions API stores filings column-wise; turn one block into rows."""
    n = len(block.get("accessionNumber", []))
    return pd.DataFrame({c: block.get(c) or [""] * n for c in _INDEX_COLUMNS})


def filing_url(cik: int, accession: str, primary_document: str) -> str:
    """URL of the main 10-K document. Falls back to the full-submission text file."""
    doc = primary_document or f"{accession}.txt"
    return ARCHIVE_URL.format(cik=int(cik), acc_nodash=accession.replace("-", ""), doc=doc)


def company_annual_filings(client: SecClient, cik: int, start_date: str) -> pd.DataFrame:
    """All original 10-K filings for one company filed on or after start_date."""
    data = client.get_json(SUBMISSIONS_URL.format(cik=int(cik)))
    filings = data.get("filings", {})
    frames = [_block_to_frame(filings.get("recent", {}))]
    # Companies with long histories have older filings split into extra files.
    for extra in filings.get("files", []):
        if extra.get("filingTo", "9999") < start_date:
            continue
        frames.append(_block_to_frame(client.get_json(SUBMISSIONS_FILE_URL.format(name=extra["name"]))))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["form"].isin(ANNUAL_FORMS) & (df["filingDate"] >= start_date)]
    df = df.rename(
        columns={
            "accessionNumber": "accession",
            "filingDate": "filing_date",
            "reportDate": "report_date",
            "acceptanceDateTime": "acceptance_datetime",
            "primaryDocument": "primary_document",
        }
    ).drop_duplicates("accession")
    df.insert(0, "cik", int(cik))
    df["url"] = [filing_url(cik, a, d) for a, d in zip(df["accession"], df["primary_document"], strict=True)]
    return df.sort_values("filing_date").reset_index(drop=True)


def build_index(
    universe: pd.DataFrame,
    start_date: str = config.DEFAULT_START_DATE,
    client: SecClient | None = None,
    workers: int = 4,
) -> pd.DataFrame:
    """Filing index for every company in the universe. Saved to FILING_INDEX."""
    client = client or SecClient()
    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []

    def one(row) -> pd.DataFrame:
        df = company_annual_filings(client, row.cik, start_date)
        df.insert(1, "ticker", row.ticker)
        return df

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, row): row for row in universe.itertuples()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Filing index"):
            row = futures[fut]
            try:
                frames.append(fut.result())
            except SecError as exc:
                failures.append((row.ticker, str(exc)))

    index = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not index.empty:
        index["filing_date"] = pd.to_datetime(index["filing_date"])
        index["report_date"] = pd.to_datetime(index["report_date"], errors="coerce")
        index = index.sort_values(["cik", "filing_date"]).reset_index(drop=True)
        config.FILING_INDEX.parent.mkdir(parents=True, exist_ok=True)
        index.to_parquet(config.FILING_INDEX, index=False)
    for ticker, err in failures:
        print(f"  index failed for {ticker}: {err}")
    return index


def raw_path(cik: int, accession: str) -> Path:
    return config.RAW_FILINGS / str(int(cik)) / f"{accession}.html.gz"


def _download_one(client: SecClient, cik: int, accession: str, url: str) -> str:
    path = raw_path(cik, accession)
    if path.exists():
        return "cached"
    content = client.get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp name, then rename: an interrupted run never leaves a half file
    # that a later run would mistake for a finished download.
    tmp = path.with_name(path.name + ".part")
    with gzip.open(tmp, "wb", compresslevel=6) as fh:
        fh.write(content)
    os.replace(tmp, path)
    return "downloaded"


def download_filings(
    index: pd.DataFrame, client: SecClient | None = None, workers: int = 6
) -> pd.DataFrame:
    """Download every filing in the index that is not on disk yet. Safe to re-run."""
    todo = index[[not raw_path(c, a).exists() for c, a in zip(index["cik"], index["accession"], strict=True)]]
    print(f"{len(index) - len(todo)} filings already on disk, {len(todo)} to download")
    if todo.empty:
        return pd.DataFrame(columns=["cik", "accession", "status"])
    client = client or SecClient()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, client, r.cik, r.accession, r.url): r for r in todo.itertuples()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading 10-Ks"):
            r = futures[fut]
            try:
                status = fut.result()
            except SecError as exc:
                status = f"failed: {exc}"
            results.append((r.cik, r.accession, status))
    out = pd.DataFrame(results, columns=["cik", "accession", "status"])
    failed = out[out["status"].str.startswith("failed")]
    if not failed.empty:
        print(f"{len(failed)} downloads failed. Re-run the command to retry them.")
        failed.to_csv(config.INTERIM / "download_failures.csv", index=False)
    return out
