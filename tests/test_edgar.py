import gzip
import json
import time

import pandas as pd
import pytest

from lazyprices import config, edgar


class FakeResponse:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content


class FakeSession:
    """Stands in for requests.Session: serves canned bodies and records calls."""

    def __init__(self, routes):
        self.routes = routes
        self.headers = {}
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        item = self.routes[url]
        if isinstance(item, list):  # a scripted sequence of responses
            item = item.pop(0)
        return item if isinstance(item, FakeResponse) else FakeResponse(200, item)


def _block(rows):
    cols = ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form", "primaryDocument"]
    return {c: [r[i] for r in rows] for i, c in enumerate(cols)}


RECENT = _block(
    [
        ("0000001-24-000010", "2024-02-20", "2023-12-31", "2024-02-20T16:30:00.000Z", "10-K", "a-10k.htm"),
        ("0000001-24-000005", "2024-01-15", "2024-01-10", "2024-01-15T09:00:00.000Z", "8-K", "a-8k.htm"),
        ("0000001-23-000011", "2023-03-01", "2022-12-31", "2023-03-01T16:30:00.000Z", "10-K/A", "a-10ka.htm"),
        ("0000001-23-000010", "2023-02-21", "2022-12-31", "2023-02-21T16:30:00.000Z", "10-K", "a-10k.htm"),
    ]
)
OLDER = _block(
    [
        ("0000001-06-000010", "2006-02-21", "2005-12-31", "2006-02-21T16:30:00.000Z", "10-K", ""),
        ("0000001-01-000010", "2001-03-21", "2000-12-31", "2001-03-21T16:30:00.000Z", "10-K405", "old.txt"),
    ]
)


def _client(extra_routes=None):
    routes = {
        "https://data.sec.gov/submissions/CIK0000000001.json": json.dumps(
            {
                "filings": {
                    "recent": RECENT,
                    "files": [
                        {"name": "CIK0000000001-submissions-001.json", "filingFrom": "2001-01-01", "filingTo": "2010-12-31"},
                        {"name": "CIK0000000001-submissions-002.json", "filingFrom": "1995-01-01", "filingTo": "2000-12-31"},
                    ],
                }
            }
        ).encode(),
        "https://data.sec.gov/submissions/CIK0000000001-submissions-001.json": json.dumps(OLDER).encode(),
    }
    routes.update(extra_routes or {})
    session = FakeSession(routes)
    return edgar.SecClient(user_agent="Test Runner test@example.com", rate_per_second=1000, session=session), session


def test_index_keeps_only_original_annual_reports_after_start():
    client, session = _client()
    df = edgar.company_annual_filings(client, 1, "2005-01-01")
    # 8-K and 10-K/A dropped; the 2001 10-K405 is before the start date.
    assert list(df["accession"]) == ["0000001-06-000010", "0000001-23-000010", "0000001-24-000010"]
    assert set(df["form"]) == {"10-K"}
    # The 1995-2000 file ends before the start date, so it must never be requested.
    assert not any("submissions-002" in url for url in session.calls)


def test_filing_url_and_text_fallback():
    assert (
        edgar.filing_url(320193, "0000320193-24-000123", "aapl-20240928.htm")
        == "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
    )
    # Old filings have no primary document listed: use the full submission text file.
    assert edgar.filing_url(1, "0000001-06-000010", "").endswith("/000000106000010/0000001-06-000010.txt")


def test_user_agent_header_is_sent():
    client, session = _client()
    assert "test@example.com" in session.headers["User-Agent"]


def test_missing_user_agent_is_rejected(monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "no-email-here")
    with pytest.raises(RuntimeError, match="SEC_USER_AGENT"):
        config.sec_user_agent()


def test_rate_limiter_spaces_requests():
    limiter = edgar.RateLimiter(rate_per_second=20)  # 50 ms apart
    start = time.monotonic()
    for _ in range(5):
        limiter.wait()
    assert time.monotonic() - start >= 4 * 0.05 * 0.9


def test_retries_then_succeeds_and_404_raises(monkeypatch):
    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)
    client, _ = _client(
        {
            "https://x/flaky": [edgar_resp(503), edgar_resp(200, b"ok")],
            "https://x/missing": edgar_resp(404),
        }
    )
    assert client.get("https://x/flaky") == b"ok"
    with pytest.raises(edgar.NotFound):
        client.get("https://x/missing")


def edgar_resp(status, content=b""):
    return FakeResponse(status, content)


def test_download_is_resumable(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_FILINGS", tmp_path / "filings")
    monkeypatch.setattr(config, "INTERIM", tmp_path)
    url = "https://www.sec.gov/Archives/edgar/data/1/000000124000010/a-10k.htm"
    client, session = _client({url: b"<html>hello</html>"})
    index = pd.DataFrame({"cik": [1], "accession": ["0000001-24-000010"], "url": [url]})

    first = edgar.download_filings(index, client=client, workers=1)
    assert list(first["status"]) == ["downloaded"]
    path = edgar.raw_path(1, "0000001-24-000010")
    assert gzip.open(path).read() == b"<html>hello</html>"
    assert not list(path.parent.glob("*.part"))

    calls_before = len(session.calls)
    second = edgar.download_filings(index, client=client, workers=1)
    assert second.empty and len(session.calls) == calls_before  # nothing fetched twice
