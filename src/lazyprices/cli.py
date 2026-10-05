"""Command line entry point. Run `uv run lp --help` to see every stage."""

from __future__ import annotations

import pandas as pd
import typer

from . import config

app = typer.Typer(add_completion=False, help="Lazy Prices pipeline: EDGAR -> text -> signal -> backtest.")

TickersOpt = typer.Option(None, "--tickers", help="Comma-separated subset, e.g. AAPL,JPM,XOM")
StartOpt = typer.Option(config.DEFAULT_START_DATE, "--start", help="Earliest filing date")


def _universe(tickers: str | None) -> pd.DataFrame:
    from .universe import load_universe

    return load_universe(tickers.split(",") if tickers else None)


@app.command()
def index(tickers: str = TickersOpt, start: str = StartOpt) -> None:
    """Build the 10-K filing index from the SEC submissions API."""
    from . import edgar

    config.ensure_dirs()
    uni = _universe(tickers)
    idx = edgar.build_index(uni, start_date=start)
    print(f"{len(idx)} filings indexed for {idx['cik'].nunique()} of {len(uni)} companies")


@app.command()
def download(tickers: str = TickersOpt) -> None:
    """Download the raw 10-K documents listed in the index (resumable)."""
    from . import edgar

    config.ensure_dirs()
    idx = pd.read_parquet(config.FILING_INDEX)
    if tickers:
        idx = idx[idx["ticker"].isin(_universe(tickers)["ticker"])]
    out = edgar.download_filings(idx)
    if len(out):
        print(out["status"].str.split(":").str[0].value_counts().to_string())


@app.command()
def ingest(tickers: str = TickersOpt, start: str = StartOpt) -> None:
    """index + download in one go."""
    index(tickers=tickers, start=start)
    download(tickers=tickers)


@app.command()
def parse(force: bool = typer.Option(False, help="Re-parse everything")) -> None:
    """Extract clean text, Item 1A and Item 7 from every downloaded filing."""
    from . import parse as parse_mod

    config.ensure_dirs()
    idx = pd.read_parquet(config.FILING_INDEX)
    qa = parse_mod.parse_all(idx, force=force)
    print(parse_mod.qa_summary(qa, idx).round(3).to_string())


@app.command()
def similarity() -> None:
    """Compute year-over-year similarity for consecutive 10-Ks."""
    from . import similarity as sim_mod

    idx = pd.read_parquet(config.FILING_INDEX)
    qa = pd.read_parquet(config.PARSE_QA)
    out = sim_mod.build_similarity(idx, qa)
    print(f"{len(out)} filing pairs for {out['cik'].nunique()} companies")
    print(out[[c for c in out.columns if c.endswith(("cosine", "jaccard", "edit"))]].describe().round(3).to_string())


@app.command()
def market(force: bool = typer.Option(False, help="Re-download prices")) -> None:
    """Download monthly returns (Yahoo) and factor returns (Ken French library)."""
    from . import factors, prices

    config.ensure_dirs()
    fac = factors.build_factors()
    print(f"Factors: {fac.index.min().date()} to {fac.index.max().date()}")
    ret = prices.build_monthly_returns(_universe(None), force=force)
    print(f"Returns: {ret.shape[1]} tickers, {ret.index.min().date()} to {ret.index.max().date()}")


@app.command()
def report() -> None:
    """Run the backtest and factor attribution, then write figures and RESULTS.md."""
    from . import report as report_mod

    config.ensure_dirs()
    report_mod.build()


@app.command()
def semantic() -> None:
    """Optional extension: embedding-based similarity (needs --extra semantic)."""
    from . import semantic as semantic_mod

    out = semantic_mod.build_semantic()
    cols = [c for c in out.columns if c.endswith(("semantic", "novel_share"))]
    print(out[cols].describe().round(3).to_string())


@app.command()
def analyze() -> None:
    """parse + similarity + market + report: everything after the download."""
    parse()
    similarity()
    market()
    report()


if __name__ == "__main__":
    app()
