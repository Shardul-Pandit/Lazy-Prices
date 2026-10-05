"""Year-over-year text similarity between a company's consecutive 10-Ks.

Three of the four measures from Cohen, Malloy and Nguyen (2020), computed on words:
- cosine: angle between the two term-frequency vectors (which words, how often)
- jaccard: overlap of the two vocabularies, ignoring frequency
- edit:    1 - (word-level Levenshtein distance / length of the longer document),
           which is sensitive to word order and to inserted or deleted passages

All three are in [0, 1]; 1 means the text is unchanged. LOW similarity = a "changer".
The paper's fourth measure (word-processor "track changes" counts) is not replicated.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

from . import config
from .parse import load_text

SECTION_NAMES = ("full", "item1a", "item7")
MEASURES = ("cosine", "jaccard", "edit")

# Consecutive fiscal years: period-of-report dates roughly one year apart. Anything
# else (a skipped year, a fiscal-year change) is not a like-for-like comparison.
MIN_GAP_DAYS, MAX_GAP_DAYS = 300, 430

_TOKEN = re.compile(r"[a-z]{2,}")


def tokenize(text: str) -> list[str]:
    """Lowercase alphabetic words of 2+ letters. Numbers are dropped on purpose:
    updated figures would otherwise register as 'changed language' every year."""
    return _TOKEN.findall(text.lower())


def cosine_similarity(a: Counter, b: Counter) -> float:
    if not a or not b:
        return float("nan")
    if len(a) > len(b):
        a, b = b, a
    dot = sum(count * b.get(word, 0) for word, count in a.items())
    norm = math.sqrt(sum(c * c for c in a.values())) * math.sqrt(sum(c * c for c in b.values()))
    return dot / norm


def jaccard_similarity(a: set, b: set) -> float:
    if not a or not b:
        return float("nan")
    return len(a & b) / len(a | b)


def edit_similarity(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return float("nan")
    # Map words to integers once so the distance works on small ints, not strings.
    vocab: dict[str, int] = {}
    ia = [vocab.setdefault(w, len(vocab)) for w in a]
    ib = [vocab.setdefault(w, len(vocab)) for w in b]
    return Levenshtein.normalized_similarity(ia, ib)


def compare(prev_tokens: list[str], cur_tokens: list[str]) -> dict[str, float]:
    return {
        "cosine": cosine_similarity(Counter(prev_tokens), Counter(cur_tokens)),
        "jaccard": jaccard_similarity(set(prev_tokens), set(cur_tokens)),
        "edit": edit_similarity(prev_tokens, cur_tokens),
    }


def consecutive_pairs(filings: pd.DataFrame) -> list[tuple[int, int]]:
    """Index pairs (previous, current) of filings for consecutive fiscal years.

    `filings` is one company's 10-Ks. Uses the period of report when available and
    the filing date otherwise. Duplicate periods keep the earliest filing.
    """
    df = filings.copy()
    df["period"] = df["report_date"].fillna(df["filing_date"])
    df = df.sort_values(["period", "filing_date"]).drop_duplicates("period", keep="first")
    idx = list(df.index)
    pairs = []
    for prev, cur in zip(idx[:-1], idx[1:], strict=True):
        gap = (df.at[cur, "period"] - df.at[prev, "period"]).days
        if MIN_GAP_DAYS <= gap <= MAX_GAP_DAYS:
            pairs.append((prev, cur))
    return pairs


def _firm_similarity(filings: pd.DataFrame) -> list[dict]:
    rows = []
    tokens_cache: dict[str, dict[str, list[str]]] = {}

    def tokens_for(i) -> dict[str, list[str]]:
        acc = filings.at[i, "accession"]
        if acc not in tokens_cache:
            text = load_text(filings.at[i, "cik"], acc)
            tokens_cache[acc] = {
                s: tokenize(text[s]) if bool(filings.at[i, f"{s}_ok"]) else [] for s in SECTION_NAMES
            }
        return tokens_cache[acc]

    for prev, cur in consecutive_pairs(filings):
        row = {
            "cik": int(filings.at[cur, "cik"]),
            "ticker": filings.at[cur, "ticker"],
            "accession": filings.at[cur, "accession"],
            "prev_accession": filings.at[prev, "accession"],
            "filing_date": filings.at[cur, "filing_date"],
            "report_date": filings.at[cur, "report_date"],
            "prev_filing_date": filings.at[prev, "filing_date"],
        }
        try:
            t_prev, t_cur = tokens_for(prev), tokens_for(cur)
        except (OSError, ValueError, KeyError):
            continue
        for s in SECTION_NAMES:
            a, b = t_prev[s], t_cur[s]
            row[f"{s}_words"] = len(b)
            row[f"{s}_prev_words"] = len(a)
            scores = compare(a, b) if a and b else dict.fromkeys(MEASURES, float("nan"))
            for m in MEASURES:
                row[f"{s}_{m}"] = scores[m]
        rows.append(row)
        tokens_cache.pop(filings.at[prev, "accession"], None)  # keep memory flat
    return rows


def build_similarity(index: pd.DataFrame, qa: pd.DataFrame, workers: int | None = None) -> pd.DataFrame:
    """Similarity panel: one row per (company, fiscal year) with a valid prior year."""
    ok_cols = ["accession", "full_ok", "item1a_ok", "item7_ok"]
    df = index.merge(qa.loc[qa["error"].fillna("") == "", ok_cols], on="accession", how="inner")
    groups = [g.reset_index(drop=True) for _, g in df.groupby("cik") if len(g) >= 2]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for firm_rows in tqdm(pool.map(_firm_similarity, groups, chunksize=2), total=len(groups), desc="Similarity"):
            rows.extend(firm_rows)
    out = pd.DataFrame(rows).sort_values(["cik", "filing_date"]).reset_index(drop=True)
    config.SIMILARITY.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.SIMILARITY, index=False)
    return out
