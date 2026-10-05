"""Extension beyond the paper: semantic similarity from sentence embeddings.

Why: the paper's measures count words. They cannot tell a paragraph that was reworded
from a paragraph that says something new. Embeddings can. Each section is split into
passages, each passage is embedded, and every passage in this year's filing is matched
to its closest passage in last year's filing.

- semantic similarity = average best-match similarity (weighted by passage length)
- novel share        = share of this year's text whose best match is weak, meaning
                        content with no close counterpart in last year's filing

The lowest-scoring passages are stored too, so the dashboard can show WHAT changed.
Requires the optional dependency group:  uv sync --extra app --extra semantic
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .parse import load_text

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SECTIONS = ("item1a", "item7")
TARGET_WORDS = 120  # merge short lines up to about this size
MAX_WORDS = 180  # the model reads ~256 tokens; longer passages are split
NOVEL_THRESHOLD = 0.60  # best-match cosine below this = no close counterpart last year
TOP_CHANGES = 3
EMBED_DIR = config.INTERIM / "embeddings"
CHANGES = config.PROCESSED / "top_changes.parquet"


def split_passages(text: str) -> list[str]:
    """Split section text into passages of roughly TARGET_WORDS..MAX_WORDS words."""
    passages: list[str] = []
    buffer: list[str] = []
    for line in text.split("\n"):
        words = line.split()
        while len(words) > MAX_WORDS:  # a very long paragraph: cut it into pieces
            if buffer:
                passages.append(" ".join(buffer))
                buffer = []
            passages.append(" ".join(words[:MAX_WORDS]))
            words = words[MAX_WORDS:]
        if len(buffer) + len(words) > MAX_WORDS and buffer:
            passages.append(" ".join(buffer))
            buffer = []
        buffer.extend(words)
        if len(buffer) >= TARGET_WORDS:
            passages.append(" ".join(buffer))
            buffer = []
    if buffer:
        passages.append(" ".join(buffer))
    return [p for p in passages if len(p.split()) >= 8]  # drop stray headings


def pair_scores(prev: np.ndarray, cur: np.ndarray, cur_weights: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(semantic similarity, novel share, best-match score per current passage).

    Rows of prev and cur are unit-length embeddings, so a dot product is a cosine.
    """
    best = (cur.astype(np.float32) @ prev.astype(np.float32).T).max(axis=1)
    w = cur_weights / cur_weights.sum()
    return float((best * w).sum()), float(w[best < NOVEL_THRESHOLD].sum()), best


def _embed(model, cik: int, accession: str, section: str, text: str) -> tuple[np.ndarray, list[str]]:
    passages = split_passages(text)
    path = EMBED_DIR / str(cik) / f"{accession}_{section}.npy"
    if path.exists():
        vectors = np.load(path)
        if len(vectors) == len(passages):
            return vectors, passages
    if not passages:
        return np.zeros((0, 1), dtype=np.float16), passages
    vectors = model.encode(passages, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype=np.float16)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors, passages


def build_semantic(model=None) -> pd.DataFrame:
    """Add semantic columns to the similarity panel and store the top changed passages."""
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)
    sim = pd.read_parquet(config.SIMILARITY)
    for s in SECTIONS:
        sim[f"{s}_semantic"] = np.nan
        sim[f"{s}_novel_share"] = np.nan
    changes = []
    for cik, firm in tqdm(sim.groupby("cik"), desc="Embedding"):
        cache: dict[tuple[str, str], tuple[np.ndarray, list[str]]] = {}

        def get(accession: str, section: str, cik=cik, cache=cache):
            key = (accession, section)
            if key not in cache:
                cache[key] = _embed(model, int(cik), accession, section, load_text(cik, accession)[section])
            return cache[key]

        for i, row in firm.iterrows():
            for s in SECTIONS:
                if pd.isna(row[f"{s}_cosine"]):
                    continue  # the section failed QA in one of the two years
                prev_vec, _ = get(row["prev_accession"], s)
                cur_vec, cur_passages = get(row["accession"], s)
                if not len(prev_vec) or not len(cur_passages):
                    continue
                weights = np.array([len(p.split()) for p in cur_passages], dtype=float)
                score, novel, best = pair_scores(prev_vec, cur_vec, weights)
                sim.at[i, f"{s}_semantic"] = score
                sim.at[i, f"{s}_novel_share"] = novel
                for j in np.argsort(best)[:TOP_CHANGES]:
                    changes.append(
                        {"cik": int(cik), "ticker": row["ticker"], "accession": row["accession"],
                         "filing_date": row["filing_date"], "section": s, "match_score": float(best[j]),
                         "passage": cur_passages[j][:700]}
                    )
    sim.to_parquet(config.SIMILARITY, index=False)
    pd.DataFrame(changes).to_parquet(CHANGES, index=False, compression="zstd")
    return sim
