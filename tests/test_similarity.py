import math
from collections import Counter

import pandas as pd

from lazyprices import similarity as sim


def test_identical_text_scores_one_and_disjoint_text_scores_zero():
    a = sim.tokenize("We face intense competition in all of our markets.")
    assert sim.compare(a, a) == {"cosine": 1.0, "jaccard": 1.0, "edit": 1.0}
    b = sim.tokenize("Completely unrelated vocabulary appears here instead today")
    scores = sim.compare(a, b)
    assert scores["cosine"] == 0.0 and scores["jaccard"] == 0.0 and scores["edit"] == 0.0


def test_measures_match_hand_computed_values():
    a, b = ["risk", "risk", "growth"], ["risk", "growth", "growth", "debt"]
    # cosine: dot = 2*1 + 1*2 = 4; norms sqrt(5) and sqrt(6)
    assert math.isclose(sim.cosine_similarity(Counter(a), Counter(b)), 4 / math.sqrt(30))
    # jaccard: {risk, growth} shared out of {risk, growth, debt}
    assert math.isclose(sim.jaccard_similarity(set(a), set(b)), 2 / 3)
    # edit: risk risk growth -> risk growth growth debt takes 2 edits; longer length is 4
    assert math.isclose(sim.edit_similarity(a, b), 1 - 2 / 4)


def test_tokenizer_drops_numbers_and_case():
    assert sim.tokenize("Revenue grew 12% to $4,500 million in FY2024.") == [
        "revenue", "grew", "to", "million", "in", "fy",
    ]


def test_reordering_lowers_edit_but_not_cosine_or_jaccard():
    a = sim.tokenize("alpha beta gamma delta epsilon zeta eta theta")
    b = list(reversed(a))
    scores = sim.compare(a, b)
    assert math.isclose(scores["cosine"], 1.0) and scores["jaccard"] == 1.0 and scores["edit"] < 0.3


def test_pairs_require_consecutive_fiscal_years():
    filings = pd.DataFrame(
        {
            "accession": ["a", "b", "b2", "c", "d"],
            "report_date": pd.to_datetime(["2018-12-31", "2019-12-31", "2019-12-31", "2021-12-31", "2022-12-31"]),
            "filing_date": pd.to_datetime(["2019-02-20", "2020-02-20", "2020-03-01", "2022-02-20", "2023-02-20"]),
        }
    )
    pairs = sim.consecutive_pairs(filings)
    acc = filings["accession"]
    # 2018->2019 and 2021->2022 are valid; 2019->2021 skips a year; b2 is a duplicate period.
    assert [(acc[p], acc[c]) for p, c in pairs] == [("a", "b"), ("c", "d")]
