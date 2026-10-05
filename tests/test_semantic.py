import numpy as np

from lazyprices import semantic


def test_passages_respect_size_limits_and_keep_all_words():
    text = "\n".join(["short heading", " ".join(["word"] * 500), " ".join(["tail"] * 60), " ".join(["more"] * 70)])
    passages = semantic.split_passages(text)
    assert all(len(p.split()) <= semantic.MAX_WORDS for p in passages)
    # The 2-word heading is flushed on its own and dropped as a stray; all prose is kept.
    assert sum(len(p.split()) for p in passages) == 500 + 60 + 70


def test_pair_scores_flags_new_content():
    prev = np.eye(3, dtype=np.float32)  # three unrelated passages last year
    cur = np.array([[1, 0, 0], [0, 1, 0], [0.5, 0.5, 0.7071]], dtype=np.float32)
    cur[2] /= np.linalg.norm(cur[2])
    weights = np.array([100.0, 100.0, 200.0])
    score, novel, best = semantic.pair_scores(prev, cur, weights)
    assert np.allclose(best[:2], 1.0) and best[2] < 0.75
    assert np.isclose(score, 0.25 + 0.25 + 0.5 * best[2])
    # Reworded text (identical embedding) scores 1.0 and contributes nothing novel.
    same, novel_same, _ = semantic.pair_scores(prev, prev, np.ones(3))
    assert np.isclose(same, 1.0) and novel_same == 0.0
