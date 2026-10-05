"""Figures and the auto-generated results write-up (reports/RESULTS.md).

Every number in RESULTS.md is read from the result tables produced by research.run().
Nothing is typed in by hand, so the write-up cannot drift from the data.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

from . import config, research

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed categorical order, validated for CVD

LABELS = {
    "full_cosine": "Full 10-K",
    "item1a_cosine": "Risk Factors (Item 1A)",
    "item7_cosine": "MD&A (Item 7)",
    "full_ok": "Full document",
    "item1a_ok": "Risk Factors (Item 1A)",
    "item7_ok": "MD&A (Item 7)",
}


def _style(ax, title: str, ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _lines(ax, frame: pd.DataFrame, fmt: str) -> None:
    """Draw up to three series with a legend and an end-of-line direct label."""
    ends = []
    for color, col in zip(SERIES, frame.columns, strict=False):
        s = frame[col].dropna()
        ax.plot(s.index, s.values, color=color, linewidth=2, label=LABELS.get(col, col))
        if len(s):
            ends.append((s.index[-1], s.iloc[-1]))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="best")
    # End labels that would overlap are nudged apart (at least 12 px between them).
    ax.figure.canvas.draw()
    to_points = 72.0 / ax.figure.dpi
    last_px = None
    for x, y in sorted(ends, key=lambda e: e[1]):
        px = ax.transData.transform((matplotlib.dates.date2num(x) if hasattr(x, "year") else x, y))[1]
        target = px if last_px is None else max(px, last_px + 12)
        ax.annotate(fmt.format(y), (x, y), xytext=(6, (target - px) * to_points),
                    textcoords="offset points", va="center", fontsize=9, color=INK_2)
        last_px = target


def _save(fig, name: str) -> None:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(config.FIGURES / name, dpi=160)
    plt.close(fig)


def fig_cumulative(ports: pd.DataFrame) -> None:
    wide = ports.pivot(index="date", columns="signal", values="LS")[research.PRIMARY_SIGNALS]
    growth = (1 + wide.fillna(0)).cumprod().where(wide.notna().cummax())
    fig, ax = plt.subplots(figsize=(9, 4.6))
    _style(ax, "Growth of $1 in the long-short portfolio (non-changers minus changers), before costs")
    _lines(ax, growth, "${:.2f}")
    ax.axhline(1, color=MUTED, linewidth=0.8)
    ax.axvline(pd.Timestamp("2015-01-01"), color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.annotate("paper's sample ends", (pd.Timestamp("2015-01-01"), ax.get_ylim()[1]), xytext=(4, -10),
                textcoords="offset points", fontsize=8, color=MUTED)
    _save(fig, "cumulative_long_short.png")


def fig_quintiles(quintiles: pd.DataFrame) -> None:
    rows = quintiles[(quintiles["sample"] == "Full sample") & quintiles["signal"].isin(research.PRIMARY_SIGNALS)]
    fig, axes = plt.subplots(1, len(rows), figsize=(9, 3.8), sharey=True, squeeze=False)
    for ax, (_, row) in zip(axes[0], rows.iterrows(), strict=True):
        qs = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        vals = [row[q] for q in qs]
        _style(ax, LABELS.get(row["signal"], row["signal"]), "Mean monthly return (bps)")
        bars = ax.bar(qs, vals, color=SERIES[0], width=0.62)
        ax.bar_label(bars, fmt="{:.0f}", fontsize=9, color=INK_2, padding=2)
        ax.set_xlabel("Q1 = most changed text, Q5 = least changed", color=INK_2, fontsize=9)
    _save(fig, "quintile_returns.png")


def fig_parse_quality(qa_by_year: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _style(ax, "Share of 10-Ks with a usable extraction, by filing year", "Share of filings")
    _lines(ax, qa_by_year[["full_ok", "item1a_ok", "item7_ok"]], "{:.0%}")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _save(fig, "extraction_quality.png")


def fig_similarity_trend(sim: pd.DataFrame) -> None:
    by_year = sim.assign(year=pd.to_datetime(sim["filing_date"]).dt.year).groupby("year")[
        ["full_cosine", "item1a_cosine", "item7_cosine"]
    ].median()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _style(ax, "Median year-over-year cosine similarity, by filing year", "Cosine similarity")
    _lines(ax, by_year, "{:.3f}")
    _save(fig, "similarity_trend.png")


def md_table(df: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    formats = formats or {}

    def cell(col, v):
        if pd.isna(v):
            return ""
        return formats[col].format(v) if col in formats else str(v)

    head = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(cell(c, v) for c, v in row.items()) + " |" for _, row in df.iterrows()]
    return "\n".join([head, sep, *body])


def write_results(out: dict, sim: pd.DataFrame, qa_by_year: pd.DataFrame, meta: dict) -> str:
    perf, attr, rob, quint = out["performance"], out["attribution"], out["robustness"], out["quintiles"]
    f_perf = {"mean_monthly_bps": "{:.1f}", "ann_return": "{:.2%}", "ann_vol": "{:.2%}", "sharpe": "{:.2f}",
              "t_stat": "{:.2f}", "max_drawdown": "{:.1%}", "hit_rate": "{:.0%}", "months": "{:.0f}"}
    f_attr = {"alpha_monthly_bps": "{:.1f}", "alpha_t": "{:.2f}", "alpha_p": "{:.3f}", "r2": "{:.2f}",
              "months": "{:.0f}", **{f"beta_{c}": "{:.2f}" for c in ("MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM")}}

    parts = ["# Results", "", "_Generated by `lp report`. Every number is read from data/processed; none is typed by hand._", ""]
    parts += ["## Data", "", md_table(pd.DataFrame([meta])), ""]

    parts += ["## Primary specification", "",
              "Quintile sort, equal-weighted, 12-month hold. LS = Q5 (least changed text) minus Q1 (most changed). "
              "t-statistics use Newey-West standard errors (6 lags). LS_net subtracts trading costs.", ""]
    primary = perf[perf["signal"].isin(research.PRIMARY_SIGNALS)]
    cols = ["signal", "sample", "series", "months", "mean_monthly_bps", "ann_return", "ann_vol", "sharpe", "t_stat", "max_drawdown", "hit_rate"]
    parts += [md_table(primary[[c for c in cols if c in primary]], f_perf), ""]

    parts += ["## Factor attribution of the long-short return", "",
              "Alpha is the monthly return left after the listed factors. A loading (beta) far from zero means the "
              "strategy is partly that factor in disguise.", ""]
    a = attr[attr["signal"].isin(research.PRIMARY_SIGNALS) & (attr["series"] == "LS")]
    cols = ["signal", "sample", "model", "months", "alpha_monthly_bps", "alpha_t", "alpha_p", "r2",
            "beta_MKT_RF", "beta_SMB", "beta_HML", "beta_RMW", "beta_CMA", "beta_MOM"]
    parts += [md_table(a[[c for c in cols if c in a]], f_attr), ""]

    parts += ["## Mean monthly return by quintile (bps)", ""]
    parts += [md_table(quint, {**{q: "{:.0f}" for q in ("Q1", "Q2", "Q3", "Q4", "Q5")}, "avg_names": "{:.0f}", "avg_monthly_turnover": "{:.1%}"}), ""]

    parts += ["## Robustness: every signal, FF5+MOM alpha of the long-short return", ""]
    r = attr[(attr["series"] == "LS") & (attr["model"] == "FF5+MOM")]
    parts += [md_table(r[[c for c in ("signal", "sample", "months", "alpha_monthly_bps", "alpha_t", "alpha_p") if c in r]], f_attr), ""]

    parts += ["## Robustness: holding period (primary signals, full sample)", ""]
    cols = ["signal", "hold_months", "months", "mean_monthly_bps", "t_stat", "alpha_monthly_bps", "alpha_t", "avg_names"]
    parts += [md_table(rob[[c for c in cols if c in rob]], {**f_perf, **f_attr, "avg_names": "{:.0f}"}), ""]

    parts += ["## Parser scorecard by filing year", ""]
    parts += [md_table(qa_by_year.reset_index(), {"full_ok": "{:.1%}", "item1a_ok": "{:.1%}", "item7_ok": "{:.1%}", "median_words": "{:,.0f}"}), ""]

    parts += ["## Figures", "", "![Cumulative](figures/cumulative_long_short.png)", "![Quintiles](figures/quintile_returns.png)",
              "![Extraction](figures/extraction_quality.png)", "![Similarity](figures/similarity_trend.png)", ""]
    text = "\n".join(parts)
    (config.REPORTS / "RESULTS.md").write_text(text)
    return text


def build(out: dict | None = None) -> None:
    from .parse import qa_summary

    out = out or research.run()
    sim = pd.read_parquet(config.SIMILARITY)
    qa = pd.read_parquet(config.PARSE_QA)
    index = pd.read_parquet(config.FILING_INDEX)
    qa_by_year = qa_summary(qa, index)
    qa_by_year.to_csv(config.PROCESSED / "parse_qa_by_year.csv")
    ports = out["portfolios"]
    primary = ports[ports["signal"] == research.PRIMARY_SIGNALS[0]]
    meta = {
        "companies_in_universe": index["cik"].nunique(),
        "10-K_filings_parsed": int((qa["error"].fillna("") == "").sum()),
        "year_over_year_pairs": len(sim),
        "first_filing": str(index["filing_date"].min().date()),
        "last_filing": str(index["filing_date"].max().date()),
        "return_months": f"{primary['date'].min().date()} to {primary['date'].max().date()}",
    }
    fig_cumulative(ports)
    fig_quintiles(out["quintiles"])
    fig_parse_quality(qa_by_year)
    fig_similarity_trend(sim)
    write_results(out, sim, qa_by_year, meta)
    print(f"Wrote {config.REPORTS / 'RESULTS.md'} and figures")
