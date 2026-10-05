"""Dashboard for the Lazy Prices replication. Reads only the small result files in
data/processed, so it runs anywhere without the raw filings.

    uv run streamlit run app/streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
PRIMARY = ["full_cosine", "item1a_cosine"]
NAMES = {
    "full": "Full 10-K", "item1a": "Risk Factors (Item 1A)", "item7": "MD&A (Item 7)",
    "cosine": "cosine", "jaccard": "Jaccard", "edit": "edit distance", "semantic": "semantic (embeddings)",
}

st.set_page_config(page_title="Lazy Prices replication", layout="wide")


def label(signal: str) -> str:
    section, measure = signal.split("_", 1)
    return f"{NAMES.get(section, section)}, {NAMES.get(measure, measure)}"


@st.cache_data
def load(name: str) -> pd.DataFrame | None:
    path = DATA / name
    if not path.exists():
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def line_chart(frame: pd.DataFrame, y_title: str, tick_format: str | None = None) -> go.Figure:
    fig = go.Figure()
    for color, col in zip(SERIES, frame.columns, strict=False):
        fig.add_trace(go.Scatter(x=frame.index, y=frame[col], name=col, mode="lines", line={"color": color, "width": 2}))
    fig.update_layout(height=380, margin={"l": 10, "r": 10, "t": 10, "b": 10}, hovermode="x unified",
                      legend={"orientation": "h", "y": 1.1}, yaxis_title=y_title)
    if tick_format:
        fig.update_yaxes(tickformat=tick_format)
    return fig


ports, perf, attr, quint = load("portfolio_returns.parquet"), load("performance.csv"), load("attribution.csv"), load("quintile_means.csv")
sim, qa_year, changes = load("similarity.parquet"), load("parse_qa_by_year.csv"), load("top_changes.parquet")

st.title("Do investors read the changes? A Lazy Prices replication")
st.caption(
    "Cohen, Malloy and Nguyen (Journal of Finance, 2020) found that companies whose 10-K text changes the most "
    "underperform afterwards. This project rebuilds the signal from SEC EDGAR filings for S&P 500 companies, "
    "tests it out of sample, and checks whether the return is explained by standard risk factors."
)

if ports is None or sim is None:
    st.error("No results found in data/processed. Run the pipeline first: `uv run lp analyze`.")
    st.stop()

tab_results, tab_factors, tab_company, tab_quality = st.tabs(["Results", "Factor attribution", "Company explorer", "Data quality"])

with tab_results:
    signals = sorted(ports["signal"].unique(), key=lambda s: (s not in PRIMARY, s))
    c1, c2 = st.columns([2, 1])
    chosen = c1.multiselect("Signals (up to 3)", signals, default=[s for s in PRIMARY if s in signals], format_func=label, max_selections=3)
    sample = c2.selectbox("Sample", list(perf["sample"].unique()))
    net = st.toggle("After trading costs", value=False)
    series = "LS_net" if net else "LS"

    cols = st.columns(max(len(chosen), 1))
    for col, s in zip(cols, chosen, strict=False):
        p = perf[(perf["signal"] == s) & (perf["sample"] == sample) & (perf["series"] == series)].iloc[0]
        a = attr[(attr["signal"] == s) & (attr["sample"] == sample) & (attr["series"] == series) & (attr["model"] == "FF5+MOM")].iloc[0]
        col.markdown(f"**{label(s)}**")
        col.metric("Long-short return, annualized", f"{p['ann_return']:.1%}", help="Least-changed quintile minus most-changed quintile")
        col.metric("t-statistic (Newey-West)", f"{p['t_stat']:.2f}")
        col.metric("Six-factor alpha, bps per month", f"{a['alpha_monthly_bps']:.0f}", f"t = {a['alpha_t']:.2f}", delta_color="off", delta_arrow="off")

    st.subheader("Growth of $1 in the long-short portfolio")
    wide = ports.pivot(index="date", columns="signal", values=series)[chosen]
    start, end = {"Full sample": (None, None)}.get(sample, (None, "2014-12-31") if "2014" in sample else ("2015-01-01", None))
    wide = wide.loc[start:end]
    growth = (1 + wide.fillna(0)).cumprod().where(wide.notna().cummax()).rename(columns=label)
    st.plotly_chart(line_chart(growth, "Value of $1", "$.2f"), width="stretch")

    st.subheader("Mean monthly return by quintile (bps)")
    st.caption("Q1 = most changed text, Q5 = least changed. The paper predicts returns rising from Q1 to Q5.")
    q = quint[(quint["sample"] == sample) & quint["signal"].isin(chosen)]
    qcols = st.columns(max(len(q), 1))
    for col, (_, row) in zip(qcols, q.iterrows(), strict=False):
        vals = [row[k] for k in ("Q1", "Q2", "Q3", "Q4", "Q5")]
        fig = go.Figure(go.Bar(x=["Q1", "Q2", "Q3", "Q4", "Q5"], y=vals, marker_color=SERIES[0], text=[f"{v:.0f}" for v in vals], textposition="outside"))
        fig.update_layout(height=300, margin={"l": 10, "r": 10, "t": 30, "b": 10}, title=label(row["signal"]))
        col.plotly_chart(fig, width="stretch")

with tab_factors:
    st.caption(
        "Each row regresses the long-short return on a set of factors. Alpha is what the factors cannot explain. "
        "Betas show which factors the strategy leans on. Standard errors are Newey-West with 6 lags."
    )
    sample_f = st.selectbox("Sample", list(attr["sample"].unique()), key="attr_sample")
    table = attr[(attr["sample"] == sample_f) & (attr["series"] == "LS")].copy()
    table = table.rename(columns={"model": "factor model"})
    table.insert(0, "Signal", table["signal"].map(label))
    show = ["Signal", "factor model", "months", "alpha_monthly_bps", "alpha_t", "alpha_p", "r2"] + [c for c in table if c.startswith("beta_")]
    st.dataframe(table[show].round(3), width="stretch", hide_index=True)

with tab_company:
    tickers = sorted(sim["ticker"].unique())
    ticker = st.selectbox("Company", tickers, index=tickers.index("AAPL") if "AAPL" in tickers else 0)
    firm = sim[sim["ticker"] == ticker].sort_values("filing_date")
    measure = st.radio("Measure", [m for m in ("cosine", "jaccard", "edit", "semantic") if f"item1a_{m}" in firm], horizontal=True, format_func=lambda m: NAMES[m])
    cols_m = [c for c in (f"full_{measure}", f"item1a_{measure}", f"item7_{measure}") if c in firm]
    hist = firm.set_index("filing_date")[cols_m].rename(columns=lambda c: NAMES[c.split("_")[0]])
    st.subheader(f"{ticker}: year-over-year 10-K similarity")
    st.plotly_chart(line_chart(hist, "Similarity to prior year (1 = unchanged)"), width="stretch")
    if changes is not None:
        firm_changes = changes[changes["ticker"] == ticker]
        if len(firm_changes):
            dates = sorted(firm_changes["filing_date"].dt.date.unique(), reverse=True)
            when = st.selectbox("Most novel passages in the 10-K filed on", dates)
            st.caption("Passages with the weakest match to anything in the prior year's filing. Lower score = newer content.")
            picked = firm_changes[firm_changes["filing_date"].dt.date == when].sort_values("match_score")
            for _, row in picked.iterrows():
                st.markdown(f"**{NAMES[row['section']]}** (match score {row['match_score']:.2f})")
                st.write(row["passage"] + " ...")
    with st.expander("Data table"):
        table = firm.drop(columns=["cik"])
        numeric = table.select_dtypes("number").columns
        table[numeric] = table[numeric].round(4)
        st.dataframe(table, width="stretch", hide_index=True)

with tab_quality:
    st.caption(
        "10-K HTML is inconsistent across years and filers. These are the parser's measured success rates. "
        "Filings whose section fails the checks are excluded from that section's signal instead of being guessed."
    )
    if qa_year is not None:
        rates = qa_year.set_index("year")[["full_ok", "item1a_ok", "item7_ok"]].rename(
            columns={"full_ok": NAMES["full"], "item1a_ok": NAMES["item1a"], "item7_ok": NAMES["item7"]}
        )
        st.plotly_chart(line_chart(rates, "Share of filings with a usable extraction", ".0%"), width="stretch")
        st.dataframe(qa_year.round(3), width="stretch", hide_index=True)
    st.markdown(
        "**Known limitations.** The universe is current S&P 500 members, so companies that were delisted or "
        "dropped are missing (survivorship bias). Prices are free adjusted closes, not CRSP. Portfolios are "
        "equal-weighted. See docs/METHODOLOGY.md in the repository for how each one is handled."
    )
