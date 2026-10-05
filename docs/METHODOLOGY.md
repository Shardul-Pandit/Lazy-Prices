# Methodology and design decisions

This document explains what each stage does and, more importantly, why. It is written
so that every choice can be defended out loud.

## 1. The idea being tested

Cohen, Malloy and Nguyen, "Lazy Prices" (Journal of Finance, 2020), studied all US 10-K
and 10-Q filings from 1995 to 2014. Their finding: when a company changes the language
of its report a lot compared with the previous year, its stock tends to underperform
over the following months. A portfolio that buys "non-changers" and shorts "changers"
earned roughly 30 to 60 basis points per month in their sample, and the effect was
strongest for changes in the Risk Factors section. Their interpretation is investor
inattention: reports are long and mostly repeated, so readers do not notice the edits.

This project asks three questions:
1. Can the signal be rebuilt from raw SEC filings with a transparent pipeline?
2. Does it still work after the paper's sample ends (2015 onward) and in large caps,
   where attention is highest and the effect should be hardest to find?
3. Does an embedding-based similarity measure, which separates rewording from new
   content, behave differently from the paper's word-counting measures?

A null result is a legitimate answer to question 2. The design below is fixed in
advance so the answer cannot be steered.

## 2. Pre-registered design

Decided before any return data was examined:

| Choice | Value | Why |
| --- | --- | --- |
| Primary signals | full-document cosine; Item 1A cosine | The paper's headline construction and its strongest section |
| Sort | quintiles, Q5 minus Q1 | Same as the paper |
| Weighting | equal-weighted | No free point-in-time market caps; S&P 500 names are all liquid |
| Holding period | 12 months | Only annual filings are used, and the paper reports returns accruing for up to 18 months. A 12-month hold also keeps about the whole universe in the sort every month |
| Risk model | Fama-French 5 factors + momentum | The standard set a reviewer will ask about |
| Inference | Newey-West standard errors, 6 lags | Monthly portfolio returns are autocorrelated and heteroskedastic |
| Out-of-sample split | returns through 2014 vs 2015 onward | 2014 is the last year of the paper's sample |

Everything else (Jaccard, edit distance, MD&A, 3- and 6-month holds, semantic measures)
is reported as robustness. With this many variations some will look significant by
chance, which is exactly why only the two primary signals count as tests.

## 3. Data

**Universe.** S&P 500 constituents from a dated snapshot (data/reference). Dual share
classes are collapsed to one company because they file one 10-K.

**Filings.** The SEC submissions API gives every filing with its filing date, period of
report and main document. Only original annual reports are used (10-K, 10-K405).
Amendments are excluded because a 10-K/A is often a partial document, and comparing a
partial document with a full one would look like a huge change.

**EDGAR access rules.** The SEC requires a User-Agent with a name and email and limits
clients to 10 requests per second. The client sends the declared User-Agent, runs at 8
requests per second through one limiter shared by all threads, backs off on 403/429
and retries server errors. Downloads are cached and written atomically (temp file, then
rename), so an interrupted run can never leave a half-written file that looks complete.

**Prices.** Daily adjusted closes from Yahoo Finance, converted to month-end to
month-end returns. Adjusted closes include dividends and splits, so these are total
returns. An unfinished final month is dropped.

**Factors.** Ken French Data Library, monthly, converted from percent to decimals.

## 4. Text processing

1. **HTML to text.** Hidden inline-XBRL blocks are removed (machine-readable facts, not
   prose). Tables where digits are more than 15% of alphanumeric characters are removed,
   the Loughran-McDonald convention, so the signal measures language and not updated
   financial statements. Tables are checked innermost first so a page-wide layout table
   is not discarded because of the numeric tables inside it.
2. **Sections.** Item 1A and Item 7 are located by their "Item N" heading lines. Each
   heading normally appears at least twice, in the table of contents and at the section
   itself, so the extractor considers every occurrence and takes the longest span that
   contains no other item heading. Lines that only mention an item ("Item 7 of this
   report describes...") are rejected by checking the title.
3. **Quality gates.** A section must have at least 500 words and be at most 75% of the
   document. A full document must have at least 5,000 words, otherwise the company has
   incorporated its content by reference to another exhibit. Failing filings are
   excluded from that section's signal and counted in the parser scorecard
   (reports/RESULTS.md). Nothing is silently guessed.

Item 1A only became a required 10-K item for fiscal years ending after 1 December 2005,
which is why the sample starts with 2005 filings.

## 5. Similarity measures

Text is lowercased and split into alphabetic words. Numbers are dropped on purpose,
otherwise every updated figure would count as changed language.

| Measure | What it captures | Blind spot |
| --- | --- | --- |
| Cosine | Same words used with similar frequency | Ignores word order |
| Jaccard | Overlap of vocabulary | Ignores frequency and order |
| Edit | Word-level Levenshtein distance, scaled by the longer document | Penalises moved text |
| Semantic (extension) | Each passage matched to its closest passage last year using sentence embeddings | Depends on the embedding model |

Filings are paired only when their periods of report are 300 to 430 days apart, so a
skipped year or a fiscal-year change is never treated as a one-year comparison.

**Why the semantic measure is the interesting extension.** Word-based measures treat
"we may be adversely affected by litigation" and "lawsuits could hurt our results" as
different text. If the paper's effect is about new information, a measure that ignores
pure rewording should be at least as informative. If it is not, that suggests the
word-level change itself (for example lawyers editing risk language) carries the signal.

## 6. Avoiding look-ahead bias

This is the most common way a backtest becomes fiction, so the rule is strict:

- A filing can be used at a month-end only if it was filed at least one business day
  before the last business day of that month. EDGAR accepts filings after the market
  closes, so a filing dated on the last day may not have been public at the close.
- The portfolio formed at the end of month t earns the return of month t+1. The signal
  matrix is never lined up with same-month returns.
- `tests/test_backtest.py` plants a large return in the filing month. A look-ahead bug
  would capture it; the test proves the strategy does not.

A separate placebo check: keeping the real Risk Factors signal but replacing real
returns with random ones, the long-short t-statistic averages 0.02 across 200
simulations, so the machinery itself does not manufacture returns. Its standard
deviation is 1.04 and 7.5% of simulations exceed |t| = 1.96 (5% would be exact), so
t-statistics near 2 are slightly less impressive than they look.

## 7. Trading costs

Each month the portfolio's weights are compared with last month's weights after price
drift. The sum of absolute weight changes is the fraction traded, charged at 10 basis
points per dollar traded on each leg. Because the signal changes once a year, turnover
is low and costs are small relative to a monthly-rebalanced signal.

## 8. Factor attribution

The long-short return is regressed on factor returns. The intercept (alpha) is the
return the factors do not explain. Loadings matter as much as alpha: if non-changers
are simply large, profitable, low-volatility companies, the strategy will load on the
profitability and investment factors and the alpha will shrink. Long-only legs are
regressed in excess of the risk-free rate; the long-short spread is self-financing, so
nothing is subtracted.

## 9. Known limitations

- **Survivorship bias.** The universe is today's index members. Companies that failed,
  were acquired or were dropped are missing, and free price sources do not carry them.
  This inflates the level of both the long and the short leg. It biases the long-short
  spread only if delisted firms were disproportionately changers or non-changers. It
  cannot be removed without CRSP-style data, so it is stated rather than hidden.
- **Large caps only.** The paper's effect is stronger in smaller stocks. S&P 500 is the
  hardest place to find an attention-based anomaly.
- **10-Ks only.** The paper also uses 10-Qs, which gives four signals per year. Using
  annual filings alone means fewer, staler signals.
- **Equal weights and free prices.** No point-in-time market caps, and adjusted closes
  from Yahoo are revised over time.
- **Short samples.** Roughly 9 years on each side of the split. Statistical power to
  detect an effect of 20 to 30 bps per month is limited, so "not significant" does not
  mean "zero".

## 10. Reading the results

- **Parser.** 9,553 of 9,554 filings parsed. Since 2011, Risk Factors is extracted in
  about 96 to 97% of filings and MD&A in 86 to 93%. Most misses are companies that
  incorporate the section by reference to their annual report exhibit (for example
  U.S. Bancorp, Clorox, Fifth Third) or use a non-standard layout (Intel, GE). Those
  are structural, so they are excluded instead of patched with company-specific rules.
- **Survivorship in levels.** The equal-weighted universe returned about 16% a year
  over the sample against about 12% for the market factor plus the risk-free rate.
  Part of that gap is equal weighting and part is that every company in the sample
  survived to the present. This is why only the long-short spread and factor-adjusted
  alphas are interpreted, never the level of either leg.
- **Primary tests.** Full-document cosine: -10 bps a month (t = -1.19). Risk Factors
  cosine: +21 bps (t = 2.08), six-factor alpha +15 bps (t = 1.74, p = 0.08). Two tests
  were pre-registered, so the Bonferroni threshold is p < 0.025. Neither passes.
- **Robustness.** Of the nine word-based signals, only Risk Factors cosine is positive
  with |t| near 2, and MD&A cosine is negative with |t| near 2. One or two results of
  that size out of nine or eleven tries is what chance alone produces. The honest
  conclusion is "no reliable evidence of the effect in large caps since 2006, with a
  directional hint in Risk Factors after 2015".
- **Semantic measure.** No return spread (t = -0.07 for Risk Factors). Its value in
  this project is descriptive: it identifies which passages are new.

## 11. What I would do next

1. Add 10-Qs, which quadruples the number of signals and matches the paper's design.
2. Extend the universe to small and mid caps, where the paper finds the effect.
3. Use point-in-time index membership and delisting returns (CRSP) to remove
   survivorship bias.
4. Test targeted changes the paper highlights: litigation language, CEO/CFO mentions
   and negative-sentiment words, instead of overall similarity.
