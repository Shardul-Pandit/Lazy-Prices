import math

import pandas as pd

from lazyprices import factors, prices

FRENCH = """This file was created by CMPT_ME_BEME_OP_INV_RETS using the 202607 CRSP database.
The 1-month TBill rate data until 202405 are from Ibbotson Associates.

,Mkt-RF,SMB,HML,RMW,CMA,RF
196307,   -0.39,   -0.41,   -0.97,    0.68,   -1.18,    0.27
196308,    5.07,   -0.80,    1.80,    0.36,   -0.35,    0.25
196309,   -1.57,   -0.52,    0.13,   -0.71,    0.29,  -99.99

 Annual Factors: January-December 
,Mkt-RF,SMB,HML,RMW,CMA,RF
   1964,   12.68,   -0.36,    9.54,   -2.05,    6.53,    3.54
"""


def test_french_parser_reads_only_the_monthly_block_in_decimals():
    df = factors.parse_french_csv(FRENCH)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert len(df) == 3  # the annual block is ignored
    assert df.index[0] == pd.Timestamp("1963-07-31")
    assert math.isclose(df.loc["1963-08-31", "Mkt-RF"], 0.0507)
    assert pd.isna(df.loc["1963-09-30", "RF"])  # missing-value code becomes NaN


def test_monthly_returns_use_month_end_prices_and_drop_unfinished_month():
    days = pd.bdate_range("2024-01-01", "2024-04-10")
    close = pd.DataFrame({"AAA": range(100, 100 + len(days))}, index=days, dtype=float)
    monthly = prices.to_monthly_returns(close)
    # Jan is the base month, April is incomplete: only Feb and Mar returns remain.
    assert list(monthly.index) == [pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-31")]
    jan, feb = close.loc["2024-01-31", "AAA"], close.loc["2024-02-29", "AAA"]
    assert math.isclose(monthly.loc["2024-02-29", "AAA"], feb / jan - 1)
