import numpy as np
import pandas as pd

from open_core_runtime import exact_aq_raw,split_adjusted_shares,market_raw,weighted_component


def quarters(n=10):
    dates=pd.date_range('2018-03-31',periods=n,freq='QE')
    return pd.DataFrame({'datadate':dates,'fqtr':[(i%4)+1 for i in range(n)],'revenue_q':[100*(1.03**i) for i in range(n)],'shares_q':[100]*n})


def test_aq_accepts_exact_10_quarters():
    assert np.isfinite(exact_aq_raw(quarters(10)))


def test_aq_rejects_internal_missing_revenue():
    q=quarters(10);q.loc[5,'revenue_q']=np.nan
    assert np.isnan(exact_aq_raw(q))


def test_aq_rejects_skipped_fiscal_quarter():
    q=quarters(10);q.loc[5,'fqtr']=4
    assert np.isnan(exact_aq_raw(q))


def test_split_adjust_shares_uses_only_splits_by_signal():
    q=quarters(5);q['shares_q']=100.0
    actions={'splits':[{'date':'2019-01-15','ratio':2.0},{'date':'2025-01-01','ratio':4.0}]}
    z,sus=split_adjusted_shares(q,actions,'2020-01-01')
    # earliest rows before 2019 split get doubled, future 2025 split never enters
    assert z.loc[0,'shares_adj']==200.0
    assert z.iloc[-1]['shares_adj']==100.0
    assert not sus


def test_market_cap_uses_raw_close_and_adv_raw_dollars():
    dates=pd.bdate_range('2020-01-01',periods=320)
    p=pd.DataFrame({'date':dates,'raw_close':10.0,'raw_volume':100_000.0,'adj_close':5.0,'ret':0.0})
    spy=pd.DataFrame({'date':dates,'ret':0.0})
    out=market_raw(p,spy,dates[-1],shares=1_000_000)
    assert out['mcap_raw']==10_000_000.0
    assert out['ADV60_raw']==1_000_000.0


def test_weighted_coverage_boundary_80_percent():
    row={'A':80.0,'B':np.nan}
    score,cov=weighted_component(row,{'A':.8,'B':.2},.8)
    assert abs(cov-.8)<1e-12 and score==80.0


def test_weighted_coverage_below_boundary_missing():
    row={'A':80.0,'B':np.nan}
    score,cov=weighted_component(row,{'A':.79,'B':.21},.8)
    assert np.isnan(score) and cov<.8
