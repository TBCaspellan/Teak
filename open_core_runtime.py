from __future__ import annotations
import math
import numpy as np
import pandas as pd
import open_core_engine as _core


def exact_aq_raw(q: pd.DataFrame):
    """
    Ten exact chronological fiscal quarters.

    Do not trust SEC FSD `fy` as the sequence key: real issuer audits showed that
    FY labels can collide across adjacent submissions. Quarter position comes from
    fqtr and chronology from datadate.
    """
    if len(q)<10:return np.nan
    z=q.sort_values('datadate').tail(10).copy()
    if z[['fqtr','datadate','revenue_q']].isna().any().any():return np.nan
    dates=pd.to_datetime(z['datadate']).to_list();fq=z['fqtr'].astype(int).to_list()
    for i in range(1,10):
        if fq[i] != (fq[i-1] % 4)+1:return np.nan
        days=(dates[i]-dates[i-1]).days
        if not (60<=days<=125):return np.nan
    rev=pd.to_numeric(z['revenue_q'],errors='coerce').to_numpy(dtype=float)
    if not np.all(np.isfinite(rev)) or np.any(rev<=0):return np.nan
    g=np.array([math.log(rev[i]/rev[i-4]) for i in range(4,10)],dtype=float)
    t=np.arange(1,7,dtype=float);coef=np.polyfit(t,g,1);beta=float(coef[0]);pred=np.polyval(coef,t)
    ss_res=float(np.sum((g-pred)**2));ss_tot=float(np.sum((g-g.mean())**2));r2=1.0 if ss_tot<1e-18 else max(0.0,1.0-ss_res/ss_tot)
    return beta*r2

# The original fundamental_raw resolves exact_aq_raw from its module globals at
# call time. Replace that symbol before re-exporting the engine. This keeps one
# implementation of every other frozen formula while fixing the now-audited SEC
# period-sequence law.
_core.exact_aq_raw = exact_aq_raw

fundamental_raw=_core.fundamental_raw
market_raw=_core.market_raw
add_industry_raw=_core.add_industry_raw
score_snapshot=_core.score_snapshot
finalize_raw=_core.finalize_raw
split_adjusted_shares=_core.split_adjusted_shares
weighted_component=_core.weighted_component
