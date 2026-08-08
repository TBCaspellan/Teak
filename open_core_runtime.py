from __future__ import annotations
import math
import numpy as np
import pandas as pd
import open_core_engine as _core


def exact_aq_raw(q: pd.DataFrame):
    """Ten exact chronological fiscal quarters; no FY-label shortcut and no gaps."""
    if len(q)<10:return np.nan
    z=q.sort_values('datadate').tail(10).copy()
    if z[['fqtr','datadate','revenue_q']].isna().any().any():return np.nan
    dates=pd.to_datetime(z['datadate']).to_list();fq=z['fqtr'].astype(int).to_list()
    for i in range(1,10):
        if fq[i] != (fq[i-1] % 4)+1:return np.nan
        if not (60 <= (dates[i]-dates[i-1]).days <= 125):return np.nan
    rev=pd.to_numeric(z['revenue_q'],errors='coerce').to_numpy(dtype=float)
    if not np.all(np.isfinite(rev)) or np.any(rev<=0):return np.nan
    g=np.array([math.log(rev[i]/rev[i-4]) for i in range(4,10)],dtype=float)
    t=np.arange(1,7,dtype=float);coef=np.polyfit(t,g,1);beta=float(coef[0]);pred=np.polyval(coef,t)
    ss_res=float(np.sum((g-pred)**2));ss_tot=float(np.sum((g-g.mean())**2));r2=1.0 if ss_tot<1e-18 else max(0.0,1.0-ss_res/ss_tot)
    return beta*r2


def split_adjusted_shares(q:pd.DataFrame,actions,signal_date):
    """
    Normalize every *observed* SEC share count to signal-date split basis, then
    carry the latest public observation forward across later quarters.

    Forward-fill is information-safe: it represents "last known shares", not a
    fabricated new filing. `shares_source_date` preserves staleness lineage.
    """
    z=q.sort_values('datadate').copy();z['shares_adj_observed']=np.nan;z['shares_source_date']=pd.NaT
    splits=[];sig=pd.Timestamp(signal_date).normalize()
    for a in (actions or {}).get('splits',[]):
        try:d=pd.Timestamp(a['date']).normalize();ratio=float(a['ratio'])
        except Exception:continue
        if np.isfinite(ratio) and ratio>0 and d<=sig:splits.append((d,ratio))
    for idx,r in z.iterrows():
        sh=pd.to_numeric(pd.Series([r.get('shares_q')]),errors='coerce').iloc[0]
        if pd.isna(sh) or sh<=0:continue
        d=pd.Timestamp(r['datadate']).normalize();factor=1.0
        for sd,ratio in splits:
            if d<sd<=sig:factor*=ratio
        z.at[idx,'shares_adj_observed']=float(sh)*factor;z.at[idx,'shares_source_date']=d
    vals=z[['datadate','shares_adj_observed']].dropna().sort_values('datadate')
    suspect=False
    if len(vals)>=2:
        ratios=vals.shares_adj_observed.to_numpy()[1:]/vals.shares_adj_observed.to_numpy()[:-1]
        suspect=bool(np.any((ratios>10.0)|(ratios<0.10)))
    z['shares_adj']=z['shares_adj_observed'].ffill()
    z['shares_source_date']=z['shares_source_date'].ffill()
    return z,suspect

# Patch audited implementation points before re-exporting the rest of the frozen engine.
_core.exact_aq_raw=exact_aq_raw
_core.split_adjusted_shares=split_adjusted_shares

fundamental_raw=_core.fundamental_raw
market_raw=_core.market_raw
add_industry_raw=_core.add_industry_raw
score_snapshot=_core.score_snapshot
finalize_raw=_core.finalize_raw
weighted_component=_core.weighted_component
