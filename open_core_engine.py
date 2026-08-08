from __future__ import annotations
import math
import numpy as np
import pandas as pd

from config_open_core import (
    OPEN_CORE_WEIGHTS,F_WEIGHTS,Q_WEIGHTS,RQ_OPEN_WEIGHTS,M_WEIGHTS,D_WEIGHTS,
    FR_WEIGHTS,EB_OPEN_WEIGHTS,LR_OPEN_WEIGHTS,COMPONENT_COVERAGE_MIN,
    COS_COVERAGE_MIN,ADV60_MIN_USD,
)


def safe_div(a,b):
    return float(a/b) if pd.notna(a) and pd.notna(b) and float(b)!=0 else np.nan


def cumret(s):
    x=pd.to_numeric(pd.Series(s),errors='coerce')
    if x.isna().any() or len(x)==0:return np.nan
    return float((1.0+x).prod()-1.0)


def _ttm(s):
    x=pd.to_numeric(pd.Series(s),errors='coerce')
    return float(x.tail(4).sum()) if len(x)>=4 and x.tail(4).notna().all() else np.nan


def exact_aq_raw(q:pd.DataFrame):
    """AQ requires ten consecutive fiscal quarters and six exact YoY log-growth observations."""
    if len(q)<10:return np.nan
    z=q.sort_values(['fy','fqtr','datadate']).tail(10).copy()
    if z[['fy','fqtr','revenue_q']].isna().any().any():return np.nan
    ords=(pd.to_numeric(z['fy']).astype(int)*4+pd.to_numeric(z['fqtr']).astype(int)).to_numpy()
    if not np.all(np.diff(ords)==1):return np.nan
    rev=pd.to_numeric(z['revenue_q'],errors='coerce').to_numpy(dtype=float)
    if not np.all(np.isfinite(rev)) or np.any(rev<=0):return np.nan
    g=np.array([math.log(rev[i]/rev[i-4]) for i in range(4,10)],dtype=float)
    t=np.arange(1,7,dtype=float)
    beta=float(np.polyfit(t,g,1)[0])
    pred=np.polyval(np.polyfit(t,g,1),t)
    ss_res=float(np.sum((g-pred)**2));ss_tot=float(np.sum((g-g.mean())**2))
    r2=1.0 if ss_tot<1e-18 else max(0.0,1.0-ss_res/ss_tot)
    return beta*r2


def split_adjusted_shares(q:pd.DataFrame,actions,signal_date):
    """
    Put historical SEC point-in-time shares onto the signal-date split basis.
    Only splits effective by signal date are applied. Later splits never enter.
    """
    z=q.copy();z['shares_adj']=np.nan
    splits=[]
    for a in (actions or {}).get('splits',[]):
        try:
            d=pd.Timestamp(a['date']).normalize();ratio=float(a['ratio'])
        except Exception:continue
        if np.isfinite(ratio) and ratio>0 and d<=pd.Timestamp(signal_date).normalize():splits.append((d,ratio))
    for idx,r in z.iterrows():
        sh=pd.to_numeric(pd.Series([r.get('shares_q')]),errors='coerce').iloc[0]
        if pd.isna(sh) or sh<=0:continue
        d=pd.Timestamp(r['datadate']).normalize();factor=1.0
        for sd,ratio in splits:
            if d<sd<=pd.Timestamp(signal_date).normalize():factor*=ratio
        z.at[idx,'shares_adj']=float(sh)*factor
    vals=z[['datadate','shares_adj']].dropna().sort_values('datadate')
    suspect=False
    if len(vals)>=2:
        ratios=vals['shares_adj'].to_numpy()[1:]/vals['shares_adj'].to_numpy()[:-1]
        # This is a data-unit tripwire, not an economic dilution threshold.
        suspect=bool(np.any((ratios>10.0)|(ratios<0.10)))
    return z,suspect


def _rolling_durability(q):
    if len(q)<10:return {k:np.nan for k in ('GMP_level_raw','GMP_stability_raw','ROICP_level_raw','ROICP_stability_raw','FCFS_level_raw','FCFS_stability_raw')}
    gm=[];roic=[];fcfm=[]
    for end in range(len(q)-5,len(q)+1):
        z=q.iloc[:end]
        if len(z)<5:continue
        rev=_ttm(z['revenue_q']);gp=_ttm(z['revenue_q']-z['cogs_q']);oi=_ttm(z['op_income_q'])
        gm.append(safe_div(gp,rev))
        ic=z['assets_q']-z['cash_q']-(z['curr_liab_q']-z['curr_debt_q'].fillna(0)).clip(lower=0)
        avg_ic=(ic.iloc[-1]+ic.iloc[-5])/2 if pd.notna(ic.iloc[-1]) and pd.notna(ic.iloc[-5]) else np.nan
        roic.append(safe_div(oi,avg_ic) if pd.notna(avg_ic) and avg_ic>0 else np.nan)
        fcf=_ttm(z['cfo_q']-z['capex_q']);fcfm.append(safe_div(fcf,rev))
    out={}
    for prefix,arr in [('GMP',gm),('ROICP',roic),('FCFS',fcfm)]:
        s=pd.Series(arr,dtype=float)
        if len(s)==6 and s.notna().all():
            out[prefix+'_level_raw']=float(s.median());out[prefix+'_stability_raw']=-float(s.std(ddof=1))
        else:
            out[prefix+'_level_raw']=np.nan;out[prefix+'_stability_raw']=np.nan
    return out


def fundamental_raw(q,actions,signal_date):
    q=q.sort_values('datadate').copy()
    q,share_suspect=split_adjusted_shares(q,actions,signal_date)
    out={'raw_quarters':int(len(q)),'share_history_suspect':share_suspect}
    rev=pd.to_numeric(q.get('revenue_q'),errors='coerce');cogs=pd.to_numeric(q.get('cogs_q'),errors='coerce');gp=rev-cogs
    oi=pd.to_numeric(q.get('op_income_q'),errors='coerce');ni=pd.to_numeric(q.get('net_income_q'),errors='coerce')
    cfo=pd.to_numeric(q.get('cfo_q'),errors='coerce');capx=pd.to_numeric(q.get('capex_q'),errors='coerce');fcf=cfo-capx
    out['AQ_raw']=exact_aq_raw(q)
    out['RG_raw']=rev.iloc[-1]/rev.iloc[-5]-1 if len(q)>=5 and pd.notna(rev.iloc[-1]) and pd.notna(rev.iloc[-5]) and rev.iloc[-5]>0 else np.nan
    out['GPG_raw']=gp.iloc[-1]/gp.iloc[-5]-1 if len(q)>=5 and pd.notna(gp.iloc[-1]) and pd.notna(gp.iloc[-5]) and gp.iloc[-5]>0 else np.nan
    rn=_ttm(rev);r0=float(rev.iloc[-8:-4].sum()) if len(q)>=8 and rev.iloc[-8:-4].notna().all() else np.nan
    gn=_ttm(gp);g0=float(gp.iloc[-8:-4].sum()) if len(q)>=8 and gp.iloc[-8:-4].notna().all() else np.nan
    on=_ttm(oi);o0=float(oi.iloc[-8:-4].sum()) if len(q)>=8 and oi.iloc[-8:-4].notna().all() else np.nan
    fn=_ttm(fcf);f0=float(fcf.iloc[-8:-4].sum()) if len(q)>=8 and fcf.iloc[-8:-4].notna().all() else np.nan
    out.update({'revenue_ttm_raw':rn,'revenue_ttm_prev_raw':r0,'gp_ttm_raw':gn,'FCF_TTM_raw':fn})
    om=safe_div(on,rn);om0=safe_div(o0,r0);fm=safe_div(fn,rn);fm0=safe_div(f0,r0)
    out['operating_margin_raw']=om;out['gross_margin_raw']=safe_div(gn,rn);out['GM_raw']=out['gross_margin_raw']
    out['dOM_raw']=om-om0 if pd.notna(om) and pd.notna(om0) else np.nan
    out['dFCFM_raw']=fm-fm0 if pd.notna(fm) and pd.notna(fm0) else np.nan
    assets=pd.to_numeric(q.get('assets_q'),errors='coerce');cash=pd.to_numeric(q.get('cash_q'),errors='coerce')
    ca=pd.to_numeric(q.get('curr_assets_q'),errors='coerce');cl=pd.to_numeric(q.get('curr_liab_q'),errors='coerce')
    cd=pd.to_numeric(q.get('curr_debt_q'),errors='coerce');ld=pd.to_numeric(q.get('lt_debt_q'),errors='coerce')
    ic=assets-cash-(cl-cd.fillna(0)).clip(lower=0)
    avg_ic=(ic.iloc[-1]+ic.iloc[-5])/2 if len(q)>=5 and pd.notna(ic.iloc[-1]) and pd.notna(ic.iloc[-5]) else np.nan
    out['CE_raw']=safe_div(on,avg_ic) if pd.notna(avg_ic) and avg_ic>0 else np.nan
    drev=rn-r0 if pd.notna(rn) and pd.notna(r0) else np.nan;doi=on-o0 if pd.notna(on) and pd.notna(o0) else np.nan
    out['IM_raw']=doi/drev if pd.notna(drev) and drev>0 and pd.notna(doi) else np.nan
    nin=_ttm(ni);cfon=_ttm(cfo);avg_assets=(assets.iloc[-1]+assets.iloc[-5])/2 if len(q)>=5 and pd.notna(assets.iloc[-1]) and pd.notna(assets.iloc[-5]) else np.nan
    out['CQ_raw']=-((nin-cfon)/avg_assets) if pd.notna(nin) and pd.notna(cfon) and pd.notna(avg_assets) and avg_assets>0 else np.nan
    # Per-share growth on common signal-date split basis.
    sh=q['shares_adj'];cur_sh=sh.iloc[-1] if len(sh) else np.nan;prev_sh=sh.iloc[-5] if len(sh)>=5 else np.nan
    if share_suspect:cur_sh=prev_sh=np.nan
    out['shares_signal_raw']=float(cur_sh) if pd.notna(cur_sh) and cur_sh>0 else np.nan
    out['shares_staleness_days']=(pd.Timestamp(signal_date).normalize()-pd.Timestamp(q.loc[sh.last_valid_index(),'datadate']).normalize()).days if sh.notna().any() else np.nan
    if pd.notna(cur_sh) and cur_sh>0 and pd.notna(prev_sh) and prev_sh>0:
        out['REVPS_raw']=(rn/cur_sh)/(r0/prev_sh)-1 if pd.notna(rn) and pd.notna(r0) and r0!=0 else np.nan
        out['GPPS_raw']=(gn/cur_sh)/(g0/prev_sh)-1 if pd.notna(gn) and pd.notna(g0) and g0!=0 else np.nan
        out['FCFPS_raw']=(fn/cur_sh)/(f0/prev_sh)-1 if pd.notna(fn) and pd.notna(f0) and f0!=0 else np.nan
        out['DIL_raw']=cur_sh/prev_sh-1
    else:out['REVPS_raw']=out['GPPS_raw']=out['FCFPS_raw']=out['DIL_raw']=np.nan
    debt=(cd.iloc[-1] if len(cd) and pd.notna(cd.iloc[-1]) else 0)+(ld.iloc[-1] if len(ld) and pd.notna(ld.iloc[-1]) else 0)
    out['debt_raw']=float(debt);out['cash_raw']=float(cash.iloc[-1]) if len(cash) and pd.notna(cash.iloc[-1]) else np.nan
    out['LEV_raw']=safe_div(debt-out['cash_raw'],assets.iloc[-1]) if len(assets) and pd.notna(out['cash_raw']) and pd.notna(assets.iloc[-1]) and assets.iloc[-1]>0 else np.nan
    out['LIQ_raw']=-safe_div(ca.iloc[-1],cl.iloc[-1]) if len(ca) and len(cl) and pd.notna(cl.iloc[-1]) and cl.iloc[-1]>0 else np.nan
    if pd.notna(fn):
        out['RUN_raw']=0.0 if fn>=0 else 1.0/(1.0+max(out['cash_raw']/abs(fn),0.0)) if pd.notna(out['cash_raw']) else np.nan
    else:out['RUN_raw']=np.nan
    intr=_ttm(pd.to_numeric(q.get('interest_q'),errors='coerce')) if 'interest_q' in q else np.nan
    out['INT_raw']=safe_div(intr,rn) if pd.notna(rn) and rn>0 else np.nan
    out.update(_rolling_durability(q))
    return out


def _aligned_returns(stock,benchmark,n):
    s=stock[['date','ret']].dropna().sort_values('date').tail(n)
    if len(s)!=n:return np.nan,np.nan
    b=benchmark[['date','ret']].dropna().rename(columns={'ret':'mret'})
    z=s.merge(b,on='date',how='inner')
    if len(z)!=n:return np.nan,np.nan
    return cumret(z['ret']),cumret(z['mret'])


def market_raw(price,spy,signal_date,shares):
    x=price[price['date']<=pd.Timestamp(signal_date)].sort_values('date').copy()
    if x.empty:return {}
    cur=x.iloc[-1];out={'price_last_date':cur['date']}
    raw_close=float(cur['raw_close']) if pd.notna(cur.get('raw_close')) else np.nan
    raw_vol=pd.to_numeric(x.get('raw_volume'),errors='coerce')
    out['raw_close_signal']=raw_close
    out['mcap_raw']=raw_close*shares if pd.notna(raw_close) and pd.notna(shares) and shares>0 else np.nan
    tail60=x.tail(60)
    out['ADV60_raw']=float((tail60['raw_close'].abs()*tail60['raw_volume']).mean()) if len(tail60)==60 and tail60[['raw_close','raw_volume']].notna().all().all() else np.nan
    out['CAP_raw']=-math.log(out['mcap_raw']) if pd.notna(out['mcap_raw']) and out['mcap_raw']>0 else np.nan
    out['SIZE_raw']=out['CAP_raw'];out['ILLIQ_raw']=-math.log(out['ADV60_raw']) if pd.notna(out['ADV60_raw']) and out['ADV60_raw']>0 else np.nan
    s6,m6=_aligned_returns(x,spy,126);s12,m12=_aligned_returns(x,spy,252)
    out['stock126_raw']=s6;out['stock252_raw']=s12
    out['RS6_raw']=s6-m6 if pd.notna(s6) and pd.notna(m6) else np.nan;out['RS12_raw']=s12-m12 if pd.notna(s12) and pd.notna(m12) else np.nan
    a=x.tail(252)
    out['HIGH_raw']=float(cur['adj_close']/a['adj_close'].max()) if len(a)==252 and a['adj_close'].notna().all() and a['adj_close'].max()>0 else np.nan
    r=x.tail(21)['ret'];out['MAX_raw']=float(r.max()) if len(r)==21 and r.notna().all() else np.nan
    # 126-session market-model IVOL.
    s=x[['date','ret']].dropna().tail(126);b=spy[['date','ret']].dropna().rename(columns={'ret':'mret'});z=s.merge(b,on='date')
    if len(z)==126:
        X=np.column_stack([np.ones(len(z)),z['mret'].to_numpy()]);y=z['ret'].to_numpy();coef=np.linalg.lstsq(X,y,rcond=None)[0];resid=y-X@coef
        out['IVOL_raw']=float(np.std(resid,ddof=1))
    else:out['IVOL_raw']=np.nan
    # Signed raw dollar-volume accumulation: current 63 versus preceding 252 sessions.
    if len(x)>=315:
        ev=x.tail(63);base=x.iloc[:-63].tail(252)
        bsdv=base['raw_close'].abs()*base['raw_volume']*np.sign(base['ret'].fillna(0));esdv=ev['raw_close'].abs()*ev['raw_volume']*np.sign(ev['ret'].fillna(0))
        sd=float(bsdv.std(ddof=1))
        out['ACC_raw']=float(((esdv-bsdv.mean())/sd).mean()) if np.isfinite(sd) and sd>0 and bsdv.notna().all() and esdv.notna().all() else np.nan
    else:out['ACC_raw']=np.nan
    return out


def add_industry_raw(raw):
    x=raw.copy();eligible=x['eligible'].fillna(False)
    # Peer fundamental aggregates use only eligible rows with valid history.
    cur=x['revenue_ttm_raw'].where(eligible);prev=x['revenue_ttm_prev_raw'].where(eligible)
    cur_sum=x.assign(_v=cur).groupby('industry_code')['_v'].transform('sum');prev_sum=x.assign(_v=prev).groupby('industry_code')['_v'].transform('sum')
    x['PR_raw']=-np.log((x['revenue_ttm_raw']/cur_sum).where((x['revenue_ttm_raw']>0)&(cur_sum>0)))
    ind_growth=cur_sum/prev_sum.replace(0,np.nan)-1;x['SG_raw']=x['RG_raw']-ind_growth
    # Value-weighted FF48 peer 126-session return, excluding focal stock where possible.
    ir=[]
    for idx,r in x.iterrows():
        peers=x[(x.index!=idx)&eligible&(x['industry_code'].eq(r['industry_code']))&x['stock126_raw'].notna()&x['mcap_raw'].notna()&(x['mcap_raw']>0)]
        if peers.empty:ir.append(np.nan)
        else:ir.append(float(np.average(peers['stock126_raw'],weights=peers['mcap_raw'])))
    x['industry126_raw']=ir;x['IRS_raw']=x['stock126_raw']-x['industry126_raw']
    return x


def pct(s):return pd.to_numeric(s,errors='coerce').rank(method='average',pct=True)*100.0


def _rank_col(pop,col,industry=False):
    out=pd.Series(np.nan,index=pop.index,dtype=float)
    if col not in pop:return out
    if industry:
        for _,g in pop.groupby('industry_code',dropna=False):
            v=g[col].dropna()
            if len(v)>=2:out.loc[v.index]=pct(v)
    else:
        v=pop[col].dropna()
        if len(v)>=2:out.loc[v.index]=pct(v)
    return out


def weighted_component(row,weights,min_cov=COMPONENT_COVERAGE_MIN):
    total=sum(weights.values());observed=[(k,w) for k,w in weights.items() if pd.notna(row.get(k,np.nan))]
    cov=sum(w for _,w in observed)/total
    if cov+1e-12<min_cov:return np.nan,cov
    den=sum(w for _,w in observed)
    return (sum(w*float(row[k]) for k,w in observed)/den if den else np.nan),cov


def score_snapshot(raw):
    if raw.empty:return raw.copy()
    x=raw.copy();pop=x[x['eligible']].copy()
    industry={
      'RG_raw':'RG','GPG_raw':'GPG','dOM_raw':'dOM','dFCFM_raw':'dFCFM','CE_raw':'CE','GM_raw':'GM','IM_raw':'IM','CQ_raw':'CQ',
      'REVPS_raw':'REVPS','GPPS_raw':'GPPS','FCFPS_raw':'FCFPS','GMP_level_raw':'GMP_L','GMP_stability_raw':'GMP_S',
      'ROICP_level_raw':'ROIC_L','ROICP_stability_raw':'ROIC_S','FCFS_level_raw':'FCFS_L','FCFS_stability_raw':'FCFS_S',
      'INT_raw':'INT','EVS_raw':'EVS','EVGP_raw':'EVGP','FCFB_raw':'FCFB'}
    market={'AQ_raw':'AQ','CAP_raw':'CAP','PR_raw':'PR','SG_raw':'SG','RS6_raw':'RS6','RS12_raw':'RS12','IRS_raw':'IRS','HIGH_raw':'HIGH','ACC_raw':'ACC',
            'LEV_raw':'LEV','LIQ_raw':'LIQ','DIL_raw':'DIL','SIZE_raw':'SIZE','IVOL_raw':'IVOL','MAX_raw':'MAX','ILLIQ_raw':'ILLIQ'}
    for a,b in industry.items():pop[b]=_rank_col(pop,a,True)
    for a,b in market.items():pop[b]=_rank_col(pop,a,False)
    # Positive FCF = exactly zero burn risk. Only burners are risk-ranked.
    pop['RUN']=0.0;burn=pop['FCF_TTM_raw']<0;v=pop.loc[burn,'RUN_raw'].dropna()
    if len(v)>=2:pop.loc[v.index,'RUN']=pct(v)
    pop['PSG']=.4*pop['REVPS']+.3*pop['GPPS']+.3*pop['FCFPS']
    pop['GMP']=.5*pop['GMP_L']+.5*pop['GMP_S'];pop['ROICP']=.5*pop['ROIC_L']+.5*pop['ROIC_S'];pop['FCFS']=.5*pop['FCFS_L']+.5*pop['FCFS_S']
    groups=[('F',F_WEIGHTS),('Q',Q_WEIGHTS),('R_Q',RQ_OPEN_WEIGHTS),('M',M_WEIGHTS),('D',D_WEIGHTS),('FR',FR_WEIGHTS),('EB',EB_OPEN_WEIGHTS),('LR',LR_OPEN_WEIGHTS)]
    for name,w in groups:
        vals=pop.apply(lambda r:weighted_component(r,w),axis=1);pop[name]=[a for a,_ in vals];pop[name+'_coverage']=[b for _,b in vals]
    vals=pop.apply(lambda r:weighted_component(r,OPEN_CORE_WEIGHTS,COS_COVERAGE_MIN),axis=1);pop['COS_OPEN']=[a for a,_ in vals];pop['COS_coverage']=[b for _,b in vals]
    risk_ok=pop[['FR','EB','LR']].notna().all(axis=1)
    pop['OFS_A_OPEN']=np.nan;pop['OFS_B_OPEN']=np.nan
    ok=pop['COS_OPEN'].notna()&risk_ok
    pop.loc[ok,'OFS_A_OPEN']=pop.loc[ok,'COS_OPEN']*(1-.003*pop.loc[ok,'FR'])*(1-.002*pop.loc[ok,'EB'])*(1-.0025*pop.loc[ok,'LR'])
    pop.loc[ok,'OFS_B_OPEN']=pop.loc[ok,'COS_OPEN']*np.exp(-1.25*(pop.loc[ok,'FR']/100)**2)*np.exp(-.60*(pop.loc[ok,'EB']/100)**2)*np.exp(-(pop.loc[ok,'LR']/100)**2)
    pop['scorable']=ok;pop['scorability_class']=np.where(pop['AQ'].notna(),'MATURE_AQ','YOUNG_OR_INCOMPLETE_AQ')
    cols=['security_id']+[c for c in pop.columns if c not in x.columns or c in ('F','Q','R_Q','M','D','FR','EB','LR','COS_OPEN','OFS_A_OPEN','OFS_B_OPEN','scorable','scorability_class')]
    cols=list(dict.fromkeys([c for c in cols if c in pop.columns]))
    return x.merge(pop[cols],on='security_id',how='left',suffixes=('','_score'))


def finalize_raw(rows,spy):
    """Add runway/industry raw fields, valuation burdens, then score. No outcomes enter here."""
    x=pd.DataFrame(rows)
    if x.empty:return x
    x=add_industry_raw(x)
    ev=x['mcap_raw']+x['debt_raw']-x['cash_raw']
    x['EV_raw']=ev;x['EVS_raw']=ev/x['revenue_ttm_raw'].replace(0,np.nan);x['EVGP_raw']=ev/x['gp_ttm_raw'].replace(0,np.nan)
    x['FCFB_raw']=-(x['FCF_TTM_raw']/x['mcap_raw'].replace(0,np.nan))
    return score_snapshot(x)
