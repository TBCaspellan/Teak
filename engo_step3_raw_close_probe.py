from __future__ import annotations
import json, math, os, time
from pathlib import Path
import requests
import pandas as pd
import numpy as np

BASE='https://engo.capital'
TICKERS=['SPY','AAPL','NVDA','CRM','RNOW']
ACTIVE_REF=['SPY','AAPL','NVDA','CRM']
START='2010-01-01'
END='2099-12-31'


def get_json(session,url,params=None,tries=4):
    last=None
    for k in range(tries):
        try:
            r=session.get(url,params=params,timeout=90)
            r.raise_for_status(); return r.json()
        except Exception as e:
            last=e; time.sleep(1.0+k)
    raise last


def engo_bars(session,ticker):
    obj=get_json(session,f'{BASE}/api/v1/lake/eod/{ticker}',{'from':START,'to':END})
    rows=obj.get('bars') or obj.get('data') or obj.get('rows') or obj.get('eod') or []
    x=pd.DataFrame(rows)
    if x.empty: return x,obj
    x.columns=[str(c).lower() for c in x.columns]
    x['date']=pd.to_datetime(x['date']).dt.tz_localize(None)
    for c in ['open','high','low','close','volume']:
        x[c]=pd.to_numeric(x[c],errors='coerce')
    x=x.dropna(subset=['date','close']).sort_values('date').drop_duplicates('date').reset_index(drop=True)
    return x,obj


def engo_actions(session,ticker):
    return get_json(session,f'{BASE}/api/v1/lake/actions/{ticker}')


def split_factor_after(d, terminal, splits):
    f=1.0
    for sd,ratio in splits:
        if d < sd <= terminal: f*=ratio
    return f


def reconstruct(bars, actions):
    """
    Recover two series from Engo's total-return adjusted close:
      split_close: price adjusted for splits only, normalized to terminal share basis
      raw_close:   as-traded historical closing price

    Let A_t be Engo adjusted close, S_t split-only adjusted close, P_t the product
    of splits strictly after t through the terminal date, and D_t the cash dividend
    effective on trading day t in raw per-share units. Then

        A_t / A_{t-1} = (S_t + D_t/P_t) / S_{t-1}

    so backward recursion is

        S_{t-1} = (S_t + D_t/P_t) / (A_t/A_{t-1}).

    At the terminal bar there are no later adjustments, so S_T=A_T. Finally
    raw_close_t = S_t * P_t and raw_volume_t = adjusted_volume_t / P_t.

    Corporate actions are used only to invert vendor back-adjustments. They are not
    exposed as predictive model inputs.
    """
    x=bars.copy().sort_values('date').reset_index(drop=True)
    terminal=x.date.iloc[-1].normalize()
    splits=[]
    for a in (actions or {}).get('splits',[]):
        try:
            d=pd.Timestamp(a['date']).normalize(); r=float(a['ratio'])
            if np.isfinite(r) and r>0: splits.append((d,r))
        except Exception: pass
    splits=sorted(splits)
    divs={}
    for a in (actions or {}).get('dividends',[]):
        try:
            d=pd.Timestamp(a['date']).normalize(); v=a.get('unadjusted_value',a.get('value'))
            v=float(v)
            if np.isfinite(v): divs[d]=divs.get(d,0.0)+v
        except Exception: pass

    P=np.array([split_factor_after(pd.Timestamp(d).normalize(),terminal,splits) for d in x.date],dtype=float)
    A=x.close.to_numpy(float)
    S=np.full(len(x),np.nan,dtype=float)
    S[-1]=A[-1]
    for i in range(len(x)-1,0,-1):
        if not(np.isfinite(A[i]) and np.isfinite(A[i-1]) and A[i]>0 and A[i-1]>0 and np.isfinite(S[i])):
            continue
        d=pd.Timestamp(x.date.iloc[i]).normalize()
        div_raw=divs.get(d,0.0)
        div_terminal=div_raw/P[i] if np.isfinite(P[i]) and P[i]>0 else 0.0
        tr=A[i]/A[i-1]
        S[i-1]=(S[i]+div_terminal)/tr
    x['future_split_factor']=P
    x['split_close_reconstructed']=S
    x['raw_close_reconstructed']=S*P
    x['raw_volume_reconstructed']=x.volume/P if 'volume' in x else np.nan
    return x


def yahoo(session,ticker,start,end):
    p1=int(pd.Timestamp(start,tz='UTC').timestamp()); p2=int((pd.Timestamp(end,tz='UTC')+pd.Timedelta(days=1)).timestamp())
    url=f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
    obj=get_json(session,url,{'period1':p1,'period2':p2,'interval':'1d','events':'div,splits','includeAdjustedClose':'true'})
    res=obj['chart']['result'][0]; ts=res.get('timestamp') or []
    q=(res.get('indicators') or {}).get('quote',[{}])[0]
    ac=(res.get('indicators') or {}).get('adjclose',[{}])[0].get('adjclose',[])
    y=pd.DataFrame({'date':pd.to_datetime(ts,unit='s',utc=True).tz_convert(None),
                    'y_close_split':q.get('close',[]),
                    'y_volume_split':q.get('volume',[]),
                    'y_adj_total':ac})
    y['date']=y.date.dt.normalize()
    return y


def relerr(a,b):
    a=pd.to_numeric(a,errors='coerce'); b=pd.to_numeric(b,errors='coerce')
    ok=a.notna()&b.notna()&(b.abs()>1e-12)
    return ((a[ok]-b[ok]).abs()/b[ok].abs()) if ok.any() else pd.Series(dtype=float)


def main():
    key=os.environ.get('ENGO_API_KEY')
    if not key: raise SystemExit('ENGO_API_KEY missing')
    es=requests.Session(); es.headers.update({'Authorization':f'Bearer {key}','User-Agent':'Teak-OPEN-Step3/1.0'})
    ys=requests.Session(); ys.headers.update({'User-Agent':'Mozilla/5.0 TeakResearch/1.0'})
    out=Path('engo_step3_raw_close'); out.mkdir(exist_ok=True)
    cases={}; all_pass=True

    for ticker in TICKERS:
        bars,meta=engo_bars(es,ticker); acts=engo_actions(es,ticker)
        if bars.empty:
            cases[ticker]={'status':'FAIL','error':'NO_ENGO_BARS'}; all_pass=False; continue
        r=reconstruct(bars,acts)
        # Raw-close invariant: reconstructed as-traded close should sit inside raw daily OHLC.
        v=r[['low','high','raw_close_reconstructed']].dropna(); v=v[v.high>=v.low]
        inside=((v.raw_close_reconstructed>=v.low*.998)&(v.raw_close_reconstructed<=v.high*1.002)) if len(v) else pd.Series(dtype=bool)
        ohlc_rate=float(inside.mean()) if len(inside) else np.nan
        terminal_rel=abs(float(r.raw_close_reconstructed.iloc[-1]/r.close.iloc[-1]-1)) if r.close.iloc[-1] else np.nan

        case={'rows':len(r),'first_date':str(r.date.min().date()),'last_date':str(r.date.max().date()),
              'n_splits':len((acts or {}).get('splits',[])),'n_dividends':len((acts or {}).get('dividends',[])),
              'raw_close_ohlc_containment_rate':ohlc_rate,'terminal_anchor_relerr':terminal_rel,
              'engo_close_basis':meta.get('close_basis') if isinstance(meta,dict) else None}

        if ticker in ACTIVE_REF:
            y=yahoo(ys,ticker,max(pd.Timestamp('2012-01-01'),r.date.min()),r.date.max())
            z=r.merge(y,on='date',how='inner')
            eS=relerr(z.split_close_reconstructed,z.y_close_split)
            eV=relerr(z.volume,z.y_volume_split)
            case.update({'yahoo_overlap_rows':len(z),
                         'split_close_vs_yahoo_median_relerr':float(eS.median()) if len(eS) else np.nan,
                         'split_close_vs_yahoo_p99_relerr':float(eS.quantile(.99)) if len(eS) else np.nan,
                         'engo_volume_vs_yahoo_split_volume_median_relerr':float(eV.median()) if len(eV) else np.nan})
            z.tail(1500).to_parquet(out/f'{ticker}_validation_tail.parquet',index=False)
        else:
            r.tail(1500).to_parquet(out/f'{ticker}_validation_tail.parquet',index=False)

        # Hard gate: OHLC reconstruction must be nearly exact. Yahoo is an independent
        # active-name validation reference only, never a production source.
        ok_ohlc=bool(np.isfinite(ohlc_rate) and ohlc_rate>=.995)
        ok_ref=True
        if ticker in ACTIVE_REF:
            ok_ref=bool(case.get('yahoo_overlap_rows',0)>=100 and case['split_close_vs_yahoo_median_relerr']<=.001 and case['split_close_vs_yahoo_p99_relerr']<=.01)
        case['status']='PASS' if ok_ohlc and ok_ref else 'FAIL'
        all_pass=all_pass and case['status']=='PASS'
        cases[ticker]=case

    report={'status':'PASS' if all_pass else 'FAIL',
            'method':'BACKWARD_TOTAL_RETURN_TO_SPLIT_ONLY_THEN_RAW',
            'formula':{'split_only_backward':'S[t-1]=(S[t]+Dividend[t]/P[t])/(A[t]/A[t-1])',
                       'raw_close':'C[t]=S[t]*P[t]','raw_volume':'V_raw[t]=V_engo[t]/P[t]',
                       'P':'product of split ratios strictly after t through terminal bar'},
            'pit_policy':'Current corporate-action records are normalization metadata only; reconstructed raw prices/volumes are model inputs. Corporate-action events themselves are not predictive features.',
            'yahoo_policy':'Yahoo is validation-only for active securities; it is never a production historical-universe source.',
            'cases':cases,
            'NO_FORWARD_OUTCOMES_ACCESSED':True}
    (out/'step3_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print(json.dumps(report,indent=2,default=str))
    if report['status']!='PASS': raise SystemExit(1)

if __name__=='__main__': main()
