from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import requests

IDENTITY_PATH='identity/universe_identity_panel.parquet'
SIGNALS=['2012-03-30','2016-12-30','2020-06-30','2024-12-31']
BASE='https://engo.capital'


def fetch_panel(session,symbols,start,end):
    payload={'symbols':symbols,'start':str(pd.Timestamp(start).date()),'end':str(pd.Timestamp(end).date()),'fields':['close','volume'],'allow_partial':True}
    r=session.post(BASE+'/api/v1/lake/panel',json=payload,timeout=180)
    r.raise_for_status(); obj=r.json()
    return pd.DataFrame(obj.get('panel') or []), obj.get('receipt') or {}


def main():
    outdir=Path('price_audit_sample'); outdir.mkdir(exist_ok=True)
    panel=pd.read_parquet(IDENTITY_PATH)
    panel['signal_date']=pd.to_datetime(panel['signal_date']).dt.normalize()
    s=requests.Session(); s.headers.update({'Authorization':f"Bearer {os.environ['ENGO_API_KEY']}",'Content-Type':'application/json'})
    all_rows=[]; receipts=[]; summaries=[]
    for sig in SIGNALS:
        sd=pd.Timestamp(sig)
        u=panel[(panel['signal_date']==sd)&(panel['identity_eligible'])].copy()
        tickers=sorted(u['ticker'].dropna().astype(str).str.upper().unique())
        pieces=[]; missing=[]
        start=sd-pd.Timedelta(days=400)
        for i in range(0,len(tickers),100):
            chunk=tickers[i:i+100]
            df,receipt=fetch_panel(s,chunk,start,sd)
            receipt['signal_date']=sig; receipt['batch_start']=i; receipts.append(receipt)
            if not df.empty: pieces.append(df)
            missing.extend(receipt.get('missing_symbols') or [])
        px=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=['symbol','date','close','volume'])
        if not px.empty:
            px['symbol']=px['symbol'].astype(str).str.upper(); px['date']=pd.to_datetime(px['date']).dt.normalize()
            px['close']=pd.to_numeric(px['close'],errors='coerce'); px['volume']=pd.to_numeric(px['volume'],errors='coerce')
        rows=[]
        for t in tickers:
            z=px[(px['symbol']==t)&(px['date']<=sd)].sort_values('date')
            rec={'signal_date':sd,'ticker':t,'rows_lookback':len(z),'last_date':pd.NaT,'obs60':0,'adv60':np.nan,'price_liquidity_pass':False}
            if len(z):
                rec['last_date']=z.iloc[-1]['date']; tail=z.tail(60); rec['obs60']=int(tail[['close','volume']].dropna().shape[0]); rec['adv60']=float((tail['close'].abs()*tail['volume']).mean()) if rec['obs60'] else np.nan
                recent=(sd-pd.Timestamp(rec['last_date'])).days<=10
                rec['price_liquidity_pass']=bool(recent and rec['obs60']>=40 and pd.notna(rec['adv60']) and rec['adv60']>=1_000_000)
            rows.append(rec)
        r=pd.DataFrame(rows); all_rows.append(r)
        summaries.append({'signal_date':sig,'identity_eligible':len(tickers),'price_rows_received':len(px),'provider_missing_symbols':len(set(missing)),'symbols_with_any_history':int((r['rows_lookback']>0).sum()),'liquidity_pass':int(r['price_liquidity_pass'].sum()),'liquidity_pass_rate':float(r['price_liquidity_pass'].mean()) if len(r) else None,'median_adv60':float(r['adv60'].median()) if len(r) else None})
    detail=pd.concat(all_rows,ignore_index=True)
    detail.to_parquet(outdir/'price_audit_detail.parquet',index=False)
    pd.DataFrame(summaries).to_csv(outdir/'price_audit_summary.csv',index=False)
    report={'status':'PASS','signals':summaries,'total_identity_rows':len(detail),'overall_liquidity_pass_rate':float(detail['price_liquidity_pass'].mean()),'receipt_batches':len(receipts),'receipts_complete_rate':float(np.mean([bool(r.get('complete')) for r in receipts])) if receipts else None}
    (outdir/'price_audit_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    (outdir/'panel_receipts.json').write_text(json.dumps(receipts,indent=2,default=str),encoding='utf-8')
    print(json.dumps(report,indent=2,default=str))

if __name__=='__main__': main()
