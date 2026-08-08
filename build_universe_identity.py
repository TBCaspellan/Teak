from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sec_mirror import SECFinancialStatementMirror
from engo_provider import EngoPriceProvider

EVIDENCE_PATH='form345/historical_ticker_evidence_form345.parquet'
START='2012-01-01'
END='2024-12-31'


def signal_dates_from_spy(provider):
    spy=provider.history('SPY',START,END)
    if spy.empty: raise RuntimeError('Cannot derive trading calendar from SPY')
    spy=spy.sort_values('date')
    qends=pd.date_range(START,END,freq='QE')
    out=[]
    for q in qends:
        z=spy.loc[spy['date']<=q,'date']
        if len(z): out.append(pd.Timestamp(z.iloc[-1]).normalize())
    return sorted(set(out))


def main():
    outdir=Path('universe_identity'); outdir.mkdir(exist_ok=True)
    evall=pd.read_parquet(EVIDENCE_PATH)
    evall['cik']=evall['cik'].astype(str).str.replace(r'\D','',regex=True).str.zfill(10)
    evall['ticker']=evall['ticker'].astype(str).str.upper().str.strip()
    evall['evidence_date']=pd.to_datetime(evall['evidence_date'],errors='coerce').dt.normalize()
    evall=evall.dropna(subset=['cik','ticker','evidence_date']).sort_values(['cik','evidence_date','ticker'])

    engo=EngoPriceProvider()
    book=engo.symbol_book()
    book.columns=[str(c).lower() for c in book.columns]
    book=book.rename(columns={'code':'ticker'})
    book['ticker']=book['ticker'].astype(str).str.upper().str.strip()
    book['exchange']=book['exchange'].astype(str).str.upper().str.strip()
    book['type']=book['type'].astype(str)
    book['status']=book['status'].astype(str).str.lower()
    book=book[book['exchange'].isin(['NYSE','NASDAQ','AMEX'])]
    book=book[book['type'].str.contains('Common Stock',case=False,na=False)]
    book=book.sort_values(['ticker','status']).drop_duplicates('ticker',keep='first')

    sec=SECFinancialStatementMirror()
    q=f"""
    SELECT DISTINCT cik, name, sic, form, period, accepted
    FROM sec.main.submissions
    WHERE form IN ('10-Q','10-K')
      AND accepted BETWEEN TIMESTAMP '2010-06-01 00:00:00'
                       AND TIMESTAMP '2025-01-02 00:00:00'
    ORDER BY accepted
    """
    subs=sec.con.execute(q).df()
    subs['cik']=subs['cik'].astype(str).str.replace(r'\D','',regex=True).str.zfill(10)
    subs['accepted']=pd.to_datetime(subs['accepted'],errors='coerce')
    subs['period']=pd.to_datetime(subs['period'],errors='coerce')
    subs['sic']=pd.to_numeric(subs['sic'],errors='coerce')
    subs=subs.dropna(subset=['cik','accepted','sic'])

    signals=signal_dates_from_spy(engo)
    panels=[]; summary=[]
    for sd in signals:
        signal_close=pd.Timestamp(f'{sd.date()} 16:00:00')
        filing_lo=signal_close-pd.Timedelta(days=550)
        cand=subs[(subs['accepted']<=signal_close)&(subs['accepted']>=filing_lo)].copy()
        cand=cand[~cand['sic'].between(6000,6799,inclusive='both')]
        cand=cand.sort_values(['cik','accepted']).groupby('cik',as_index=False).tail(1)
        sec_n=len(cand)

        # Historical identity evidence is carried forward until contradicted.
        # Date-only evidence on the signal date is conservatively unavailable until next session.
        ev=evall[evall['evidence_date']<sd].copy()
        ev=ev.sort_values(['cik','evidence_date','ticker']).groupby('cik',as_index=False).tail(1)
        m=cand.merge(ev[['cik','ticker','evidence_date','issuer_name','evidence_source']],on='cik',how='left')
        hist_n=int(m['ticker'].notna().sum())
        m=m.merge(book[['ticker','name','type','exchange','status']],on='ticker',how='left',suffixes=('','_engo'))
        book_n=int(m['exchange'].notna().sum())
        m['signal_date']=sd
        m['historical_ticker_resolved']=m['evidence_date'].notna()
        m['engo_major_exchange_common']=m['exchange'].notna()
        m['identity_eligible']=m['historical_ticker_resolved'] & m['engo_major_exchange_common']
        m['exclusion_reason']=np.select(
            [~m['historical_ticker_resolved'], m['historical_ticker_resolved'] & ~m['engo_major_exchange_common']],
            ['NO_HISTORICAL_TICKER','NO_ENGO_MAJOR_EXCHANGE_COMMON'],
            default=None
        )
        panels.append(m)
        summary.append({
            'signal_date':str(sd.date()),'sec_candidates':sec_n,
            'historical_ticker_resolved':hist_n,'engo_book_resolved':book_n,
            'identity_eligible':int(m['identity_eligible'].sum()),
            'ticker_resolution_rate':hist_n/sec_n if sec_n else None,
            'identity_eligible_rate':float(m['identity_eligible'].mean()) if len(m) else None,
        })

    panel=pd.concat(panels,ignore_index=True) if panels else pd.DataFrame()
    panel.to_parquet(outdir/'universe_identity_panel.parquet',index=False)
    pd.DataFrame(summary).to_csv(outdir/'universe_identity_by_quarter.csv',index=False)
    eligible=panel[panel['identity_eligible']]
    report={
        'status':'PASS' if len(eligible) else 'FAIL',
        'identity_evidence':'FORM345_HF_HISTORICAL',
        'signal_dates':len(signals),
        'panel_rows':len(panel),
        'eligible_rows':len(eligible),
        'unique_eligible_ciks':int(eligible['cik'].nunique()) if len(eligible) else 0,
        'unique_eligible_tickers':int(eligible['ticker'].nunique()) if len(eligible) else 0,
        'first_signal':None if not signals else str(signals[0].date()),
        'last_signal':None if not signals else str(signals[-1].date()),
        'overall_identity_eligible_rate':float(panel['identity_eligible'].mean()) if len(panel) else None,
        'exclusion_reasons':panel.loc[~panel['identity_eligible'],'exclusion_reason'].fillna('UNKNOWN').value_counts().to_dict(),
        'engo_status_counts':eligible['status'].fillna('UNKNOWN').value_counts().to_dict(),
        'quarter_summary':summary,
    }
    (outdir/'universe_identity_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print(json.dumps(report,indent=2,default=str))
    if report['status']!='PASS': raise SystemExit(1)

if __name__=='__main__': main()
