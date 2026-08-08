from __future__ import annotations
import numpy as np
import pandas as pd

CONCEPTS = {
    'revenue_q': ('pnl','USD', ['RevenueFromContractWithCustomerExcludingAssessedTax','SalesRevenueNet','Revenues','SalesRevenueServicesNet']),
    'cogs_q': ('pnl','USD', ['CostOfRevenue','CostOfGoodsAndServicesSold','CostOfGoodsSold','CostOfServices','CostOfServiceRevenue']),
    'op_income_q': ('pnl','USD',['OperatingIncomeLoss']),
    'net_income_q': ('pnl','USD',['NetIncomeLoss','ProfitLoss']),
    'interest_q': ('pnl','USD',['InterestExpenseNonOperating','InterestExpenseDebt','InterestAndDebtExpense','InterestExpense']),
    'cfo_q': ('ytd','USD',['NetCashProvidedByUsedInOperatingActivities','NetCashProvidedByUsedInOperatingActivitiesContinuingOperations']),
    'capex_q': ('ytd','USD',['PaymentsToAcquirePropertyPlantAndEquipment','PaymentsToAcquireProductiveAssets']),
    'assets_q': ('instant','USD',['Assets']),
    'cash_q': ('instant','USD',['CashAndCashEquivalentsAtCarryingValue','CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents']),
    'curr_assets_q': ('instant','USD',['AssetsCurrent']),
    'curr_liab_q': ('instant','USD',['LiabilitiesCurrent']),
    'short_debt_total_q': ('instant','USD',['ShortTermDebtCurrent','LongTermDebtAndFinanceLeaseObligationsCurrent']),
    'current_lt_debt_q': ('instant','USD',['LongTermDebtCurrent']),
    'short_borrowings_q': ('instant','USD',['ShortTermBorrowings']),
    'lt_debt_noncurrent_q': ('instant','USD',['LongTermDebtNoncurrent','LongTermDebtAndFinanceLeaseObligationsNoncurrent']),
    'lt_debt_total_q': ('instant','USD',['LongTermDebt']),
    'shares_q': ('instant','shares',['EntityCommonStockSharesOutstanding','CommonStockSharesOutstanding','SharesOutstanding']),
}
TAG_LOOKUP={tag:(concept,kind,uom,p) for concept,(kind,uom,tags) in CONCEPTS.items() for p,tag in enumerate(tags)}
FP_ORDER={'Q1':1,'Q2':2,'Q3':3,'Q4':4,'FY':4}


def signal_close(signal_date):return pd.Timestamp(f'{pd.Timestamp(signal_date).date()} 16:00:00')


def canonical_filing_facts(facts):
    if facts.empty:return pd.DataFrame()
    x=facts.copy()
    for c in ('period','ddate','accepted','filed'):
        if c in x:x[c]=pd.to_datetime(x[c],errors='coerce')
    x['fp']=x['fp'].astype(str).str.upper().str.strip();x['qnum']=x['fp'].map(FP_ORDER)
    x=x[x['qnum'].notna()&x['tag'].isin(TAG_LOOKUP)].copy()
    if 'coreg' in x:x=x[x['coreg'].isna()]
    if 'segments' in x:x=x[x['segments'].isna()|x['segments'].astype(str).isin(['','nan','None'])]
    if x.empty:return pd.DataFrame()
    mapped=x['tag'].map(TAG_LOOKUP);x['concept']=[m[0] for m in mapped];x['kind']=[m[1] for m in mapped];x['expected_uom']=[m[2] for m in mapped];x['tag_priority']=[m[3] for m in mapped]
    x=x[x['uom'].astype(str).str.lower()==x['expected_uom'].str.lower()].copy();x['period_diff_days']=(x['ddate']-x['period']).dt.days.abs();x=x[x['period_diff_days']<=45]
    def qrank(r):
        q=int(r.qnum);a=int(r.qtrs)
        if r.kind=='instant':return 0 if a==0 else 99
        if r.kind=='pnl':
            desired=4 if q==4 else 1
            if a==desired:return 0
            if q in (2,3) and a==q:return 1
            return 99
        if a==q:return 0
        if q in (2,3) and a==1:return 1
        return 99
    x['qtrs_rank']=x.apply(qrank,axis=1);x=x[x.qtrs_rank<99].copy()
    x=x.sort_values(['adsh','concept','qtrs_rank','period_diff_days','tag_priority','tag']).drop_duplicates(['adsh','concept'],keep='first')
    x['value']=pd.to_numeric(x['value'],errors='coerce');return x


def _derive_debt(row):
    st=row.get('short_debt_total_q',np.nan);clt=row.get('current_lt_debt_q',np.nan);sb=row.get('short_borrowings_q',np.nan)
    cur=float(st) if pd.notna(st) else (float(sum(v for v in (clt,sb) if pd.notna(v))) if any(pd.notna(v) for v in (clt,sb)) else np.nan)
    non=row.get('lt_debt_noncurrent_q',np.nan);total=row.get('lt_debt_total_q',np.nan)
    lt=float(non) if pd.notna(non) else (max(float(total)-(cur if pd.notna(cur) else 0),0.0) if pd.notna(total) else np.nan)
    return cur,lt


def _sequence_ok(prev_row,row):
    if prev_row is None:return False
    expected=(int(prev_row['fqtr'])%4)+1
    days=(pd.Timestamp(row['period'])-pd.Timestamp(prev_row['period'])).days
    return int(row['fqtr'])==expected and 60<=days<=125


def quarterly_history_asof(facts,signal_date):
    """
    Point-in-time quarterly reconstruction.

    SEC FSD's FY field is not a stable fiscal-period identifier across all issuers
    (a Salesforce audit showed adjacent filings with colliding FY/FP labels).
    Therefore restatement/version identity is `period + concept`, while `fp` supplies
    quarter position. Amendments replace only concepts they actually republish.
    """
    if facts.empty:return pd.DataFrame()
    x=facts.copy();x['accepted']=pd.to_datetime(x.accepted,errors='coerce');x=x[x.accepted<=signal_close(signal_date)].copy()
    cf=canonical_filing_facts(x)
    if cf.empty:return pd.DataFrame()
    cf['fp']=cf.fp.astype(str).str.upper().str.strip();cf=cf[cf.fp.isin(FP_ORDER)].copy()
    # Period date is the authoritative historical period identity; concept-level latest accepted version wins.
    cf=cf.sort_values(['period','concept','accepted','adsh']).drop_duplicates(['period','concept'],keep='last')
    rows=[]
    for period,g in cf.groupby('period',sort=True):
        newest=g.sort_values(['accepted','adsh']).iloc[-1];fp=str(newest.fp).upper()
        meta={'cik':newest.cik,'name':newest['name'],'sic':newest.sic,'period':pd.Timestamp(period),'fy':newest.fy,'fp':fp,'fqtr':FP_ORDER.get(fp),'accepted':g.accepted.max(),'filed':g.filed.max()}
        for r in g.itertuples():
            meta[r.concept]=r.value;meta[r.concept+'_reported_qtrs']=int(r.qtrs);meta[r.concept+'_tag']=r.tag;meta[r.concept+'_accepted']=r.accepted;meta[r.concept+'_adsh']=r.adsh
        rows.append(meta)
    q=pd.DataFrame(rows).sort_values('period').reset_index(drop=True)

    for concept,(kind,_,_) in CONCEPTS.items():
        if kind not in ('pnl','ytd') or concept not in q:continue
        raw=q[concept].copy();rq=q.get(concept+'_reported_qtrs',pd.Series(np.nan,index=q.index));stand=[]
        for i,row in q.iterrows():
            val=raw.iloc[i];qt=rq.iloc[i];fq=int(row.fqtr)
            if pd.isna(val) or pd.isna(qt):stand.append(np.nan);continue
            qt=int(qt)
            if qt==1:stand.append(float(val));continue
            prev=q.iloc[i-1] if i>0 else None
            if prev is not None and _sequence_ok(prev,row):
                pval=raw.iloc[i-1];pqt=rq.iloc[i-1]
                if pd.notna(pval) and pd.notna(pqt) and int(pqt)==fq-1:
                    stand.append(float(val-pval));continue
            if fq==4 and i>=3:
                prior=q.iloc[i-3:i]
                if list(prior.fqtr.astype(int))==[1,2,3] and all(_sequence_ok(q.iloc[j-1] if j>0 else None,q.iloc[j]) for j in range(i-2,i+1)):
                    vals=stand[i-3:i]
                    stand.append(float(val-sum(vals)) if all(pd.notna(v) for v in vals) else np.nan);continue
            stand.append(np.nan)
        q[concept]=stand
    debts=q.apply(_derive_debt,axis=1);q['curr_debt_q']=[d[0] for d in debts];q['lt_debt_q']=[d[1] for d in debts]
    q['datadate']=q.period;q['public_date']=q.accepted;q['gvkey']=q.cik.astype(str).str.zfill(10)
    return q.sort_values('datadate').reset_index(drop=True)
