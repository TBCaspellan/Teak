from __future__ import annotations
import numpy as np
import pandas as pd

# Public OPEN canonicalization of standard US-GAAP / DEI concepts.
# Priority is explicit and frozen before any forward-return inspection.
#
# The low-priority service tags below were added only after a pre-outcome taxonomy
# golden audit showed Salesforce migrated away from generic Revenues/CostOfRevenue.
# We do NOT sum arbitrary product/service components or infer proprietary tags.
CONCEPTS = {
    'revenue_q': ('pnl','USD', [
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'SalesRevenueNet',
        'Revenues',
        'SalesRevenueServicesNet',
    ]),
    'cogs_q': ('pnl','USD', [
        'CostOfRevenue',
        'CostOfGoodsAndServicesSold',
        'CostOfGoodsSold',
        'CostOfServices',
        'CostOfServiceRevenue',
    ]),
    'op_income_q': ('pnl','USD',['OperatingIncomeLoss']),
    'net_income_q': ('pnl','USD',['NetIncomeLoss','ProfitLoss']),
    'interest_q': ('pnl','USD',[
        'InterestExpenseNonOperating','InterestExpenseDebt','InterestAndDebtExpense','InterestExpense']),
    'cfo_q': ('ytd','USD',[
        'NetCashProvidedByUsedInOperatingActivities',
        'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',
    ]),
    'capex_q': ('ytd','USD',[
        'PaymentsToAcquirePropertyPlantAndEquipment',
        'PaymentsToAcquireProductiveAssets',
    ]),
    'assets_q': ('instant','USD',['Assets']),
    'cash_q': ('instant','USD',[
        'CashAndCashEquivalentsAtCarryingValue',
        'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents']),
    'curr_assets_q': ('instant','USD',['AssetsCurrent']),
    'curr_liab_q': ('instant','USD',['LiabilitiesCurrent']),
    'short_debt_total_q': ('instant','USD',[
        'ShortTermDebtCurrent','LongTermDebtAndFinanceLeaseObligationsCurrent']),
    'current_lt_debt_q': ('instant','USD',['LongTermDebtCurrent']),
    'short_borrowings_q': ('instant','USD',['ShortTermBorrowings']),
    'lt_debt_noncurrent_q': ('instant','USD',[
        'LongTermDebtNoncurrent','LongTermDebtAndFinanceLeaseObligationsNoncurrent']),
    'lt_debt_total_q': ('instant','USD',['LongTermDebt']),
    'shares_q': ('instant','shares',[
        'EntityCommonStockSharesOutstanding',
        'CommonStockSharesOutstanding',
        'SharesOutstanding',
    ]),
}

TAG_LOOKUP={}
for concept,(kind,uom,tags) in CONCEPTS.items():
    for priority,tag in enumerate(tags): TAG_LOOKUP[tag]=(concept,kind,uom,priority)

FP_ORDER={'Q1':1,'Q2':2,'Q3':3,'Q4':4,'FY':4}


def signal_close(signal_date):
    return pd.Timestamp(f'{pd.Timestamp(signal_date).date()} 16:00:00')


def canonical_filing_facts(facts: pd.DataFrame) -> pd.DataFrame:
    """
    Select one canonical fact per accession/concept for the filing's current period.

    FSD laws used:
    - qtrs=0 is point-in-time.
    - pnl concepts prefer a one-quarter value for Q1/Q2/Q3 and four-quarter annual for FY.
    - cash-flow concepts prefer YTD qtrs=1/2/3 and annual qtrs=4.
    - dimensional/segment facts are excluded; consolidated coreg must be null.
    - current-period ddate must be near SUB.period (<=45 days after SEC month-end rounding).
    - alias priority is deterministic and source-generic; no company-specific tag mapping.
    """
    if facts.empty: return pd.DataFrame()
    x=facts.copy()
    for c in ('period','ddate','accepted','filed'):
        if c in x: x[c]=pd.to_datetime(x[c],errors='coerce')
    x['fp']=x['fp'].astype(str).str.upper().str.strip()
    x['qnum']=x['fp'].map(FP_ORDER)
    x=x[x['qnum'].notna() & x['tag'].isin(TAG_LOOKUP)].copy()
    if 'coreg' in x: x=x[x['coreg'].isna()]
    if 'segments' in x: x=x[x['segments'].isna() | x['segments'].astype(str).isin(['','nan','None'])]
    if x.empty:return pd.DataFrame()

    mapped=x['tag'].map(TAG_LOOKUP)
    x['concept']=[m[0] for m in mapped]
    x['kind']=[m[1] for m in mapped]
    x['expected_uom']=[m[2] for m in mapped]
    x['tag_priority']=[m[3] for m in mapped]
    x=x[x['uom'].astype(str).str.lower()==x['expected_uom'].str.lower()].copy()
    x['period_diff_days']=(x['ddate']-x['period']).dt.days.abs()
    x=x[x['period_diff_days']<=45].copy()
    if x.empty:return pd.DataFrame()

    def qrank(r):
        q=int(r['qnum']); actual=int(r['qtrs'])
        if r['kind']=='instant': return 0 if actual==0 else 99
        if r['kind']=='pnl':
            desired=4 if q==4 else 1
            if actual==desired:return 0
            if q in (2,3) and actual==q:return 1  # YTD fallback
            return 99
        # ytd concepts
        desired=q
        if actual==desired:return 0
        if q in (2,3) and actual==1:return 1      # quarter-only fallback
        return 99
    x['qtrs_rank']=x.apply(qrank,axis=1)
    x=x[x['qtrs_rank']<99].copy()
    x=x.sort_values(['adsh','concept','qtrs_rank','period_diff_days','tag_priority','tag'])
    x=x.drop_duplicates(['adsh','concept'],keep='first')
    x['value']=pd.to_numeric(x['value'],errors='coerce')
    return x


def _derive_debt(row):
    st=row.get('short_debt_total_q',np.nan)
    clt=row.get('current_lt_debt_q',np.nan)
    sb=row.get('short_borrowings_q',np.nan)
    if pd.notna(st): cur=float(st)
    else:
        pieces=[v for v in (clt,sb) if pd.notna(v)]
        cur=float(sum(pieces)) if pieces else np.nan
    non=row.get('lt_debt_noncurrent_q',np.nan)
    total=row.get('lt_debt_total_q',np.nan)
    if pd.notna(non): lt=float(non)
    elif pd.notna(total): lt=max(float(total)-(cur if pd.notna(cur) else 0.0),0.0)
    else: lt=np.nan
    return cur,lt


def quarterly_history_asof(facts: pd.DataFrame, signal_date) -> pd.DataFrame:
    """
    Reconstruct standalone quarterly canonical facts using only filings accepted
    on/before the structural signal close. Later amendments therefore cannot backfill.
    """
    if facts.empty:return pd.DataFrame()
    x=facts.copy(); x['accepted']=pd.to_datetime(x['accepted'],errors='coerce')
    x=x[x['accepted']<=signal_close(signal_date)].copy()
    if x.empty:return pd.DataFrame()

    # Latest filing version for each fiscal period. Amendments replace prior versions only after accepted.
    filings=(x[['adsh','cik','name','sic','form','period','fy','fp','filed','accepted']]
             .drop_duplicates('adsh'))
    filings['period']=pd.to_datetime(filings['period'],errors='coerce')
    filings['fp']=filings['fp'].astype(str).str.upper().str.strip()
    filings=filings[filings['fp'].isin(FP_ORDER)].copy()
    filings=filings.sort_values(['fy','fp','accepted','adsh']).drop_duplicates(['fy','fp'],keep='last')
    keep=set(filings['adsh'])
    cf=canonical_filing_facts(x[x['adsh'].isin(keep)])
    if cf.empty:return pd.DataFrame()

    base=filings[['adsh','cik','name','sic','form','period','fy','fp','filed','accepted']].copy()
    rows=[]
    for adsh,g in cf.groupby('adsh'):
        meta=base[base['adsh'].eq(adsh)].iloc[0].to_dict()
        meta['fqtr']=FP_ORDER.get(str(meta['fp']).upper())
        for r in g.itertuples():
            meta[r.concept]=r.value
            meta[r.concept+'_reported_qtrs']=int(r.qtrs)
            meta[r.concept+'_tag']=r.tag
        rows.append(meta)
    q=pd.DataFrame(rows).sort_values(['fy','fqtr','period','accepted']).reset_index(drop=True)

    # Convert duration concepts to standalone quarters.
    for concept,(kind,_,_) in CONCEPTS.items():
        if kind not in ('pnl','ytd') or concept not in q: continue
        raw=q[concept].copy(); rq=q.get(concept+'_reported_qtrs',pd.Series(np.nan,index=q.index))
        out=[]
        for i,row in q.iterrows():
            val=raw.iloc[i]; qt=rq.iloc[i]; fq=int(row['fqtr']); fy=row['fy']
            if pd.isna(val) or pd.isna(qt): out.append(np.nan); continue
            qt=int(qt)
            if qt==1:
                out.append(float(val)); continue
            # Cumulative / annual value: subtract immediately preceding cumulative value in same FY.
            prev=q.iloc[:i]
            prev=prev[prev['fy'].eq(fy) & (prev['fqtr']<fq)].sort_values('fqtr')
            if prev.empty:
                out.append(np.nan); continue
            j=prev.index[-1]
            pval=raw.loc[j]; pqt=rq.loc[j]
            if pd.notna(pval) and pd.notna(pqt) and int(pqt)==fq-1:
                out.append(float(val-pval))
            elif fq==4:
                prior_idx=q.index[(q['fy'].eq(fy)) & (q['fqtr'].isin([1,2,3])) & (q.index<i)]
                prior_vals=[out[list(q.index).index(jj)] for jj in prior_idx]
                out.append(float(val-sum(prior_vals)) if len(prior_vals)==3 and all(pd.notna(v) for v in prior_vals) else np.nan)
            else:
                out.append(np.nan)
        q[concept]=out

    debts=q.apply(_derive_debt,axis=1)
    q['curr_debt_q']=[d[0] for d in debts]
    q['lt_debt_q']=[d[1] for d in debts]
    q['datadate']=q['period']
    q['public_date']=q['accepted']
    q['gvkey']=q['cik'].astype(str).str.zfill(10)  # OPEN stable company key, not a Compustat GVKEY.
    return q.sort_values('datadate').reset_index(drop=True)
