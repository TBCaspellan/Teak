from __future__ import annotations

def renorm(d):
    s=sum(d.values());return {k:v/s for k,v in d.items()}

OPEN_CORE_WEIGHTS=renorm({'F':.24,'Q':.18,'R_Q':.14,'M':.12,'D':.08})
F_WEIGHTS={'RG':.25,'AQ':.25,'GPG':.20,'dOM':.15,'dFCFM':.15}
Q_WEIGHTS={'CE':.20,'GM':.20,'IM':.20,'CQ':.20,'PSG':.20}
RQ_OPEN_WEIGHTS=renorm({'CAP':.30,'PR':.25,'SG':.20})
M_WEIGHTS={'RS6':.25,'RS12':.20,'IRS':.20,'HIGH':.15,'ACC':.20}
D_WEIGHTS={'GMP':.35,'ROICP':.35,'FCFS':.30}
FR_WEIGHTS={'LEV':.25,'LIQ':.20,'RUN':.20,'DIL':.20,'INT':.15}
EB_OPEN_WEIGHTS=renorm({'EVS':.35,'EVGP':.25,'FCFB':.20})
LR_OPEN_WEIGHTS=renorm({'SIZE':.20,'IVOL':.25,'MAX':.20,'ILLIQ':.20})
COMPONENT_COVERAGE_MIN=.80
COS_COVERAGE_MIN=.90
ADV60_MIN_USD=1_000_000.0

for name,w in [('OPEN_CORE',OPEN_CORE_WEIGHTS),('F',F_WEIGHTS),('Q',Q_WEIGHTS),('RQ',RQ_OPEN_WEIGHTS),('M',M_WEIGHTS),('D',D_WEIGHTS),('FR',FR_WEIGHTS),('EB',EB_OPEN_WEIGHTS),('LR',LR_OPEN_WEIGHTS)]:
    assert abs(sum(w.values())-1)<1e-12,(name,sum(w.values()))
