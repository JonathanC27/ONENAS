#!/usr/bin/env python3
"""Does ADAPTATION FREQUENCY close the gap to persistence?

My earlier delta_min.py fitted ridge ONCE on 60% and tested on 40% -- a static
model on a non-stationary series, i.e. the worst case. The campaign's own
baseline table already hinted this was the wrong read:
    persistence              0.1445
    monthly-retrained ridge  0.1492   <- 3% gap, not 38%
    static ridge             0.4140
So adaptation matters enormously here, and ONE-NAS adapts every window_step
(5 days), far more often than monthly. This sweeps the retrain interval to see
whether the gap closes as adaptation gets more frequent -- which is exactly the
axis ONE-NAS is built to exploit.

Reports per-window results too: if learned models beat persistence on MOST
windows and lose only on average because of a few regime-break blow-ups, then
continuous adaptation has a real shot and the whole-stream mean is misleading.
"""
import numpy as np, pandas as pd, json
from sklearn.linear_model import Ridge

df = pd.read_csv("/anvil/scratch/x-jchang5/wwtp/aved_5min.csv")
FE = [c for c in ["N2O","NH4","NO3","PO4","O2_T1","O2_T2","O2_SP","AIR_T1",
                  "AIR_T2","AIR_BLOWER","SS_T1","TEMP","INLET_Q","SWM"] if c in df.columns]
valid = df["n2o_valid"].astype(bool).values if "n2o_valid" in df.columns else np.isfinite(df["N2O"].values)
LAGS=[0,1,2,3,6,12,24]; PPD=288   # 5-min bins per day

def nmse(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    den=np.sum((y-y.mean())**2)
    return float(np.sum((y-p)**2)/den) if den>0 else np.nan

out={}
for H in (6,24,72):
    X=[]
    for c in FE:
        v=df[c].values.astype(float)
        for L in LAGS: X.append(np.roll(v,L))
    X=np.column_stack(X)
    y0=df["N2O"].values.astype(float); yH=np.roll(y0,-H)
    ok=np.isfinite(X).all(1)&np.isfinite(yH)&np.isfinite(y0)&valid&np.roll(valid,-H)
    ok[:max(LAGS)]=False; ok[-H:]=False
    idx=np.where(ok)[0]; Xo,y0o,yHo=X[idx],y0[idx],yH[idx]
    n=len(idx)
    out[H]={}
    for days in (30,14,7,3,1):
        step=days*PPD; train=30*PPD
        preds=np.full(n,np.nan); 
        s=train
        while s<n:
            e=min(s+step,n)
            tr0=max(0,s-train)
            Xtr,ytr=Xo[tr0:s],yHo[tr0:s]
            if len(ytr)>200:
                mu,sd=Xtr.mean(0),Xtr.std(0); sd[sd==0]=1
                m=Ridge(alpha=10.0).fit((Xtr-mu)/sd,ytr)
                preds[s:e]=m.predict((Xo[s:e]-mu)/sd)
            s=e
        m_=np.isfinite(preds)
        r={"retrain_days":days,"n":int(m_.sum()),
           "ridge":nmse(yHo[m_],preds[m_]),"persistence":nmse(yHo[m_],y0o[m_])}
        # per-window (weekly) win rate
        wins=tot=0; wk=7*PPD
        for s in range(train,n,wk):
            e=min(s+wk,n); sel=m_[s:e]
            if sel.sum()<50: continue
            a=nmse(yHo[s:e][sel],preds[s:e][sel]); b=nmse(yHo[s:e][sel],y0o[s:e][sel])
            if np.isfinite(a) and np.isfinite(b):
                tot+=1; wins+= (a<b)
        r["weekly_win_rate"]=round(wins/tot,3) if tot else None; r["n_weeks"]=tot
        out[H][days]=r
        print(f"H={H:>2} retrain every {days:>2}d: ridge={r['ridge']:.4f} "
              f"persistence={r['persistence']:.4f} ratio={r['ridge']/r['persistence']:.2f} "
              f"weekly_wins={r['weekly_win_rate']} of {tot}", flush=True)
    print(flush=True)
json.dump(out, open("/anvil/scratch/x-jchang5/wwtp/adapt_freq.json","w"), indent=1, default=str)
