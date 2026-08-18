#!/usr/bin/env python3
"""Levels vs delta parameterisation: is persistence the ceiling, or is the
LEVELS framing burning the model's capacity on rediscovering y(t+H)~y(t)?

Same rows, same metric, same features. Only the target changes:
  levels: predict y(t+H)                  -> yhat
  delta : predict y(t+H) - y(t)           -> yhat = y(t) + deltahat
nMSE is computed against the SAME y(t+H) either way, so the two are directly
comparable to each other and to the persistence gate.
"""
import numpy as np, pandas as pd, json, sys
from sklearn.linear_model import Ridge

CSV = "/anvil/scratch/x-jchang5/wwtp/aved_5min.csv"
df = pd.read_csv(CSV)
tcol = [c for c in df.columns if c.lower() in ("timestamp","time","datetime","t")]
if tcol: df[tcol[0]] = pd.to_datetime(df[tcol[0]]); df = df.sort_values(tcol[0])
FE = [c for c in ["N2O","NH4","NO3","PO4","O2_T1","O2_T2","O2_SP","AIR_T1",
                  "AIR_T2","AIR_BLOWER","SS_T1","TEMP","INLET_Q","SWM"] if c in df.columns]
valid = df["n2o_valid"].astype(bool).values if "n2o_valid" in df.columns else np.isfinite(df["N2O"].values)

def nmse(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    den=np.sum((y-y.mean())**2)
    return float(np.sum((y-p)**2)/den) if den>0 else np.nan

LAGS=[0,1,2,3,6,12,24]
out={}
for H in (6,24,72):
    X=[]; names=[]
    for c in FE:
        v=df[c].values.astype(float)
        for L in LAGS:
            X.append(np.roll(v,L)); names.append(f"{c}_l{L}")
    X=np.column_stack(X)
    y0=df["N2O"].values.astype(float)
    yH=np.roll(y0,-H)
    ok=np.isfinite(X).all(1)&np.isfinite(yH)&np.isfinite(y0)&valid&np.roll(valid,-H)
    ok[:max(LAGS)]=False; ok[-H:]=False
    Xo,y0o,yHo=X[ok],y0[ok],yH[ok]
    n=len(yHo); cut=int(n*0.6)
    mu,sd=Xo[:cut].mean(0),Xo[:cut].std(0); sd[sd==0]=1
    Xs=(Xo-mu)/sd
    res={"n_train":cut,"n_test":n-cut,"persistence":nmse(yHo[cut:],y0o[cut:])}
    m=Ridge(alpha=10.0).fit(Xs[:cut],yHo[:cut])
    res["ridge_levels"]=nmse(yHo[cut:],m.predict(Xs[cut:]))
    m=Ridge(alpha=10.0).fit(Xs[:cut],(yHo-y0o)[:cut])
    res["ridge_delta"]=nmse(yHo[cut:],y0o[cut:]+m.predict(Xs[cut:]))
    out[f"H={H} ({H*5}min)"]=res
    print(f"H={H:>2} ({H*5:>3} min)  n_test={n-cut:>6}  persistence={res['persistence']:.4f}  "
          f"ridge_levels={res['ridge_levels']:.4f}  ridge_delta={res['ridge_delta']:.4f}", flush=True)
json.dump(out, open("/anvil/scratch/x-jchang5/wwtp/delta_min.json","w"), indent=1)
print("\nIf ridge_delta << ridge_levels, the LEVELS framing is the problem and")
print("ONE-NAS should predict the increment. If ridge_delta ~ persistence, the")
print("series is a near-random-walk at that horizon and no NAS will clear it.")
