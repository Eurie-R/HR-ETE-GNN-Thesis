import sys, time, numpy as np, pandas as pd
sys.path.insert(0,'/home/uriel/repositories/HR-ETE-GNN-THESIS_FINAL')
sys.path.insert(0,'/home/uriel/repositories/HR-ETE-GNN-THESIS_FINAL/.build')
import yfinance as yf
from core import *
from thesis_stats import (qlike, mse, mae, diebold_mariano, giacomini_white,
                          benjamini_hochberg, model_confidence_set,
                          har_rv_forecast, random_walk_forecast, min_detectable_effect)

TICKERS=['EWA','EWC','EWG','EWJ','EWT','EWU','EWW','EWY','EWZ','EZA']
PROXY='URTH'
raw=yf.download(TICKERS+[PROXY],start='2018-01-01',end='2024-05-29',auto_adjust=True,progress=False)['Close'].dropna()
log_ret=np.log(raw/raw.shift(1)).dropna()
rv=(100*log_ret).pow(2).rolling(20).mean().pow(0.5).dropna()
print("rv",rv.shape)

sp=make_splits(rv[TICKERS].values, lookback=20)
print({k:(v.shape if hasattr(v,'shape') else v) for k,v in sp.items() if k.startswith(('X','y'))})
last_train=rv.index[sp['t_train'][-1]]
print("last train date:",last_train.date(),"first test date:",rv.index[sp['t_test'][0]].date())

train_ret=log_ret.loc[:last_train,TICKERS]
print("train returns for graph:",train_ret.shape)

t0=time.time()
A_sh,_,P_sh=build_erte_adjacency(train_ret,TICKERS,alpha=1.0,m_surrogates=200,verbose=True)
A_re,_,P_re=build_erte_adjacency(train_ret,TICKERS,alpha=0.5,m_surrogates=200,verbose=True)
print(f"adjacency build {time.time()-t0:.0f}s")
np.save('A_sh.npy',A_sh); np.save('A_re.npy',A_re)
np.save('P_sh.npy',P_sh); np.save('P_re.npy',P_re)

t0=time.time()
SEEDS=range(8)
res_sh=train_multiseed(A_sh,sp,len(TICKERS),seeds=SEEDS,verbose=True)
res_re=train_multiseed(A_re,sp,len(TICKERS),seeds=SEEDS,verbose=True)
res_id=train_multiseed(np.eye(len(TICKERS)),sp,len(TICKERS),seeds=SEEDS)
print(f"training {time.time()-t0:.0f}s  spread sh={res_sh['seed_spread']:.4f} re={res_re['seed_spread']:.4f}")
np.savez('preds.npz',sh=res_sh['pred'],re=res_re['pred'],idn=res_id['pred'],
         sh_all=res_sh['per_seed'],re_all=res_re['per_seed'],y=sp['y_test'])

y=sp['y_test']
for nm,r in [('Shannon',res_sh),('Renyi0.5',res_re),('NoGraph',res_id)]:
    print(f"{nm:10s} RMSE={np.sqrt(mse(y,r['pred']).mean()):.4f} QLIKE={qlike(y,r['pred'],warn=False).mean():.5f}")
