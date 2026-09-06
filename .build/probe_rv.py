import sys,time,numpy as np,pandas as pd
sys.path.insert(0,'/home/uriel/repositories/HR-ETE-GNN-THESIS_FINAL/.build')
import yfinance as yf
from core import *
T=['EWA','EWC','EWG','EWJ','EWT','EWU','EWW','EWY','EWZ','EZA']
raw=yf.download(T+['URTH'],start='2018-01-01',end='2024-05-29',auto_adjust=True,progress=False)['Close'].dropna()
lr=np.log(raw/raw.shift(1)).dropna()
rv=(100*lr).pow(2).rolling(20).mean().pow(0.5).dropna()
sp=make_splits(rv[T].values,lookback=20)
last=rv.index[sp['t_train'][-1]]

for name,df in [('log-returns',lr.loc[:last,T]),
                ('|returns|',lr.loc[:last,T].abs()),
                ('realized vol',rv.loc[:last,T]),
                ('d log RV',np.log(rv[T]).diff().dropna().loc[:last])]:
    for a in [0.5,1.0,1.5]:
        t0=time.time()
        A,_,P=build_erte_adjacency(df,T,alpha=a,m_surrogates=200)
        pv=P[~np.eye(10,dtype=bool)]
        print(f"{name:14s} a={a}: edges(BH .10)={int((A>0).sum()):2d}/90  "
              f"p<.05 raw={int((pv<0.05).sum()):2d}  min p={pv.min():.4f}  [{time.time()-t0:.0f}s]",flush=True)
