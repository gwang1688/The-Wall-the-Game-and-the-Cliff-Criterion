import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Consolidated evidence for the DICHOTOMY law: cliff <=> second-order; robust rider for heavy tails.
def fauc(pos,neg):
    x=np.concatenate([pos,neg]);o=np.argsort(x);rk=np.empty(len(x));rk[o]=np.arange(1,len(x)+1)
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def gen(kind,N,m,r,nu=2.5):
    if kind=="gauss": return r.standard_normal((N,m))
    g=r.standard_normal((N,m)); chi=r.chisquare(nu,(N,1)); return g/np.sqrt(chi/nu)*np.sqrt((nu-2)/nu)
def sample(kind,order,bad,rho,N,m,r,nu):
    s2=1.414; vis=np.sqrt(max(s2*(1-rho),0)); v=np.zeros(m); v[0]=1.0; X=gen(kind,N,m,r,nu)
    if bad:
        if order=="mean": X=X+vis*np.outer(np.ones(N),v)
        else:             X=X+vis*np.outer(r.standard_normal(N),v)
    return X
def auc(kind,order,robust,rho,m,n,seed,nu=2.5,b=40):
    r=np.random.default_rng(seed)
    Xb=sample(kind,order,True,rho,n,m,r,nu); Xc=sample(kind,order,False,rho,n,m,r,nu)
    if order=="mean":
        Sw=np.cov(Xb.T,bias=True)+np.cov(Xc.T,bias=True)+1e-2*np.eye(m)
        u=np.linalg.solve(Sw,Xb.mean(0)-Xc.mean(0)); sc_fn=lambda Z:Z@u
    else:
        if robust:
            f=lambda X:(X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9)); C=f(Xb).T@f(Xb)/len(Xb)-f(Xc).T@f(Xc)/len(Xc)
        else: C=np.cov(Xb.T)-np.cov(Xc.T)
        ev,V=np.linalg.eigh(C); u=V[:,-1]; sc_fn=lambda Z:(Z@u)**2
    Tb=sample(kind,order,True,rho,2500,m,r,nu); Tc=sample(kind,order,False,rho,2500,m,r,nu)
    if order!="mean" and robust:
        Tb=Tb/(np.linalg.norm(Tb,axis=1,keepdims=True)+1e-9); Tc=Tc/(np.linalg.norm(Tc,axis=1,keepdims=True)+1e-9)
    ns=2500//b; sb=sc_fn(Tb)[:ns*b].reshape(ns,b).mean(1); sc=sc_fn(Tc)[:ns*b].reshape(ns,b).mean(1)
    return max(fauc(sb,sc),0.5)
m,n=80,160; rhos=np.linspace(0,0.95,16)
def curve(kind,order,rob,nu=2.5): return [np.mean([auc(kind,order,rob,rr,m,n,sd,nu) for sd in range(8)]) for rr in rhos]
fig,ax=plt.subplots(1,3,figsize=(13.5,3.9))
# (a) the dichotomy: second-order gauss = cliff, first-order = slope, + a smooth "Goodhart" ref
c_spike=curve("gauss","cov",False); c_mean=curve("gauss","mean",False)
ax[0].plot(rhos,c_spike,"-o",ms=3,color="#d62728",label="second-order (covariance spike)")
ax[0].plot(rhos,c_mean,"-s",ms=3,color="#1f77b4",label="first-order (mean shift)")
ax[0].axvline(0.5,ls="--",color="gray"); ax[0].axhline(.5,ls=":",color="gray")
ax[0].set_xlabel(r"pressure $\rho$"); ax[0].set_ylabel("detection AUC")
ax[0].set_title("(a) cliff $\\Leftrightarrow$ second-order"); ax[0].legend(fontsize=8)
# (b) robustness rider: heavy-tailed second-order: naive blind, robust restores (shifted) cliff
c_nt=curve("t","cov",False); c_rt=curve("t","cov",True)
ax[1].plot(rhos,c_nt,"-o",ms=3,color="#7f7f7f",label="naive estimator (blind)")
ax[1].plot(rhos,c_rt,"-o",ms=3,color="#2ca02c",label="robust estimator (cliff restored)")
ax[1].axhline(.5,ls=":",color="gray"); ax[1].set_xlabel(r"pressure $\rho$"); ax[1].set_ylabel("AUC")
ax[1].set_title("(b) heavy tails: estimator, not information"); ax[1].legend(fontsize=8)
# (c) schematic map of the classification (text-free bars): which regime each instance lands in
labels=["monitoring","coverage","classifier\n(2nd-order)","classifier\n(1st-order)","over-opt\n(Goodhart)","heavy-tail\n+naive","heavy-tail\n+robust"]
regime=[2,2,2,1,1,0,2]  # 2=cliff,1=slope,0=blind(estimator)
colors={2:"#d62728",1:"#1f77b4",0:"#7f7f7f"}
ax[2].barh(range(len(labels)),[1]*len(labels),color=[colors[r] for r in regime])
ax[2].set_yticks(range(len(labels))); ax[2].set_yticklabels(labels,fontsize=7.5); ax[2].set_xticks([])
ax[2].invert_yaxis()
from matplotlib.patches import Patch
ax[2].legend(handles=[Patch(color="#d62728",label="cliff (2nd-order)"),Patch(color="#1f77b4",label="slope (1st-order/regression)"),Patch(color="#7f7f7f",label="blind (naive est.; fixable)")],fontsize=7,loc="lower right")
ax[2].set_title("(c) where each game lands")
plt.tight_layout(); plt.savefig("classification_law.png",dpi=145); plt.close()
print("spike wall AUC0/mid/end:",round(c_spike[0],2),round(c_spike[8],2),round(c_spike[-1],2))
print("mean  wall AUC0/mid/end:",round(c_mean[0],2),round(c_mean[8],2),round(c_mean[-1],2))
print("heavy naive/robust AUC0:",round(c_nt[0],2),round(c_rt[0],2)," robust end:",round(c_rt[-1],2))
