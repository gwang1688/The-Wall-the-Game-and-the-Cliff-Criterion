import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import detect_auc                     # covariance-spike (BBP) session detector
rng=np.random.default_rng(0)
alpha,ndr,d=0.5,1.0,300; dmon=int(alpha*d); n=int(ndr*d); s2=1.414
edge=np.sqrt(alpha/ndr); rho_star=1-edge/s2       # covariance wall = 0.50
b=120                                              # session size
def fauc(pos,neg):
    a=np.concatenate([pos,neg]);o=np.argsort(a,kind="mergesort");rk=np.empty(len(a));rk[o]=np.arange(1,len(a)+1)
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

# mean-shift hack + LDA/logistic detector (first-order). visible mean sep = m*sqrt(1-rho).
m0=0.85; b_mean=15                                # weak mean-shift so AUC declines smoothly
def auc_meanshift(rho, seed=0, classifier=False):
    r=np.random.default_rng(seed); sep=m0*np.sqrt(1-rho); Nte=6000
    # LDA/logistic on visible features; for isotropic noise both reduce to the true direction u=e0
    est=r.standard_normal((n,dmon)); est_h=r.standard_normal((n,dmon)); est_h[:,0]+=sep
    w=est_h.mean(0)-est.mean(0)
    if classifier: w=w/(np.std(est,0)+1e-9)         # calibrated-classifier reweighting (still linear)
    w/=np.linalg.norm(w)+1e-9
    Sc=r.standard_normal((Nte,dmon))@w; Xh=r.standard_normal((Nte,dmon)); Xh[:,0]+=sep; Sh=Xh@w
    ns=Nte//b_mean; sh=Sh[:ns*b_mean].reshape(ns,b_mean).mean(1); sc=Sc[:ns*b_mean].reshape(ns,b_mean).mean(1)
    a=fauc(sh,sc); return max(a,1-a)

rhos=np.linspace(0,0.95,20)
auc_cov =np.array([np.mean([detect_auc(alpha,ndr,s2,float(x),d,seed=10*k+i) for k in range(4)]) for i,x in enumerate(rhos)])
auc_lda =np.array([np.mean([auc_meanshift(float(x),seed=20*k+i) for k in range(4)]) for i,x in enumerate(rhos)])
auc_clf =np.array([np.mean([auc_meanshift(float(x),seed=30*k+i,classifier=True) for k in range(4)]) for i,x in enumerate(rhos)])

fig,ax=plt.subplots(figsize=(6.4,4.2))
ax.plot(rhos,auc_cov,"-o",color="#d62728",ms=4,label="covariance spike + BBP detector (ours)")
ax.plot(rhos,auc_lda,"-s",color="#1f77b4",ms=4,label="mean-shift hack + LDA detector")
ax.plot(rhos,auc_clf,"-^",color="#2ca02c",ms=4,label="mean-shift hack + linear classifier")
ax.axvline(rho_star,ls="--",color="crimson",lw=1.2); ax.text(rho_star+0.01,0.92,r"$\hat\rho^*{=}0.50$",color="crimson",fontsize=9)
ax.axhline(0.5,ls=":",color="gray",lw=1)
ax.annotate("sharp wall\n(AUC$\\to$chance at $\\hat\\rho^*$)",(0.52,0.62),fontsize=8,color="#d62728")
ax.annotate("smooth, thresholdless\n(detectable until $\\rho{\\to}1$)",(0.15,0.60),fontsize=8,color="#1f77b4")
ax.set_xlabel(r"hidden fraction $\rho$"); ax.set_ylabel("detection AUC"); ax.set_ylim(0.45,1.02)
ax.set_title("Sharpness is a property of the hack model, not the detector"); ax.legend(fontsize=8,loc="lower left")
plt.tight_layout(); plt.savefig("baselines.png",dpi=150); plt.close()
print("cov AUC:",[round(x,2) for x in auc_cov[::4]]); print("lda AUC:",[round(x,2) for x in auc_lda[::4]]); print("clf AUC:",[round(x,2) for x in auc_clf[::4]])
