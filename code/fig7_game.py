import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
g=1.0; alpha=0.5; ndr=1.0; gamma=alpha/ndr; edge=np.sqrt(gamma)
def fauc(pos,neg):
    a=np.concatenate([pos,neg]);o=np.argsort(a);rk=np.empty(len(a));rk[o]=np.arange(1,len(a)+1)
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def detectors_auc(V,K,d=400,seed=0,b=60):
    r=np.random.default_rng(seed); dm=int(alpha*d); n=int(ndr*d); a=V/(g*K)
    def gen(hack,N):
        X=r.standard_normal((N,d))
        if hack:
            for i in range(K): X[:,i]+=np.sqrt(a)*r.standard_normal(N)
        return X[:,:dm]
    Xh=gen(True,n); C=np.cov(Xh.T); w,Vv=np.linalg.eigh(C); U=Vv[:,-K:]
    Nte=3000; Th=gen(True,Nte); Tc=gen(False,Nte); ns=Nte//b
    up=U[:,-1]; sh=(Th@up)**2; sc=(Tc@up)**2
    auc_pk=max(fauc(sh[:ns*b].reshape(ns,b).mean(1),sc[:ns*b].reshape(ns,b).mean(1)),0.5)
    sh2=((Th@U)**2).sum(1); sc2=((Tc@U)**2).sum(1)
    auc_ag=max(fauc(sh2[:ns*b].reshape(ns,b).mean(1),sc2[:ns*b].reshape(ns,b).mean(1)),0.5)
    return auc_pk,auc_ag
def cross(Vs,A,lvl=0.72):
    for i in range(len(Vs)-1):
        if (A[i]-lvl)*(A[i+1]-lvl)<0:
            t=(lvl-A[i])/(A[i+1]-A[i]); return Vs[i]+t*(Vs[i+1]-Vs[i])
    return np.nan
SEEDS=16
# panel (a): AUC vs V at K=4, mean+/-sd
K0=4; Vs=np.array([0.5,1.0,1.4,1.8,2.2,2.6,2.83,3.3,4.0,4.8]); Vp=g*K0*edge; Vo=g*np.sqrt(K0*gamma)
PK=np.array([[detectors_auc(V,K0,seed=s)[0] for V in Vs] for s in range(SEEDS)])
AG=np.array([[detectors_auc(V,K0,seed=s)[1] for V in Vs] for s in range(SEEDS)])
# panel (b): measured wall V* vs K (both detectors), mean+/-sd
Ks=[1,2,4,8,16]; Vgrid=np.linspace(0.3,9,26)
vpk_m=[];vpk_s=[];vag_m=[];vag_s=[]
for K in Ks:
    wp=[];wa=[]
    for s in range(SEEDS):
        Ap=[detectors_auc(V,K,seed=100*s+K)[0] for V in Vgrid]; Aa=[detectors_auc(V,K,seed=100*s+K)[1] for V in Vgrid]
        wp.append(cross(Vgrid,Ap)); wa.append(cross(Vgrid,Aa))
    wp=np.array(wp); wa=np.array(wa)
    vpk_m.append(np.nanmean(wp)); vpk_s.append(np.nanstd(wp)); vag_m.append(np.nanmean(wa)); vag_s.append(np.nanstd(wa))
Ks=np.array(Ks)

fig,ax=plt.subplots(1,2,figsize=(11,4.1))
ax[0].errorbar(Vs,PK.mean(0),yerr=PK.std(0),fmt="-o",color="#d62728",ms=4,capsize=3,label="peak (BBP) monitor")
ax[0].errorbar(Vs,AG.mean(0),yerr=AG.std(0),fmt="-s",color="#1f77b4",ms=4,capsize=3,label="aggregate (optimal) monitor")
ax[0].axvline(Vp,ls="--",color="#d62728",lw=1.1); ax[0].text(Vp+0.05,0.55,r"$V^\star_{\rm spec}$",color="#d62728",fontsize=8)
ax[0].axvline(Vo,ls="--",color="#1f77b4",lw=1.1); ax[0].text(Vo-0.8,0.92,r"$V^\star_{\rm opt}$",color="#1f77b4",fontsize=8)
ax[0].axhline(0.5,ls=":",color="gray"); ax[0].set_ylim(0.45,1.02)
ax[0].set_xlabel(r"required reward $V$"); ax[0].set_ylabel("detection AUC (mean$\\pm$sd)")
ax[0].set_title(f"(a) game value at $K{{=}}{K0}$ ({SEEDS} seeds)"); ax[0].legend(fontsize=8,loc="center right")
Kf=np.linspace(1,16,50)
ax[1].plot(Kf,g*Kf*edge,"-",color="#d62728",lw=1.5,label=r"theory $gK\sqrt{\gamma}$ (peak)")
ax[1].plot(Kf,g*np.sqrt(Kf*gamma),"-",color="#1f77b4",lw=1.5,label=r"theory $g\sqrt{K\gamma}$ (optimal)")
ax[1].errorbar(Ks,vpk_m,yerr=vpk_s,fmt="o",color="#d62728",ms=6,capsize=3,label="measured wall (peak)")
ax[1].errorbar(Ks,vag_m,yerr=vag_s,fmt="s",color="#1f77b4",ms=6,capsize=3,label="measured wall (aggregate)")
ax[1].fill_between(Kf,g*np.sqrt(Kf*gamma),g*Kf*edge,color="#bbb",alpha=0.3)
ax[1].set_xlabel(r"task rank $K$"); ax[1].set_ylabel(r"critical reward $V^\star$ (measured)")
ax[1].set_title(r"(b) peak vs aggregate walls: the $\sqrt{K}$ gap"); ax[1].legend(fontsize=7.5,loc="upper left")
plt.tight_layout(); plt.savefig("game.png",dpi=150); plt.close()
print("K:",list(Ks)); print("V*_peak meas:",[round(x,2) for x in vpk_m]," theory:",[round(g*K*edge,2) for K in Ks])
print("V*_agg  meas:",[round(x,2) for x in vag_m]," theory:",[round(g*np.sqrt(K*gamma),2) for K in Ks])
