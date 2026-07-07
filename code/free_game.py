import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def fauc(pos,neg):
    x=np.concatenate([pos,neg]);o=np.argsort(x);rk=np.empty(len(x));rk[o]=np.arange(1,len(x)+1)
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def aucs(a,alpha,K,d=600,ndr=1.0,seed=0,b=40):
    r=np.random.default_rng(seed); dm=int(alpha*d); n=int(ndr*d); kt=max(K,1)
    Pm=np.linalg.qr(r.standard_normal((d,dm)))[0]; Pt=np.linalg.qr(r.standard_normal((d,kt)))[0]
    def gen(hack,N):
        X=r.standard_normal((N,d))
        if hack: X=X+np.sqrt(a)*(r.standard_normal((N,kt))@Pt.T)
        return X@Pm
    Xtr=gen(True,n); C=np.cov(Xtr.T); w,V=np.linalg.eigh(C); u=V[:,-1]
    Th=gen(True,2000); Tc=gen(False,2000); ns=2000//b
    sh=((Th@u)**2)[:ns*b].reshape(ns,b).mean(1); sc=((Tc@u)**2)[:ns*b].reshape(ns,b).mean(1)
    auc_sp=max(fauc(sh,sc),0.5)                                   # spectral (top eigenvalue)
    th=(Th**2).sum(1)[:ns*b].reshape(ns,b).mean(1); tc=(Tc**2).sum(1)[:ns*b].reshape(ns,b).mean(1)
    auc_tr=max(fauc(th,tc),0.5)                                   # trace (total visible variance)
    return auc_sp,auc_tr
def hideable_V(alpha,K,g=1.0,d=600,ndr=1.0):
    # largest a s.t. BOTH detectors < 0.72 ; V = g K a
    lo,hi=0.0,5.0
    for _ in range(15):
        a=0.5*(lo+hi); sp,tr=np.mean([aucs(a,alpha,K,d,ndr,seed=s) for s in range(3)],axis=0)
        if max(sp,tr)<0.72: lo=a
        else: hi=a
    return g*K*lo, lo
alpha=0.3; ndr=1.0; gamma=alpha/ndr; g=1.0; d=600
Ks=[2,4,8,16,32,64,128,200,300,420]
V=[]; A=[]
for K in Ks:
    v,a=hideable_V(alpha,K,g,d,ndr); V.append(v); A.append(a)
    print(f"K={K:>4} kappa={K/d:.3f}  a_max={a:.3f}  V_hideable={v:.2f}  V_lowrank={g*np.sqrt(K*gamma):.2f}")
V=np.array(V); Ks=np.array(Ks)
Vlr=g*np.sqrt(Ks*gamma); Kstar=1/alpha**2
Vcap=np.median(V[Ks>Kstar]) if np.any(Ks>Kstar) else V[-1]
fig,ax=plt.subplots(figsize=(6.6,4.3))
ax.plot(Ks,Vlr,"--",color="#1f77b4",lw=1.8,label=r"low-rank $V^*_{\rm opt}=g\sqrt{K\gamma}$ ($\sqrt{K}$)")
ax.axhline(Vcap,ls=":",color="#2ca02c",lw=1.8,label=r"trace-test cap $\sim g/\sqrt{\alpha(n/d)}$")
ax.plot(Ks,V,"-o",color="#d62728",ms=5,label="measured hideable reward (both tests)")
ax.axvline(Kstar,ls="-.",color="gray",lw=1); ax.text(Kstar*1.05,0.5,r"$K^*\!\sim\!1/\alpha^2$",fontsize=9,color="gray")
ax.set_xscale("log"); ax.set_xlabel(r"task rank $K$"); ax.set_ylabel(r"hideable reward $V$")
ax.set_title(r"The $\sqrt{K}$ advantage saturates: extensive tasks self-defeat")
ax.legend(fontsize=8,loc="upper left"); plt.tight_layout(); plt.savefig("free.png",dpi=150); plt.close()
print(f"\nK*=1/alpha^2={Kstar:.0f}   trace cap ~ {Vcap:.2f}   g/sqrt(alpha n/d)={g/np.sqrt(alpha*ndr):.2f}")
