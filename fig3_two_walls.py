import numpy as np
# Adaptive (aiming) monitor: aims its resolution at the PUBLIC task subspace (dim K_pool<=alpha d)
# and measures TOTAL task-subspace variance. Delivering reward V forces task variance V/g, which
# the trace test detects once V/g > ~sqrt(2 K_pool/n). Claim: spread-INVARIANT, wall ~ g sqrt(K_pool/n),
# FAR below the fixed-monitor wall g sqrt(K gamma). => adaptive resolution collapses the wall.
def fauc(pos,neg):
    x=np.concatenate([pos,neg]);o=np.argsort(x);rk=np.empty(len(x));rk[o]=np.arange(1,len(x)+1)
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
g=1.0; d=1200; alpha=0.3; ndr=1.0; n=int(ndr*d)
def aiming_trace_auc(V,Kpool,K,seed=0,b=40):
    # model spreads reward over K<=Kpool task dirs, spike a=V/(gK); monitor aims at the Kpool task subspace
    r=np.random.default_rng(seed); a=V/(g*K)
    def stat(hack,N):
        X=r.standard_normal((N,Kpool))                     # traces restricted to the aimed task subspace
        if hack: X[:, :K]+=np.sqrt(a)*r.standard_normal((N,K))
        return (X**2).sum(1)                               # total task-subspace variance
    sh=stat(True,2400); sc=stat(False,2400); ns=2400//b
    return max(fauc(sh[:ns*b].reshape(ns,b).mean(1), sc[:ns*b].reshape(ns,b).mean(1)),0.5)
def fixed_spectral_auc(V,Kpool,K,seed=0,b=40):
    # fixed (non-aiming) monitor: generic alpha d subspace, spectral top-eigenvalue detector
    r=np.random.default_rng(seed); dm=int(alpha*d); a=V/(g*K)
    Pm=np.linalg.qr(r.standard_normal((d,dm)))[0]; Pt=np.linalg.qr(r.standard_normal((d,Kpool)))[0]
    def gen(hack,N):
        X=r.standard_normal((N,d))
        if hack: X=X+np.sqrt(a)*(r.standard_normal((N,K))@Pt[:,:K].T)
        return X@Pm
    Xtr=gen(True,n); C=np.cov(Xtr.T); w,Vv=np.linalg.eigh(C); u=Vv[:,-1]
    Th=gen(True,2400); Tc=gen(False,2400); ns=2400//b
    sh=((Th@u)**2)[:ns*b].reshape(ns,b).mean(1); sc=((Tc@u)**2)[:ns*b].reshape(ns,b).mean(1)
    return max(fauc(sh,sc),0.5)

Kpool=200; gamma=alpha/ndr
print(f"K_pool={Kpool} (<= alpha d={int(alpha*d)}), n={n}")
print(f"predicted adaptive wall V ~ g*sqrt(2 K_pool/n) = {g*np.sqrt(2*Kpool/n):.2f}")
print(f"fixed-monitor wall (spread K) V*_opt=g sqrt(K gamma):")
print(f"\n{'V':>5} {'aim-trace AUC (K=1)':>19} {'aim-trace AUC (K=200 spread)':>28}")
for V in [0.3,0.6,1.0,1.5,2.0,3.0]:
    a1=np.mean([aiming_trace_auc(V,Kpool,1,seed=s) for s in range(3)])
    a2=np.mean([aiming_trace_auc(V,Kpool,Kpool,seed=s) for s in range(3)])   # fully spread
    print(f"{V:>5.1f} {a1:>19.2f} {a2:>28.2f}   (spread-invariant: {abs(a1-a2)<0.06})")
print(f"\nContrast at V=3.0, spread K=200:  fixed spectral AUC = {np.mean([fixed_spectral_auc(3.0,Kpool,Kpool,seed=s) for s in range(3)]):.2f}  (fixed monitor blind)  vs  aiming trace AUC = {np.mean([aiming_trace_auc(3.0,Kpool,Kpool,seed=s) for s in range(3)]):.2f}  (adaptive catches)")
