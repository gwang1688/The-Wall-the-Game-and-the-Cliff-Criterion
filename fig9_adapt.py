import numpy as np
# Correct statement: the monitor WHITENS within the range of the benign covariance Sigma0.
# If Sigma0 has effective rank r_eff (<= d_mon nominal), the whitened problem is r_eff-dimensional
# isotropic, so the BBP edge is sqrt(r_eff / n), NOT sqrt(d_mon/n).  Test with a hard rank-r Sigma0.
def fauc(pos,neg):
    x=np.concatenate([pos,neg]);o=np.argsort(x);rk=np.empty(len(x));rk[o]=np.arange(1,len(x)+1)
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def wall(dmon,r,n,b=50,seeds=5):
    def auc(beta,seed):
        rng=np.random.default_rng(seed)
        B=np.linalg.qr(rng.standard_normal((dmon,r)))[0]        # r-dim active subspace (range of Sigma0)
        v=B@rng.standard_normal(r); v/=np.linalg.norm(v)        # spike lives in the active subspace
        def gen(hack,N):
            X=(rng.standard_normal((N,r))@B.T)                  # benign noise in r-dim range, isotropic there
            if hack: X=X+np.sqrt(beta)*np.outer(rng.standard_normal(N),v)
            return X                                            # already "whitened" (Sigma0=I on range)
        Xtr=gen(True,n); C=np.cov(Xtr.T); w,V=np.linalg.eigh(C); u=V[:,-1]
        Th=gen(True,2000); Tc=gen(False,2000); ns=2000//b
        sh=((Th@u)**2)[:ns*b].reshape(ns,b).mean(1); sc=((Tc@u)**2)[:ns*b].reshape(ns,b).mean(1)
        return max(fauc(sh,sc),0.5)
    lo,hi=0.0,3.0
    for _ in range(20):
        bb=0.5*(lo+hi); a=np.mean([auc(bb,s) for s in range(seeds)])
        if a>0.75: hi=bb
        else: lo=bb
    return 0.5*(lo+hi)
n=400; dmon=300
print(f"benign effective rank r_eff, monitor nominal dmon={dmon}, n={n}")
print(f"{'r_eff':>6} {'wall_meas':>9} {'sqrt(r_eff/n)':>13} {'sqrt(dmon/n)':>12}")
for r in [50,100,200,300]:
    wb=wall(dmon,r,n)
    print(f"{r:>6} {wb:>9.3f} {np.sqrt(r/n):>13.3f} {np.sqrt(dmon/n):>12.3f}")
