import numpy as np
def auc(a,b):
    x=np.concatenate([a,b]);o=np.argsort(x);r=np.empty(len(x));r[o]=np.arange(1,len(x)+1)
    return (r[:len(a)].sum()-len(a)*(len(a)+1)/2)/(len(a)*len(b))

def detect_real(V,q,m,K,n,sessions=60,seed=0):
    # REAL BBP: observer watches m random coords of K, n samples, top eigenvalue of m x m sample cov.
    r=np.random.default_rng(seed)
    a=V/q; sig=[]; cln=[]
    for _ in range(sessions):
        loaded=set(r.choice(K,size=q,replace=False).tolist())
        watched=r.choice(K,size=m,replace=False)
        var=np.ones(m)
        for j,w in enumerate(watched):
            if w in loaded: var[j]=1+a
        Xs=r.standard_normal((n,m))*np.sqrt(var); Cs=Xs.T@Xs/n
        sig.append(np.linalg.eigvalsh(Cs)[-1])
        Xc=r.standard_normal((n,m)); Cc=Xc.T@Xc/n
        cln.append(np.linalg.eigvalsh(Cc)[-1])
    return auc(np.array(sig),np.array(cln))

def best_hide_auc(V,m,K,n,seed=0):
    qs=sorted(set([1,2,4,8,16,32,64,128,K])); qs=[q for q in qs if q<=K]
    return min(detect_real(V,q,m,K,n,seed=seed) for q in qs)   # adversary minimizes detection

print("=== DECISIVE: does the covering-game wall SHARPEN with dimension at fixed f=0.5? ===")
print("(cliff <=> AUC(V) steepens toward a step as (K,m,n) scale together)")
for scale in [1,2,4]:
    K=64*scale; m=32*scale; n=128*scale   # f=m/K=0.5 fixed
    # find rough wall V* then probe steepness around it
    Vgrid=np.linspace(0.5,6,16); aucs=[best_hide_auc(V,m,K,n,seed=scale) for V in Vgrid]
    aucs=np.array(aucs)
    # wall = first V with auc>=0.75
    idx=np.argmax(aucs>=0.75); Vstar=Vgrid[idx] if aucs[-1]>=0.75 else np.nan
    if not np.isnan(Vstar):
        lo=best_hide_auc(0.7*Vstar,m,K,n,seed=scale+10)
        hi=best_hide_auc(1.3*Vstar,m,K,n,seed=scale+10)
        print(f"scale={scale} (K={K},m={m},n={n}): V*~{Vstar:.2f}  AUC(0.7V*)={lo:.3f}  AUC(1.3V*)={hi:.3f}  jump={hi-lo:.3f}")
    else:
        print(f"scale={scale}: no wall in grid, max AUC={aucs.max():.3f}")

print("\n=== V*(f): smooth 1/sqrt scaling or a sharp covering threshold? (K=128,n=256) ===")
K=128;n=256
print(f"{'f':>6} {'m':>4} {'V*':>7} {'V*·sqrt(mn)/K':>14}  (const => smooth K/sqrt(mn))")
for m in [8,16,32,48,64,96]:
    Vgrid=np.linspace(0.5,12,24); aucs=np.array([best_hide_auc(V,m,K,n,seed=7) for V in Vgrid])
    idx=np.argmax(aucs>=0.75); Vstar=Vgrid[idx] if aucs[-1]>=0.75 else np.inf
    ref=K/np.sqrt(m*n)
    print(f"{m/K:>6.3f} {m:>4} {Vstar:>7.2f} {(Vstar*np.sqrt(m*n)/K) if np.isfinite(Vstar) else np.nan:>14.2f}")
