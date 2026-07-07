import numpy as np
rng=np.random.default_rng(0)
# Covering game: K value-directions (battlefields). Adversary loads q of them flat, energy a=V/q each.
# Observer watches m random directions, n samples. Detects via peak (max watched spike > BBP edge)
# OR aggregate (watched loaded energy > trace floor). Adversary picks q to minimize detection.
def detect_auc(V,q,m,K,n,seeds=40):
    edge=np.sqrt(m/n); afloor=np.sqrt(2*m/n)
    det=[]; clean=[]
    for s in range(seeds):
        r=np.random.default_rng(s+int(V*97)+q*7+m*3)
        loaded=r.choice(K,size=q,replace=False)          # adversary loads q dirs
        watched=r.choice(K,size=m,replace=False)          # observer watches m random dirs
        wl=np.intersect1d(watched,loaded)                 # watched & loaded
        a=V/q
        # signal: peak = a if any watched-loaded; aggregate = |wl|*a
        peak = a if len(wl)>0 else 0.0
        agg  = len(wl)*a
        # detection statistic (normalized): max over peak/edge and agg/afloor, + noise
        stat = max(peak/edge, agg/afloor) + r.normal(0,0.15)
        det.append(stat)
        clean.append(r.normal(0,0.15))                    # clean session: no signal
    det=np.array(det); clean=np.array(clean)
    # AUC
    x=np.concatenate([det,clean]);o=np.argsort(x);rk=np.empty(len(x));rk[o]=np.arange(1,len(x)+1)
    return (rk[:len(det)].sum()-len(det)*(len(det)+1)/2)/(len(det)*len(clean))

def game_value(m,K,n):
    # adversary minimizes detection over q; observer's detection is max(peak,agg). Find max V with min_q AUC<=0.75
    Vs=np.linspace(0.3,8.0,32)
    for V in Vs:
        # adversary picks best q (1..K) to MINIMIZE detection AUC
        qs=[1,2,4,8,16,32,64,128,K]; qs=[q for q in qs if q<=K]
        best=min(detect_auc(V,q,m,K,n) for q in qs)
        if best>=0.75:   # even adversary's best hiding is detected -> this V is caught
            return V
    return Vs[-1]

print("=== covering game value V* vs coverage fraction f=m/K (K=128,n=128) ===")
K=128; n=128
print(f"{'f=m/K':>8} {'m':>4} {'V*':>7} {'V*/(K/sqrt(mn))':>16}  (smooth if ratio ~const)")
fs=[]; Vstar=[]
for m in [8,16,24,32,48,64,96,120]:
    f=m/K; V=game_value(m,K,n); ref=K/np.sqrt(m*n)
    fs.append(f); Vstar.append(V)
    print(f"{f:>8.3f} {m:>4} {V:>7.2f} {V/ref:>16.2f}")

print("\n=== sharpness: does V*(f) have a kink, or smooth 1/sqrt(f)? derivative check ===")
fs=np.array(fs); Vstar=np.array(Vstar)
dV=np.gradient(Vstar,fs)
for i in range(len(fs)):
    print(f"f={fs[i]:.3f}: V*={Vstar[i]:.2f}, dV*/df={dV[i]:+.1f}")

print("\n=== does detection SHARPEN with dimension at fixed f=0.25? (BBP-style transition test) ===")
for scale in [1,2,4]:
    K_=128*scale; n_=128*scale; m_=int(0.25*K_)
    # sweep V, find AUC curve steepness near the wall
    lo=detect_auc(game_value(m_,K_,n_)*0.7, min(64*scale,K_), m_, K_, n_)
    hi=detect_auc(game_value(m_,K_,n_)*1.3, min(64*scale,K_), m_, K_, n_)
    print(f"scale={scale} (K={K_}): AUC at 0.7V*={lo:.3f}, at 1.3V*={hi:.3f}, jump={hi-lo:.3f}")
