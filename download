import numpy as np
# Two-player spectral game.  Model spreads hack energy over a K-dim TASK subspace to deliver
# reward V while minimizing detectability; monitor (fixed resolution alpha d, edge sqrt(gamma))
# aims its subspace to cover the task dirs.  Claim (analytic):
#   equilibrium spectrum = FLAT over the K task dirs, spike a = V/(g K);
#   vs PEAK (BBP) monitor:  undetectable iff a<sqrt(gamma)  =>  V < V*_peak = g K sqrt(gamma)
#   vs OPTIMAL (Le Cam) monitor: min sum h_i^2 => same flat spread; E0[L^2]=(1-a^2/gamma)^{-K/2}
#       bounded iff K a^2 < gamma  =>  V < V*_opt = g sqrt(K gamma)
g=1.0; alpha=0.5; ndr=1.0; gamma=alpha/ndr; edge=np.sqrt(gamma)   # =0.707
def model_optimal_peak(V,K,s2,d):
    # min max_i h_i  s.t. sum h=s2, sum_{task} g h >= V, h>=0  -> peak = max(V/(gK), s2/d)
    return max(V/(g*K), s2/d)

# (A) analytic walls
for K in [1,2,4,8]:
    Vp=g*K*edge; Vo=g*np.sqrt(K*gamma)
    print(f"K={K}: V*_peak=gK sqrt(g)= {Vp:.3f}   V*_opt=g sqrt(Kg)= {Vo:.3f}   ratio peak/opt={Vp/Vo:.2f} (=sqrt K={np.sqrt(K):.2f})")

# (B) simulate PEAK detector on the equilibrium spectrum: AUC vs V, collapse at V*_peak
def fauc(pos,neg):
    a=np.concatenate([pos,neg]);o=np.argsort(a);rk=np.empty(len(a));rk[o]=np.arange(1,len(a)+1)
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def peak_auc(V,K,d=400,s2=10.0,seed=0,b=60):
    r=np.random.default_rng(seed); dm=int(alpha*d); n=int(ndr*d); a=model_optimal_peak(V,K,s2,d)
    a=V/(g*K)  # task-binding regime
    def gen(hack):
        X=r.standard_normal((n,d))
        if hack:
            for i in range(K): X[:,i]+=np.sqrt(a)*r.standard_normal(n)  # K spikes in task subspace
        return X[:,:dm]
    Xh=gen(True); Xc=gen(False); C=np.cov(Xh.T); w,Vv=np.linalg.eigh(C); u=Vv[:,-1]
    Nte=3000; Th=r.standard_normal((Nte,d))
    for i in range(K): Th[:,i]+=np.sqrt(a)*r.standard_normal(Nte)
    Tc=r.standard_normal((Nte,d)); sh=(Th[:,:dm]@u)**2; sc=(Tc[:,:dm]@u)**2; ns=Nte//b
    return max(fauc(sh[:ns*b].reshape(ns,b).mean(1),sc[:ns*b].reshape(ns,b).mean(1)),0.5)
K=4; Vp=g*K*edge
print(f"\nPEAK detector, K={K}, V*_peak={Vp:.2f}:")
for V in [0.5,1.0,1.5,2.0,2.5,2.83,3.5,4.5]:
    au=np.mean([peak_auc(V,K,seed=s) for s in range(3)])
    print(f"  V={V:.2f} (a=V/gK={V/(g*K):.3f} vs edge {edge:.3f})  peak-AUC={au:.2f}")

# (C) optimal-monitor second moment E0[L^2]=(1-a^2/gamma)^{-K/2}: diverges at V*_opt
print(f"\nOPTIMAL monitor E0[L^2] (K={K}, V*_opt={g*np.sqrt(K*gamma):.2f}):")
for V in [0.5,1.0,1.414,2.0,2.5]:
    a=V/(g*K); val=(1-a**2/gamma)**(-K/2) if a**2<gamma else np.inf
    Ksum=K*a**2/gamma  # ~ divergence control
    print(f"  V={V:.2f}  E0[L^2]={val:.3f}  (K a^2/gamma={Ksum:.2f}; wall at ~1)")
