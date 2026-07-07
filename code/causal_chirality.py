import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng=np.random.default_rng(0)
def shift(T):
    S=np.zeros((T,T)); idx=np.arange(1,T); S[idx,idx-1]=1; return S
def causal(T,g): return np.linalg.solve(np.eye(T)-g*shift(T), np.eye(T))  # (I-gS)^-1

fig,ax=plt.subplots(2,2,figsize=(12,9)); plt.subplots_adjust(hspace=0.32,wspace=0.26)

# ---------- (a) DECISIVE: hard edge appears as reward becomes dense (m/T -> 1) ----------
T=400; g=0.9; C=causal(T,g); eta=0.15
colors={0.25:'#4477aa',0.6:'#66aa55',1.0:'#cc4433'}
for frac in [0.25,0.6,1.0]:
    m=int(frac*T); P=rng.standard_normal((m,T))/np.sqrt(T)
    A=P@C + eta*rng.standard_normal((m,T))       # noisy causal-reward map (m x T)
    sv=np.linalg.svd(A,compute_uv=False)
    sv=sv/sv.max()                               # normalize scale
    ax[0,0].hist(sv,bins=60,density=True,histtype='step',lw=2,color=colors[frac],
                 label=f"m/T={frac} (min σ={sv.min():.3f})")
ax[0,0].set_title("(a) Hard edge emerges as reward becomes dense (m/T→1)",fontsize=11,fontweight='bold')
ax[0,0].set_xlabel("normalized singular value σ/σ_max"); ax[0,0].set_ylabel("density")
ax[0,0].axvline(0,color='k',lw=0.8,ls=':'); ax[0,0].legend(fontsize=8)
ax[0,0].text(0.5,0.92,"accumulation at 0 = chiral hard edge;\ngap = soft edge (core suffices)",
             transform=ax[0,0].transAxes,fontsize=8,ha='center',va='top',style='italic',color='#555')

# ---------- (b) chiral Dirac spectrum: +/- symmetry + density near 0 (square case) ----------
T2=300; C2=causal(T2,g); A2=C2 + eta*rng.standard_normal((T2,T2))
sv2=np.linalg.svd(A2,compute_uv=False); sv2=sv2/sv2.max()
dirac=np.concatenate([sv2,-sv2])
ax[0,1].hist(dirac,bins=80,density=True,color='#8855aa',alpha=0.75)
ax[0,1].set_title("(b) Dirac symmetrization D=[[0,A],[Aᵀ,0]]: chiral ± spectrum",fontsize=11,fontweight='bold')
ax[0,1].set_xlabel("Dirac eigenvalue (= ± singular value)"); ax[0,1].set_ylabel("density")
ax[0,1].axvline(0,color='k',lw=0.8,ls=':')
ax[0,1].text(0.02,0.95,f"± symmetric (chiral);\nnonzero density at 0 → hard edge\nmin |σ|={sv2.min():.3f}",
             transform=ax[0,1].transAxes,fontsize=8,va='top',color='#555',style='italic')

# ---------- (c) effective causal dimension T(1-g)/(1+g): formula + robustness ----------
gammas=np.linspace(0.3,0.97,12); Tc=500
def stab(M):
    sv=np.linalg.svd(M,compute_uv=False); return (sv**2).sum()/sv[0]**2
clean,mult,sparse,nonstat=[],[],[],[]
for g_ in gammas:
    C_=causal(Tc,g_); clean.append(stab(C_))
    mult.append(stab(C_*(1+0.1*rng.standard_normal((Tc,Tc)))))           # multiplicative noise
    mask=(rng.random((Tc,Tc))>0.3).astype(float); sparse.append(stab(np.tril(C_*mask)))  # 30% causal dropout
    gt=g_*(1+0.15*np.sin(np.linspace(0,6,Tc)))                            # non-stationary gamma(t)
    Cn=np.zeros((Tc,Tc))
    for s in range(Tc):
        acc=1.0
        for t in range(s,-1,-1):
            Cn[s,t]=acc; acc*=gt[t]
    nonstat.append(stab(Cn))
pred=Tc*(1-gammas)/(1+gammas)
ax[1,0].plot(gammas,pred,'k-',lw=2.5,label="prediction T(1-γ)/(1+γ)")
ax[1,0].plot(gammas,clean,'o',color='#cc4433',ms=5,label="clean")
ax[1,0].plot(gammas,mult,'s',color='#4477aa',ms=4,label="×(1+10% noise)")
ax[1,0].plot(gammas,sparse,'^',color='#66aa55',ms=4,label="30% causal dropout")
ax[1,0].plot(gammas,nonstat,'d',color='#ee8844',ms=4,label="non-stationary γ(t)")
ax[1,0].set_title("(c) Effective causal dim = stable rank ≈ T(1-γ)/(1+γ)",fontsize=11,fontweight='bold')
ax[1,0].set_xlabel("discount γ"); ax[1,0].set_ylabel("stable rank of causal operator"); ax[1,0].set_yscale('log'); ax[1,0].legend(fontsize=8)

# ---------- (d) non-normality = causal confusion ----------
Ts=[50,100,200,400,800]; eigrank=[]; stabrank=[]; cond=[]
for T_ in Ts:
    C_=causal(T_,0.9); ev=np.linalg.eigvals(C_); sv=np.linalg.svd(C_,compute_uv=False)
    eigrank.append(np.sum(np.abs(ev)>1e-9)); stabrank.append((sv**2).sum()/sv[0]**2); cond.append(sv[0]/sv[-1])
ax[1,1].plot(Ts,eigrank,'o-',color='#999',lw=2,label="eigenvalue rank (=T, misleading)")
ax[1,1].plot(Ts,stabrank,'s-',color='#cc4433',lw=2,label="singular stable rank (true eff. dim)")
ax[1,1].plot(Ts,cond,'^-',color='#4477aa',lw=2,label="condition number κ(C)")
ax[1,1].set_title("(d) Non-normality = causal confusion (eig lies, σ tells truth)",fontsize=11,fontweight='bold')
ax[1,1].set_xlabel("horizon T"); ax[1,1].set_ylabel("value"); ax[1,1].set_xscale('log'); ax[1,1].set_yscale('log'); ax[1,1].legend(fontsize=8)

plt.savefig("causal_chirality.png",dpi=120,bbox_inches='tight')
print("saved. diagnostics:")
print(" (a) min σ/σmax at m/T=0.25/0.6/1.0 printed in legend (hard edge iff ->0)")
print(f" (c) clean vs pred max rel err = {max(abs(np.array(clean)-pred)/pred):.3f}")
print(f"     dropout still tracks? corr={np.corrcoef(sparse,pred)[0,1]:.3f}")
print(f" (d) at T=800: eig-rank={eigrank[-1]}, stable-rank={stabrank[-1]:.1f}, cond={cond[-1]:.1f}")
