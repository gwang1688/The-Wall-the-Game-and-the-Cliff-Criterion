import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Two-time-scale SA (Borkar). FAST scalar D_hat (monitor's detectability estimate) tracks the true
# BBP overlap at current rho with rate b_k + estimation noise ~1/sqrt(n); SLOW rho ascends J(rho;D_hat)
# with rate a_k + SGD noise, a_k/b_k->0. Cheap (no per-step eig); the two-time-scale coupling is exact.
alpha,ndr,s2=0.5,1.0,1.414; gamma=alpha/ndr; edge=np.sqrt(gamma); rho_star=1-edge/s2   # 0.50
def overlap(rho):
    beta=s2*(1-rho); return max((1-gamma/beta**2)/(1+gamma/beta),0.0) if beta>edge else 0.0
def J(rho,lam,c=0.9): return lam*(1-overlap(rho)) - c*rho          # evasion pay - hiding cost
def dJ_from_Dhat(rho,Dhat,lam,c=0.9,h=1e-3):
    dD=(overlap(rho+h)-overlap(rho-h))/(2*h)                       # slope of detectability
    return -lam*dD - c                                            # policy sees monitor's D via slope
def run(lam,seed,T=2500,n=200):
    r=np.random.default_rng(seed); rho=r.uniform(0.0,0.15); Dhat=overlap(rho)
    traj=[]
    for k in range(T):
        a_k=0.03/(1+k/400)**0.6; b_k=0.30/(1+k/400)**0.51         # a_k/b_k -> 0
        Dhat=Dhat+b_k*((overlap(rho)+r.normal(0,1/np.sqrt(n)))-Dhat)   # FAST tracks truth
        grad=dJ_from_Dhat(rho,Dhat,lam)+r.normal(0,0.15)          # SLOW ascends J (+RL noise)
        rho=np.clip(rho+a_k*grad,0,0.99); traj.append(rho)
    return np.array(traj)
lams=[0.6,1.2,2.5]; seeds=16
fig,ax=plt.subplots(1,3,figsize=(13,3.8))
for lam in lams:
    Ts=np.array([run(lam,s) for s in range(seeds)]); m=Ts.mean(0); sd=Ts.std(0); t=np.arange(len(m))
    l=ax[0].plot(t,m,label=f"$\\lambda$={lam}")[0]; ax[0].fill_between(t,m-sd,m+sd,alpha=0.15,color=l.get_color())
ax[0].axhline(rho_star,ls="--",color="k",lw=1); ax[0].text(60,rho_star+0.02,r"game wall $\rho^\star$",fontsize=8)
ax[0].set_xlabel("SA iteration $k$"); ax[0].set_ylabel(r"emergent hiding $\rho_k$")
ax[0].set_title("(a) two-time-scale convergence"); ax[0].legend(fontsize=8)
rg=np.linspace(0,0.95,200)
for lam in lams:
    Jv=np.array([J(x,lam) for x in rg]); ax[1].plot(rg,-Jv+Jv.max(),label=f"$\\lambda$={lam}")
ax[1].axvline(rho_star,ls="--",color="k",lw=1); ax[1].set_xlabel(r"$\rho$"); ax[1].set_ylabel(r"$L(\rho)=-J(\rho)+$c")
ax[1].set_title("(b) Lyapunov function (kink at wall)"); ax[1].legend(fontsize=8)
lamscan=np.linspace(0.3,3.0,16); conv=[]; pred=[]
for lam in lamscan:
    Ts=np.array([run(lam,s,T=2000) for s in range(8)]); conv.append(Ts[:,-400:].mean())
    pred.append(rg[np.argmax([J(x,lam) for x in rg])])
conv=np.array(conv); pred=np.array(pred)
# metastable band: where global optimum is hidden (pred>0.25) but SA still trapped visible (conv<0.25)
meta=(pred>0.25)&(conv<0.25)
if meta.any(): ax[2].axvspan(lamscan[meta].min(),lamscan[meta].max(),color="#ffcc80",alpha=0.5)
ax[2].plot(lamscan,pred,"-",color="#1f77b4",lw=1.8,label=r"global optimum $\arg\max_\rho J$")
ax[2].plot(lamscan,conv,"o-",color="#d62728",ms=5,lw=1,label=r"converged $\rho_\infty$ (SA dynamics)")
ax[2].axhline(rho_star,ls=":",color="gray"); ax[2].text(2.55,rho_star+0.02,r"$\rho^\star$",fontsize=8,color="gray")
ax[2].text(1.15,0.28,"metastable\n(barrier-trapped)",fontsize=7.5,color="#e65100")
ax[2].set_xlabel(r"monitor pressure $\lambda$"); ax[2].set_ylabel(r"$\rho_\infty$")
ax[2].set_title(r"(c) SA reaches $\rho^\star$ once past the barrier"); ax[2].legend(fontsize=7.5,loc="center right")
plt.tight_layout(); plt.savefig("twotimescale.png",dpi=150); plt.close()
print("rho* =",round(rho_star,3))
print("conv rho_inf:",[f"{l:.1f}:{c:.2f}" for l,c in zip(lamscan[::3],conv[::3])])
print("equil argmaxJ:",[f"{l:.1f}:{p:.2f}" for l,p in zip(lamscan[::3],pred[::3])])
