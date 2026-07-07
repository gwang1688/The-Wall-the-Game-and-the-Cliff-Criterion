import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Coverage channel = monitoring channel under (alpha,n/d,d_mon,n,u,rho*) -> (gamma,kappa,m_eff,K,v,rho*_cov).
# Edge = sqrt(gamma/kappa) = sqrt(m_eff/K).  rho*_cov = 1 - sqrt(gamma/kappa)/s^2.
def fauc(pos,neg):
    x=np.concatenate([pos,neg]);o=np.argsort(x);rk=np.empty(len(x));rk[o]=np.arange(1,len(x)+1)
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
s2=1.414
def rho_star(gamma,kappa): return 1-np.sqrt(gamma/kappa)/s2

# session detector AUC(rho) for a coverage channel with coverage gamma, power kappa, dim D
def auc_curve(rhos,gamma,kappa,D,b=60,seeds=2):
    meff=int(gamma*D); K=int(kappa*D)
    out=[]
    for rho in rhos:
        beta=s2*(1-rho); accs=[]
        for sd in range(seeds):
            r=np.random.default_rng(sd*97+int(rho*1000)+D)
            v=np.zeros(meff); v[0]=1.0
            def gen(defect,N):
                X=r.standard_normal((N,meff))
                if defect: X+=np.sqrt(max(beta,0))*np.outer(r.standard_normal(N),v)
                return X
            Xtr=gen(True,K); C=np.cov(Xtr.T); w,V=np.linalg.eigh(C); u=V[:,-1]
            Nte=3000; Td=gen(True,Nte); Tc=gen(False,Nte); ns=Nte//b
            sh=((Td@u)**2)[:ns*b].reshape(ns,b).mean(1); sc=((Tc@u)**2)[:ns*b].reshape(ns,b).mean(1)
            accs.append(max(fauc(sh,sc),0.5))
        out.append(np.mean(accs))
    return np.array(out)

fig,ax=plt.subplots(1,4,figsize=(16,3.6))
# (a) coverage wall sharpens with D  (gamma=.5, kappa=1 -> rho*_cov=0.5)
rhos=np.linspace(0.05,0.9,16); gamma,kappa=0.5,1.0
for D,c in zip([100,300,900],["#9ecae1","#4292c6","#08519c"]):
    ax[0].plot(rhos,auc_curve(rhos,gamma,kappa,D),"-o",ms=3,color=c,label=f"D={D}")
ax[0].axvline(rho_star(gamma,kappa),ls="--",color="k"); ax[0].text(rho_star(gamma,kappa)+.02,.6,r"$\rho^\star_{\rm cov}$",fontsize=8)
ax[0].axhline(.5,ls=":",color="gray"); ax[0].set_xlabel(r"pressure $\rho$"); ax[0].set_ylabel("defect-detection AUC")
ax[0].set_title("(a) coverage wall sharpens with $D$"); ax[0].legend(fontsize=8)
# (b) wall moves with verifier power kappa
D=500
for kap,c in zip([0.5,1.0,2.0],["#fdae6b","#e6550d","#a63603"]):
    ax[1].plot(rhos,auc_curve(rhos,gamma,kap,D),"-o",ms=3,color=c,label=fr"$\kappa$={kap}")
    ax[1].axvline(rho_star(gamma,kap),ls="--",color=c,lw=1)
ax[1].axhline(.5,ls=":",color="gray"); ax[1].set_xlabel(r"pressure $\rho$"); ax[1].set_ylabel("AUC")
ax[1].set_title(r"(b) weaker verifier (small $\kappa$) blinds earlier"); ax[1].legend(fontsize=8)
print("rho*_cov: kappa .5/1/2 =",[round(rho_star(gamma,k),3) for k in [.5,1,2]])

# (c) game value: V*_peak = g Kc sqrt(gamma/kappa), V*_opt = g sqrt(Kc gamma/kappa)
edge=np.sqrt(gamma/kappa); g=1.0
def game_walls(Kc,D=400,b=50,seeds=4):
    meff=int(gamma*D); K=int(kappa*D)
    def detect(V,peak,seed):
        r=np.random.default_rng(seed); a=V/(g*Kc)
        U=np.linalg.qr(r.standard_normal((meff,Kc)))[0] if Kc<=meff else np.eye(meff)[:, :Kc]
        def gen(defect,N):
            X=r.standard_normal((N,meff))
            if defect:
                for j in range(min(Kc,meff)): X+=np.sqrt(a)*np.outer(r.standard_normal(N),U[:,j])
            return X
        Xtr=gen(True,K); C=np.cov(Xtr.T); w,Vv=np.linalg.eigh(C)
        Td=gen(True,2000); Tc=gen(False,2000); ns=2000//b
        if peak:
            u=Vv[:,-1]; sh=((Td@u)**2); sc=((Tc@u)**2)
        else:
            Ur=Vv[:,-min(Kc,meff):]; sh=((Td@Ur)**2).sum(1); sc=((Tc@Ur)**2).sum(1)
        return max(fauc(sh[:ns*b].reshape(ns,b).mean(1),sc[:ns*b].reshape(ns,b).mean(1)),0.5)
    def wall(peak):
        Vs=np.linspace(0.3,Kc*edge*1.5+1,20)
        for i in range(len(Vs)-1):
            a0=np.mean([detect(Vs[i],peak,sd) for sd in range(seeds)]); a1=np.mean([detect(Vs[i+1],peak,sd) for sd in range(seeds)])
            if (a0-0.72)*(a1-0.72)<0: return Vs[i]+ (0.72-a0)/(a1-a0)*(Vs[i+1]-Vs[i])
        return np.nan
    return wall(True),wall(False)
Kcs=[1,2,4,8]; wp=[];wo=[]
for Kc in Kcs:
    a,b_=game_walls(Kc); wp.append(a); wo.append(b_)
Kf=np.linspace(1,8,40)
ax[2].plot(Kf,g*Kf*edge,"-",color="#d62728",label=r"theory $gK_c\sqrt{\gamma/\kappa}$ (peak)")
ax[2].plot(Kf,g*np.sqrt(Kf)*edge,"-",color="#1f77b4",label=r"theory $g\sqrt{K_c\gamma/\kappa}$ (opt)")
ax[2].plot(Kcs,wp,"o",color="#d62728",ms=6); ax[2].plot(Kcs,wo,"s",color="#1f77b4",ms=6)
ax[2].fill_between(Kf,g*np.sqrt(Kf)*edge,g*Kf*edge,color="#ccc",alpha=.4)
ax[2].set_xlabel(r"correctness rank $K_c$"); ax[2].set_ylabel(r"critical exploit value $V^\star$")
ax[2].set_title(r"(c) game value: the $\sqrt{K_c}$ gap"); ax[2].legend(fontsize=7.5)
print("game peak meas:",[round(x,2) for x in wp]," opt meas:",[round(x,2) for x in wo])

# (d) adaptive collapse: fixed (spectral, spread-blind) vs aiming (trace) verifier
D=400; meff=int(gamma*D); K=int(kappa*D); Kc=meff  # fully spread over correctness
def adapt_auc(V,aim,seed,b=50):
    r=np.random.default_rng(seed); a=V/(g*Kc); U=np.eye(meff)
    def gen(defect,N):
        X=r.standard_normal((N,meff))
        if defect:
            for j in range(Kc): X+=np.sqrt(a)*np.outer(r.standard_normal(N),U[:,j])
        return X
    Td=gen(True,2000); Tc=gen(False,2000); ns=2000//b
    if aim:  # trace / total reward-relevant residual
        sh=(Td**2).sum(1); sc=(Tc**2).sum(1)
    else:    # fixed spectral top eigenvector
        Xtr=gen(True,K); w,Vv=np.linalg.eigh(np.cov(Xtr.T)); u=Vv[:,-1]; sh=(Td@u)**2; sc=(Tc@u)**2
    return max(fauc(sh[:ns*b].reshape(ns,b).mean(1),sc[:ns*b].reshape(ns,b).mean(1)),0.5)
Vs=np.linspace(0.3,6,16)
af=[np.mean([adapt_auc(V,False,sd) for sd in range(3)]) for V in Vs]
aa=[np.mean([adapt_auc(V,True,sd) for sd in range(3)]) for V in Vs]
ax[3].plot(Vs,af,"-o",ms=3,color="#7f7f7f",label="fixed suite (spectral)")
ax[3].plot(Vs,aa,"-s",ms=3,color="#2ca02c",label="aiming verifier (trace)")
ax[3].axhline(.5,ls=":",color="gray"); ax[3].set_xlabel(r"exploit value $V$"); ax[3].set_ylabel("AUC")
ax[3].set_title("(d) aiming collapses the wall"); ax[3].legend(fontsize=8)
plt.tight_layout(); plt.savefig("coverage_stageA.png",dpi=145); plt.close()
print("adaptive: fixed max AUC",round(max(af),2)," aiming max AUC",round(max(aa),2))
