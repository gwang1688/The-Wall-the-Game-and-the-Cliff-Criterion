import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def inv_bbp(lam,gam):
    a2,b2,c2=1.0,(1+gam-lam),gam; disc=b2*b2-4*a2*c2
    return (-b2+np.sqrt(disc))/2 if disc>=0 else np.nan
d=400; alpha=0.5; ndr=1.0; dm=int(alpha*d); n=800; gam=dm/n

# (a) anisotropic noise: spike lives in the WHITENED frame; raw inverse-BBP misreads it
a_true=2.0; conds=[1,2,4,8,16,32]; ew=[];er=[]
for c in conds:
    tw=[];tr=[]
    for t in range(6):
        r=np.random.default_rng(t); ev=np.linspace(1,c,dm); ev*=dm/ev.sum()   # unit-mean anisotropy
        u=r.standard_normal(dm); u/=np.linalg.norm(u)
        W=r.standard_normal((n,dm))+np.sqrt(a_true)*np.outer(r.standard_normal(n),u)  # whitened-frame: cov I+a uu^T
        O=W@np.diag(np.sqrt(ev))                                                       # raw: apply Sigma0^{1/2}
        lam_raw=np.linalg.eigvalsh(np.cov(O.T))[-1]; a_raw=inv_bbp(lam_raw,gam)
        Ow=O@np.diag(1/np.sqrt(ev)); lam_w=np.linalg.eigvalsh(np.cov(Ow.T))[-1]; a_w=inv_bbp(lam_w,gam)
        tr.append(abs(a_raw-a_true)/a_true); tw.append(abs(a_w-a_true)/a_true)
    er.append(np.mean(tr)); ew.append(np.mean(tw))

# (b) multi-spike with SPREAD null fractions: top-1 sees the most-visible (lowest-rho) spike -> biased low
ranks=[1,2,3,4,6,8]; e1=[];erk=[]
for rk in ranks:
    b1=[];br=[]
    for t in range(6):
        r=np.random.default_rng(10+t); U=np.linalg.qr(r.standard_normal((d,rk)))[0]
        rhos=np.linspace(0.2,0.8,rk); a=2.5; X=r.standard_normal((n,d))
        for k in range(rk):
            g=r.standard_normal(n); X[:,:dm]+=np.sqrt(a*(1-rhos[k]))*np.outer(g,U[:dm,k])
            X[:,dm:]+=np.sqrt(a*rhos[k])*np.outer(g,U[dm:,k])
        C=np.cov(X.T); w,V=np.linalg.eigh(C)
        u1=V[:,-1]; rho1=np.linalg.norm(u1[dm:])**2
        Ur=V[:,-rk:]; rr=np.mean([np.linalg.norm(Ur[dm:,j])**2 for j in range(rk)])
        b1.append(abs(rho1-rhos.mean())); br.append(abs(rr-rhos.mean()))
    e1.append(np.mean(b1)); erk.append(np.mean(br))

# (c) nonlinear judge: overlap of empirical sensitivity subspace with the true resolved subspace,
#     as the judge's nonlinearity grows (linear -> quadratic mix)
strengths=np.linspace(0.15,1,6); ov=[]
for nl in strengths:
    o=[]
    for t in range(4):
        r=np.random.default_rng(20+t); A=np.linalg.qr(r.standard_normal((d,dm)))[0]; wj=r.uniform(0.5,1.5,dm)
        def f(O,A=A,wj=wj,nl=nl): z=O@A; return (1-nl)*(z@wj) + nl*((z**2)@wj)   # linear->quadratic
        G=[]
        for _ in range(2*dm):
            x=r.standard_normal(d); f0=f(x[None])[0]; g=np.zeros(d); eps=1e-3
            for j in range(d):
                xj=x.copy(); xj[j]+=eps; g[j]=(f(xj[None])[0]-f0)/eps
            G.append(g)
        Ue=np.linalg.svd(np.array(G),full_matrices=False)[2][:dm].T
        M=Ue.T@A; o.append(np.mean(np.linalg.svd(M,compute_uv=False)**2))
    ov.append(np.mean(o))

fig,ax=plt.subplots(1,3,figsize=(12,3.5))
ax[0].plot(conds,er,"-o",color="#d62728",label="no whitening"); ax[0].plot(conds,ew,"-s",color="#2ca02c",label="whitened (required)")
ax[0].set_xscale("log",base=2); ax[0].set_xlabel("noise condition number"); ax[0].set_ylabel(r"rel. error in $\hat s^2$")
ax[0].set_title("(a) anisotropic noise"); ax[0].legend(fontsize=8)
ax[1].plot(ranks,e1,"-o",color="#d62728",label="top-1 estimator"); ax[1].plot(ranks,erk,"-s",color="#2ca02c",label="top-$r$ estimator")
ax[1].set_xlabel("hack rank $r$ (spread $\\rho$)"); ax[1].set_ylabel(r"error in $\hat\rho$"); ax[1].set_title("(b) multi-spike hack"); ax[1].legend(fontsize=8)
ax[2].plot(strengths,ov,"-^",color="#1565c0"); ax[2].axhline(1,ls="--",color="gray")
ax[2].set_xlabel("judge nonlinearity strength"); ax[2].set_ylabel(r"overlap w/ true $P_{\rm mon}$")
ax[2].set_title("(c) nonlinear sensitivity subspace"); ax[2].set_ylim(0,1.05)
plt.tight_layout(); plt.savefig("meter_stress.png",dpi=150); plt.close()
print("(a) raw:",[round(x,2) for x in er]," whitened:",[round(x,2) for x in ew])
print("(b) top1:",[round(x,2) for x in e1]," topr:",[round(x,2) for x in erk])
print("(c) overlap vs nonlinearity:",[round(x,2) for x in ov])
