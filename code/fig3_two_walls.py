import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# Two order parameters share one wall at beta/sqrt(gamma)=1 (fix gamma=1 so x=beta).
gamma = 1.0
def overlap_cf(x):                    # estimation order parameter mu^2 = (b^2-g)/(b(b+g)), g=gamma
    b = x*np.sqrt(gamma); v = (b**2-gamma)/(b*(b+gamma)); return np.maximum(v,0.0)
def secondmoment_cf(x):               # detection order parameter E0[L^2] = (1 - beta^2/gamma)^{-1/2}
    return (1.0 - x**2)**-0.5
def overlap_sim(x, p=1500, trials=6, seed=0):
    r=np.random.default_rng(seed); b=x*np.sqrt(gamma); n=int(round(p/gamma)); ov=[]
    for _ in range(trials):
        Z=r.standard_normal((p,n)); Z[0,:]+=np.sqrt(b)*r.standard_normal(n)
        C=(Z@Z.T)/n; w,V=np.linalg.eigh(C); ov.append(V[0,-1]**2)
    return float(np.mean(ov))
def secondmoment_sim(x, p=4000, M=400000, seed=1):
    r=np.random.default_rng(seed); b=x*np.sqrt(gamma); n=int(round(p/gamma))
    rho=r.standard_normal(M)/np.sqrt(p); val=1.0-(b**2)*(rho**2); val=np.where(val>0,val,np.nan)
    lt=-0.5*n*np.log(val); m=np.nanmax(lt); return float(np.exp(m)*np.nanmean(np.exp(lt-m)))

fig,ax=plt.subplots(1,2,figsize=(10.5,4.0))

# (a) estimation wall: overlap
xa=np.linspace(0.02,2.2,120)
ax[0].plot(xa,overlap_cf(xa),"-",color="#1f77b4",lw=2,label=r"theory $\langle\hat u,u\rangle^2$")
xs=np.array([0.5,0.8,1.2,1.5,1.9])
ax[0].plot(xs,[overlap_sim(x,seed=100+int(10*x)) for x in xs],"o",color="#1f77b4",ms=6,
           mfc="white",label="simulation")
ax[0].axvline(1.0,ls="--",color="k",lw=1.2); ax[0].axhline(0,ls=":",color="gray",lw=1)
ax[0].text(1.03,0.9,r"wall $\beta{=}\sqrt{\gamma}$",fontsize=9)
ax[0].set_xlabel(r"signal / threshold   $\beta/\sqrt{\gamma}$")
ax[0].set_ylabel(r"recovery overlap $\langle\hat u,u\rangle^2$")
ax[0].set_title("(a) estimation wall  (Prop. 1)"); ax[0].set_ylim(-0.03,1.0)
ax[0].legend(fontsize=9,loc="upper left")
ax[0].annotate("blind\n(overlap 0)",(0.45,0.05),fontsize=8,color="#555",ha="center")
ax[0].annotate("recoverable",(1.75,0.55),fontsize=8,color="#555",ha="center")

# (b) detection wall: second moment
xb=np.linspace(0.02,0.985,120)
ax[1].plot(xb,secondmoment_cf(xb),"-",color="#d62728",lw=2,label=r"theory $(1-\beta^2/\gamma)^{-1/2}$")
xs2=np.array([0.2,0.4,0.6,0.8,0.9])
ax[1].plot(xs2,[secondmoment_sim(x,seed=200+int(10*x)) for x in xs2],"s",color="#d62728",ms=6,
           mfc="white",label="simulation")
ax[1].axvline(1.0,ls="--",color="k",lw=1.2)
ax[1].text(1.02,4.6,r"wall $\beta{=}\sqrt{\gamma}$",fontsize=9)
ax[1].annotate(r"$\to\infty$",(0.965,5.0),fontsize=11,color="#d62728")
ax[1].set_xlabel(r"signal / threshold   $\beta/\sqrt{\gamma}$")
ax[1].set_ylabel(r"LR second moment $\mathbb{E}_0[L_n^2]$")
ax[1].set_title("(b) detection wall  (Prop. 2, converse)"); ax[1].set_ylim(1.0,5.5); ax[1].set_xlim(0,2.2)
ax[1].legend(fontsize=9,loc="upper left")
ax[1].annotate("contiguous\n(no consistent test)",(0.5,1.4),fontsize=8,color="#555",ha="center")
ax[1].annotate("detectable",(1.7,3.0),fontsize=8,color="#555",ha="center")
plt.tight_layout(); plt.savefig("two_walls.png",dpi=150); plt.close()

print("overlap sim:", {float(x):round(overlap_sim(x,seed=100+int(10*x)),3) for x in xs})
print("2nd-moment sim vs cf:", {float(x):(round(secondmoment_sim(x,seed=200+int(10*x)),3),
      round(secondmoment_cf(x),3)) for x in xs2})
