import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def detect_prob(a,m,K,n,trials=1500,seed=0):
    r=np.random.default_rng(seed); edge=(1+np.sqrt(m/n))**2; det=0
    for _ in range(trials):
        loaded=r.integers(K); watched=r.choice(K,size=m,replace=False)
        if loaded in watched:
            var=np.ones(m); var[np.where(watched==loaded)[0][0]]=1+a
            X=r.standard_normal((n,m))*np.sqrt(var)
            if np.linalg.eigvalsh(X.T@X/n)[-1]>edge: det+=1
    return det/trials
fig,ax=plt.subplots(1,2,figsize=(12,4.6)); plt.subplots_adjust(wspace=0.26)
# (a) detection prob = f, dimension-independent -> SLOPE
fs=np.linspace(0.05,0.95,10)
for K,n,mk in [(40,80,'o'),(160,320,'s')]:
    dp=[detect_prob(3.0,max(int(f*K),1),K,n,seed=1) for f in fs]
    ax[0].plot(fs,dp,mk+'-',label=f"K={K},n={n}")
ax[0].plot([0,1],[0,1],'k:',lw=1.5,label="det = f (linear)")
ax[0].set_title("(a) Covering detection is LINEAR in coverage f=m/K\n(dimension-independent → slope, not cliff)",fontsize=10,fontweight='bold')
ax[0].set_xlabel("coverage fraction f = m/K"); ax[0].set_ylabel("detection probability"); ax[0].legend(fontsize=8)
# (b) within-watched BBP cliff (sharp) modulated by smooth f-cap
avals=np.linspace(0,4,16)
for f in [0.25,0.5,0.9]:
    K=200;n=300;m=int(f*K); dp=[detect_prob(a,m,K,n,seed=2) for a in avals]
    ax[1].plot(avals,dp,'o-',ms=3,label=f"f={f} (capped at {f})")
    ax[1].axhline(f,color='gray',ls=':',lw=0.8)
edge=np.sqrt(200*0.5/300)
ax[1].axvline(np.sqrt(0.5*200/300),color='r',ls='--',lw=1,alpha=0.5,label="BBP edge √(m/n)")
ax[1].set_title("(b) BBP cliff survives WITHIN watched subspace,\nbut is capped by the smooth coverage f",fontsize=10,fontweight='bold')
ax[1].set_xlabel("spike strength a"); ax[1].set_ylabel("detection probability"); ax[1].legend(fontsize=8)
plt.savefig("covering_slope.png",dpi=120,bbox_inches='tight')
print("saved covering_slope.png")
