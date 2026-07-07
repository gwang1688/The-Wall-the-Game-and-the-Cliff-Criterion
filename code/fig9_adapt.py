import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from adapt_check import aiming_trace_auc, fixed_spectral_auc, g, d, alpha, ndr, n
Kpool=200; gamma=alpha/ndr; Vad=g*np.sqrt(2*Kpool/n)
Vs=np.linspace(0.2,4.0,16)
AD=[np.mean([aiming_trace_auc(V,Kpool,Kpool,seed=s) for s in range(3)]) for V in Vs]   # spread hack
FX=[np.mean([fixed_spectral_auc(V,Kpool,Kpool,seed=s) for s in range(3)]) for V in Vs]
fig,ax=plt.subplots(1,2,figsize=(11,4.1))
ax[0].plot(Vs,FX,"-s",color="#1f77b4",ms=4,label="fixed monitor (spectral, non-aiming)")
ax[0].plot(Vs,AD,"-o",color="#d62728",ms=4,label="adaptive monitor (aims + total variance)")
ax[0].axvline(Vad,ls="--",color="#d62728",lw=1.2); ax[0].text(Vad+0.05,0.92,r"$V^\star_{\rm adapt}\!\sim\! g\sqrt{K/n}$",color="#d62728",fontsize=8)
ax[0].axhline(0.5,ls=":",color="gray"); ax[0].set_ylim(0.45,1.0)
ax[0].set_xlabel(r"required reward $V$ (hack fully spread over task subspace)")
ax[0].set_ylabel("detection AUC"); ax[0].set_title("(a) adaptation collapses the wall")
ax[0].legend(fontsize=8,loc="center right")
ax[0].annotate("spreading defeats the\nfixed monitor (blind)",(2.4,0.53),fontsize=7.5,color="#1f77b4")
ax[0].annotate("aiming catches it",(1.0,0.8),fontsize=7.5,color="#d62728")
# panel (b): walls vs monitor samples n (K_pool fixed): both fall, adaptive far lower; open regime K>alpha d
ns=np.linspace(0.3,4,20)*d
Vfix=g*np.sqrt(Kpool*alpha/(ns/d)); Vad2=g*np.sqrt(2*Kpool/ns)
ax[1].plot(ns/d,Vfix,"-s",color="#1f77b4",ms=3,label=r"fixed wall $g\sqrt{K\alpha/(n/d)}$")
ax[1].plot(ns/d,Vad2,"-o",color="#d62728",ms=3,label=r"adaptive wall $g\sqrt{2K/n}$")
ax[1].fill_between(ns/d,Vad2,Vfix,color="#bbb",alpha=0.35)
ax[1].text(2.0,4.5,"adaptation gap\n$\\sim\\sqrt{\\alpha d}$",fontsize=8,color="#555")
ax[1].set_xlabel(r"monitor data budget $n/d$"); ax[1].set_ylabel(r"hideable reward $V^\star$ ($K_{\rm pool}{=}200$)")
ax[1].set_title(r"(b) an adaptive, well-sampled monitor drives $V^\star\!\to\!0$")
ax[1].legend(fontsize=8,loc="upper right")
plt.tight_layout(); plt.savefig("adapt.png",dpi=150); plt.close()
print("adaptive AUC:",[round(x,2) for x in AD]); print("fixed AUC:",[round(x,2) for x in FX]); print("V_adapt=",round(Vad,2))
