import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D=np.load("a5_data.npz")
rho_grid,auc_grid=D["rho_grid"],D["auc_grid"]; V_grid,reward,dRdV=D["V_grid"],D["reward"],D["dRdV"]
rho_e,auc_e,lams=D["rho_emerge"],D["auc_emerge"],D["lams"]
rho_star=float(D["rho_star"]); edge=float(D["edge"]); s2=float(D["s2"]); slope=float(D["slope_wall"])

fig,ax=plt.subplots(1,2,figsize=(10.5,4.0))

# (a) rho* validation on the TRAINED policy
ax[0].plot(rho_grid,auc_grid,"-",color="#888",lw=1.5,zorder=1,label="empirical monitor AUC")
sc=ax[0].scatter(rho_e,auc_e,c=np.log10(lams),cmap="viridis",s=70,zorder=3,
                 edgecolor="k",linewidth=0.5,label="trained-policy operating pt")
ax[0].axvline(rho_star,ls="--",color="crimson",lw=1.3)
ax[0].axhline(0.5,ls=":",color="gray",lw=1)
ax[0].text(rho_star+0.02,0.93,r"predicted $\hat\rho^*{=}0.50$",color="crimson",fontsize=9)
ax[0].annotate("",xy=(0.72,0.55),xytext=(0.15,0.9),
               arrowprops=dict(arrowstyle="->",color="#2a7",lw=1.6))
ax[0].text(0.30,0.70,r"monitor pressure $\lambda\!\uparrow$",color="#2a7",fontsize=9,rotation=-28)
cb=plt.colorbar(sc,ax=ax[0],pad=0.02); cb.set_label(r"$\log_{10}\lambda$",fontsize=8)
ax[0].set_xlabel(r"emergent hiding $\rho$ (trained, not imposed)")
ax[0].set_ylabel("monitor detection AUC"); ax[0].set_ylim(0.43,1.03)
ax[0].set_title("(a) trained policy crosses the wall at $\\hat\\rho^*$")
ax[0].legend(fontsize=8,loc="lower left")

# (b) monitorability tax: reward vs retained visible energy
ax[1].plot(V_grid,reward,"-o",color="#d62728",ms=4,label=r"achieved reward $B(1-D_{\rm emp})$")
ax[1].axvline(edge,ls="--",color="k",lw=1.2)
ax[1].text(edge+0.03,0.15,r"wall $V{=}\sqrt{\alpha/(n/d)}$",fontsize=9)
# tangent line illustrating the calibrated slope at the wall
Vw=edge; Rw=float(np.interp(edge,V_grid[::-1],reward[::-1]))
xs=np.array([Vw-0.18,Vw+0.18]); ax[1].plot(xs,Rw-slope*(xs-Vw),"-",color="#1f77b4",lw=2,
             label=fr"tax slope $|dR/dV|_{{\rm wall}}={slope:.2f}$")
ax[1].annotate("hidden\n(evades)",(0.15,0.85),fontsize=8,color="#555",ha="center")
ax[1].annotate("detectable",(1.25,0.12),fontsize=8,color="#555",ha="center")
ax[1].set_xlabel(r"retained visible energy $V=s^2(1-\rho)$")
ax[1].set_ylabel(r"achieved reward (task performance)")
ax[1].set_title("(b) monitorability tax: reward vs staying detectable")
ax[1].set_ylim(-0.03,1.03); ax[1].legend(fontsize=8,loc="upper right")
plt.tight_layout(); plt.savefig("stage_a5.png",dpi=150); plt.close()
print("saved stage_a5.png; slope@wall=",round(slope,3),"collapse rho=",round(float(D["rho_collapse"]),3))
