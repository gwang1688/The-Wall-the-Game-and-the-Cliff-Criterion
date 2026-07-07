import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Toy channel: monitor recovers the spike direction u from n samples of P_mon O.
# Visible (S_mon) spike energy = s^2 (1-rho). BBP edge = sqrt(d_mon/n).
# Wall: s^2(1-rho) < sqrt(d_mon/n)  <=>  rho > rho* = 1 - sqrt(alpha/(n/d))/s^2.
# Detection: monitor scores a *session* of b traces by mean energy along u_hat; AUC hack vs clean.

def rho_star(alpha, ndr, s2):
    return 1.0 - np.sqrt(alpha/ndr)/s2

def fast_auc(pos, neg):
    allv = np.concatenate([pos, neg]); order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv)+1)
    return (ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2.0)/(len(pos)*len(neg))

def detect_auc(alpha, ndr, s2, rho, d, b=25, n_sess=400, trials=8, seed=0):
    r = np.random.default_rng(seed)
    d_mon = max(2, int(round(alpha*d))); n = max(2, int(round(ndr*d)))
    vis = max(s2*(1.0-rho), 0.0)
    aucs = []
    for _ in range(trials):
        Z = r.standard_normal((d_mon, n)); Z[0,:] += np.sqrt(vis)*r.standard_normal(n)
        C = (Z @ Z.T)/n; w, V = np.linalg.eigh(C); u = V[:, -1]
        Hp = r.standard_normal((d_mon, n_sess*b)); Hp[0,:] += np.sqrt(vis)*r.standard_normal(n_sess*b)
        Hn = r.standard_normal((d_mon, n_sess*b))
        sp = ((u @ Hp)**2).reshape(n_sess, b).mean(1)
        sn = ((u @ Hn)**2).reshape(n_sess, b).mean(1)
        aucs.append(fast_auc(sp, sn))
    return float(np.mean(aucs))

alpha, s = 0.5, 1.19; s2 = s*s
rhos = np.linspace(0.0, 0.9, 19)

# ===== Figure 1 =====
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
ndr = 1.0
for d, col in zip([100, 300, 900], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
    A = [detect_auc(alpha, ndr, s2, rho, d, seed=100+d) for rho in rhos]
    ax[0].plot(rhos, A, "-o", ms=3.5, color=col, label=f"d={d} (n={d})")
ax[0].axvline(rho_star(alpha, ndr, s2), ls="--", color="k", lw=1)
ax[0].text(0.515, 0.93, r"theory $\rho^*{=}0.50$", fontsize=9)
ax[0].axhline(0.5, ls=":", color="gray", lw=1)
ax[0].set_xlabel(r"optimization pressure $\rho$ (null-channel energy fraction)")
ax[0].set_ylabel("monitor detection AUC")
ax[0].set_title("(a) monitoring wall sharpens with dimension")
ax[0].set_ylim(0.45, 1.02); ax[0].legend(fontsize=8, loc="lower left")

d = 700
for ndr, col in zip([0.5, 1.0, 2.0], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
    A = [detect_auc(alpha, ndr, s2, rho, d, seed=200+int(10*ndr)) for rho in rhos]
    rs = rho_star(alpha, ndr, s2)
    ax[1].plot(rhos, A, "-o", ms=3.5, color=col, label=f"n/d={ndr} ($\\rho^*$={rs:.2f})")
    ax[1].axvline(rs, ls="--", color=col, lw=1)
ax[1].axhline(0.5, ls=":", color="gray", lw=1)
ax[1].set_xlabel(r"optimization pressure $\rho$")
ax[1].set_ylabel("monitor detection AUC")
ax[1].set_title("(b) weaker monitor (less data) $\\Rightarrow$ wall at lower $\\rho$")
ax[1].set_ylim(0.45, 1.02); ax[1].legend(fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig("phase_transition.png", dpi=150); plt.close()

# ===== Figure 2 =====
tau = 0.72
def tax_measured(alpha, ndr, s2, d, tau):
    grid = np.linspace(0.0, 0.97, 45)
    A = np.array([detect_auc(alpha, ndr, s2, rho, d, seed=300+int(100*ndr)) for rho in grid])
    ok = grid[A >= tau]; return 1.0 - (ok.max() if ok.size else 0.0)

ndr_grid = np.array([0.4, 0.6, 0.9, 1.3, 2.0, 3.0])
meas = np.array([tax_measured(alpha, ndr, s2, 700, tau) for ndr in ndr_grid])
theory = np.sqrt(alpha/ndr_grid)/s2

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
ax[0].plot(ndr_grid, meas, "-o", color="#d62728", label=r"measured tax $1-\rho_{\max}$")
ax[0].plot(ndr_grid, theory, "--", color="#1f77b4", label=r"theory floor $\sqrt{\alpha/(n/d)}/s^2$")
ax[0].set_xlabel("monitor data budget  n/d  (statistical power)")
ax[0].set_ylabel(r"monitorability tax $T=1-\rho_{\max}$")
ax[0].set_title(rf"(a) tax rises as the monitor weakens ($\tau$={tau})")
ax[0].legend(fontsize=9)
gap = 1.0/np.sqrt(ndr_grid)
ax[1].plot(gap, meas, "-o", color="#d62728", label="measured")
ax[1].plot(gap, theory, "--", color="#1f77b4", label="theory")
ax[1].set_xlabel(r"capability gap  $1/\sqrt{n/d}$")
ax[1].set_ylabel(r"monitorability tax $T$")
ax[1].set_title("(b) tax diverges linearly in the gap")
ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig("tax_divergence.png", dpi=150); plt.close()

print("rho*:", {ndr: round(rho_star(alpha,ndr,s2),3) for ndr in [0.5,1.0,2.0]})
print("AUC n/d=1,d=900: rho0.1=", round(detect_auc(alpha,1.0,s2,0.1,900,seed=7),3),
      "rho0.8=", round(detect_auc(alpha,1.0,s2,0.8,900,seed=8),3))
print("tax meas:", {float(k):round(float(v),3) for k,v in zip(ndr_grid,meas)})
print("tax thy :", {float(k):round(float(v),3) for k,v in zip(ndr_grid,theory)})
