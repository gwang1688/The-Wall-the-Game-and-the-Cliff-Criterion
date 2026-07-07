import numpy as np
# Converse (Le Cam second moment). Spiked Wishart: H0 z~N(0,I_p); H1 z~N(0,I+beta u u^T), n iid.
# Bayesian LR second moment:  E0[L_n^2] = E_{u,u'}[(1 - beta^2 rho^2)^{-n/2}],  rho=<u,u'>.
# Claim: with gamma=p/n, as p,n->inf,  E0[L_n^2] -> (1 - beta^2/gamma)^{-1/2},
# finite iff beta < sqrt(gamma) (= the BBP/estimation wall), diverging as beta -> sqrt(gamma).
def closed_form(beta, gamma):
    x = beta**2/gamma
    return np.inf if x>=1 else (1-x)**-0.5
def mc_second_moment(beta, gamma, p=6000, M=400000, seed=0):
    r = np.random.default_rng(seed); n = int(round(p/gamma))
    # rho = <u,u'> for independent uniform unit vectors ~ N(0,1/p) for large p; sample directly
    rho = r.standard_normal(M)/np.sqrt(p)
    val = (1.0 - (beta**2)*(rho**2))
    val = np.where(val>0, val, np.nan)               # guard (rare) boundary
    logterm = -0.5*n*np.log(val)
    m = np.nanmax(logterm)                            # log-sum-exp for stability
    est = np.exp(m)*np.nanmean(np.exp(logterm-m))
    return est
print(f"{'gamma':>6} {'beta/√γ':>8} {'closed':>9} {'MonteCarlo':>11}")
for gamma in [0.5, 1.0, 2.0]:
    for frac in [0.4, 0.6, 0.8]:
        beta = frac*np.sqrt(gamma)
        print(f"{gamma:6.2f} {frac:8.2f} {closed_form(beta,gamma):9.4f} {mc_second_moment(beta,gamma):11.4f}")
    # at/above threshold -> should blow up
    for frac in [0.95, 1.0, 1.05]:
        beta=frac*np.sqrt(gamma)
        cf = closed_form(beta,gamma)
        print(f"{gamma:6.2f} {frac:8.2f} {('inf' if cf==np.inf else f'{cf:.3f}'):>9} {mc_second_moment(beta,gamma):11.4f}   <-- near/over wall")
