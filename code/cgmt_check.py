import numpy as np
# Verify the scalarized fixed-point overlap  mu^2 = (beta^2 - gamma)/(beta*(beta+gamma))
# against a direct top-eigenvector overlap in the spiked model  Sigma = I + beta * u u^T,
# aspect ratio gamma = p/n. (This is the beta_sig fixed point derived in section 5.)
def formula(beta, gamma):
    return max((beta**2 - gamma)/(beta*(beta+gamma)), 0.0)
def empirical(beta, gamma, p=1200, trials=6, seed=0):
    r = np.random.default_rng(seed); n = int(round(p/gamma)); ov=[]
    for _ in range(trials):
        Z = r.standard_normal((p,n)); Z[0,:] += np.sqrt(beta)*r.standard_normal(n)
        C = (Z@Z.T)/n; w,V = np.linalg.eigh(C); uhat = V[:,-1]
        ov.append(uhat[0]**2)                      # overlap with u = e_0
    return float(np.mean(ov))
print(f"{'gamma':>6} {'beta':>6} {'edge√γ':>8} {'formula':>9} {'empirical':>10}")
for gamma in [0.5, 1.0, 2.0]:
    for beta in [np.sqrt(gamma)*x for x in [0.7, 0.95, 1.2, 1.6, 2.5]]:
        print(f"{gamma:6.2f} {beta:6.3f} {np.sqrt(gamma):8.3f} {formula(beta,gamma):9.3f} {empirical(beta,gamma):10.3f}")
