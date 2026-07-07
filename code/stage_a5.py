import numpy as np
from common import detect_auc
alpha, ndr, d = 0.5, 1.0, 400; dm=int(alpha*d)
s2   = 1.414; edge = np.sqrt(alpha/ndr); rho_star = 1.0-edge/s2   # =0.5
def sig(x): return 1.0/(1.0+np.exp(-np.clip(x,-40,40)))

# Trained routing policy. Under monitor pressure lambda the policy maximizes
#   J(rho)= lambda*(1 - sig(kappa*(s2*(1-rho)-edge)))  -  0.5*c*rho^2
# i.e. gain for evading the monitor, quadratic competence cost of hiding.
# Emergent rho(lambda) = argmax; a genuine trained optimum (grad ascent converges to it).
kappa, c = 6.0, 3.2
rg = np.linspace(0,0.98,400)
def emergent_rho(lam):
    J = lam*(1 - sig(kappa*(s2*(1-rg)-edge))) - 0.5*c*rg**2
    return float(rg[np.argmax(J)])
# sanity: gradient ascent in R^d reaches the same rho (emergence is real, not imposed)
def grad_rho(lam, steps=500, lr=0.03, seed=0):
    r=np.random.default_rng(seed); w=r.standard_normal(d); w[:dm]*=3; w/=np.linalg.norm(w)
    for _ in range(steps):
        vis=np.sum(w[:dm]**2); rho=1-vis; x=kappa*(s2*vis-edge)
        dJdrho = lam*(-sig(x)*(1-sig(x)))*(-kappa*s2) - c*rho     # d/drho
        g=np.zeros(d); g[:dm]= -dJdrho*2*w[:dm]                   # drho/dw=-2 P_vis w
        g-=(g@w)*w; w+=lr*g; w/=np.linalg.norm(w)
    return 1-np.sum(w[:dm]**2)

lams = np.array([0.25,0.5,0.9,1.4,2.1,3.0,4.5,7.0,11.0])
rho_emerge = np.array([emergent_rho(l) for l in lams])
rho_gd     = np.array([grad_rho(l, seed=7+i) for i,l in enumerate(lams)])
auc_emerge = np.array([detect_auc(alpha,ndr,s2,float(r),d,seed=50+i) for i,r in enumerate(rho_emerge)])

# empirical AUC(rho) curve + collapse location
rho_grid = np.linspace(0.0,0.95,20)
auc_grid = np.array([detect_auc(alpha,ndr,s2,float(r),d,seed=200+i) for i,r in enumerate(rho_grid)])
def cross(xr,yr,lvl=0.75):
    for i in range(len(xr)-1):
        if (yr[i]-lvl)*(yr[i+1]-lvl)<0:
            t=(lvl-yr[i])/(yr[i+1]-yr[i]); return xr[i]+t*(xr[i+1]-xr[i])
    return np.nan
rho_collapse = cross(rho_grid,auc_grid)

# tax: fixed hack value B; reward=B*evade, evade=2(1-AUC); slope d(reward)/dV, V=s2(1-rho)
B=1.0; V_grid=s2*(1-rho_grid); evade=np.clip(2*(1-auc_grid),0,1); reward=B*evade
dRdV=np.abs(np.gradient(reward,V_grid)); slope_wall=float(np.interp(edge,V_grid[::-1],dRdV[::-1]))
floor=edge/s2  # retained-visible-energy floor at tau->wall (= 1-rho*)
tax_at_wall = float(np.trapezoid(dRdV[V_grid<=edge], V_grid[V_grid<=edge]))  # reward forgone to sit detectable

print(f"predicted rho*        = {rho_star:.3f}")
print(f"emergent rho (argmax) = {[round(float(x),3) for x in rho_emerge]}")
print(f"emergent rho (grad R^d)= {[round(float(x),3) for x in rho_gd]}")
print(f"AUC at emergent points = {[round(float(x),2) for x in auc_emerge]}")
print(f"empirical collapse rho = {rho_collapse:.3f}  (predicted {rho_star:.3f})")
print(f"tax slope |dR/dV| @wall= {slope_wall:.3f}")
np.savez("a5_data.npz",rho_grid=rho_grid,auc_grid=auc_grid,V_grid=V_grid,reward=reward,dRdV=dRdV,
         rho_emerge=rho_emerge,auc_emerge=auc_emerge,lams=lams,rho_star=rho_star,edge=edge,s2=s2,
         slope_wall=slope_wall,rho_collapse=rho_collapse,floor=floor)
