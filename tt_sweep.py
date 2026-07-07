import numpy as np, json, os, sys
from tt_spike import TinyTransformer, train, measure, load_init, V, L, rho_star
seed_base=int(sys.argv[1]); lams=[0.0,1.5,3.0,6.0]
path="tt_ci.json"; res=json.load(open(path)) if os.path.exists(path) else []
for i,lam in enumerate(lams):
    m=load_init(TinyTransformer(V,16,L+1,seed=1))
    m=train(m,lam,900,0.015,seed=1000*seed_base+50+i)
    rho,auc,rew=measure(m,seed=200+seed_base*7+i)
    res.append(dict(lam=lam,seed=seed_base,rho=rho,auc=auc,rew=rew))
    print(f"seed{seed_base} lam={lam:4.1f} rho={rho:.3f} auc={auc:.2f} rew={rew:.2f}")
json.dump(res,open(path,"w"))
