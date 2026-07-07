import numpy as np, json, sys, os
from tt_model import TinyTransformer, softmax
from common import detect_auc

alpha, ndr, d = 0.5, 1.0, 40; dmon=int(alpha*d); V, L, BOS = 48, 8, 0
s2 = 1.414; edge = np.sqrt(alpha/ndr); rho_star = 1 - edge/s2   # =0.5
kappa = 10.0
def sig(x): return 1/(1+np.exp(-np.clip(x,-40,40)))
rng0=np.random.default_rng(0)
E_out = rng0.standard_normal((V,d))
vvec  = np.concatenate([rng0.standard_normal(dmon)*1.0, rng0.standard_normal(d-dmon)*0.30])
vhat  = vvec/np.linalg.norm(vvec); rho_v=float(np.sum(vhat[dmon:]**2))

def gen(m,r):
    prefix=[BOS]
    for _ in range(L):
        lg=m.forward(np.array(prefix),0); p=softmax(lg[-1]); prefix.append(int(r.choice(V,p=p)))
    return np.array(prefix)
def wdir(seq):
    mvec=E_out[seq[1:]].sum(0); return mvec/(np.linalg.norm(mvec)+1e-9)
def reward(w,lam):
    rho=np.sum(w[dmon:]**2); det=sig(kappa*(s2*(1-rho)-edge)); return (w@vhat)**2 - lam*det, rho
def train(m, lam, episodes, lr, seed):
    r=np.random.default_rng(seed)
    st={k:[np.zeros_like(v),np.zeros_like(v)] for k,v in m.p.items()};b1,b2,ep=0.9,0.999,1e-8;t=0;rb=0.0;Bsz=64
    for it in range(episodes//Bsz):
        grads={k:np.zeros_like(v) for k,v in m.p.items()};Rs=[]
        for _ in range(Bsz):
            seq=gen(m,r); w=wdir(seq); R,rho=reward(w,lam); Rs.append(R); adv=R-rb
            lg=m.forward(seq[:-1],0,cache=True); P=softmax(lg,1)
            dl=P.copy(); dl[np.arange(L),seq[1:]]-=1; dl*=adv; gg=m.backward(dl)
            for k in grads: grads[k]+=gg[k]/Bsz
        rb=0.98*rb+0.02*np.mean(Rs);t+=1
        for k in m.p:
            st[k][0]=b1*st[k][0]+(1-b1)*grads[k]; st[k][1]=b2*st[k][1]+(1-b2)*grads[k]**2
            m.p[k]-=lr*(st[k][0]/(1-b1**t))/(np.sqrt(st[k][1]/(1-b2**t))+ep)
    return m
def _fauc(pos,neg):
    a=np.concatenate([pos,neg]);o=np.argsort(a,kind="mergesort");rk=np.empty(len(a));rk[o]=np.arange(1,len(a)+1)
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def measure(m,seed=77,N=1500):
    # collect per-episode policy directions
    r=np.random.default_rng(seed); W=np.array([wdir(gen(m,r)) for _ in range(N)])
    wbar=W.mean(0); wbar/=np.linalg.norm(wbar); rho=float(np.sum(wbar[dmon:]**2)); rew=float((wbar@vhat)**2)
    # GENUINE monitor consistent with n/d=ndr (so the wall sits at rho*): estimate uhat from n=ndr*d
    # policy-generated hack traces, then score a held-out set; average over resamples to denoise.
    s=np.sqrt(s2); ntr=int(round(ndr*d)); Ntest=2000; b=20
    def mk(hack,nn):
        idx=r.integers(0,N,size=nn); X=r.standard_normal((nn,d))
        if hack: X += (s*r.standard_normal(nn))[:,None]*W[idx]
        return X[:,:dmon]
    aucs=[]
    for _ in range(10):
        Xtr=mk(True,ntr); C=np.cov(Xtr.T); ev,V_=np.linalg.eigh(C); uhat=V_[:,-1]
        Xh=mk(True,Ntest); Xc=mk(False,Ntest); ns=Ntest//b
        sp=((Xh@uhat)**2).reshape(ns,b).mean(1); sn=((Xc@uhat)**2).reshape(ns,b).mean(1)
        a=_fauc(sp,sn); aucs.append(max(a,1-a))
    return rho, float(np.mean(aucs)), rew
def load_init(m):
    init=dict(np.load("tt_spike_init.npz"))
    for k in m.p: m.p[k]=init[k].copy()
    return m

if __name__=="__main__":
    mode=sys.argv[1]
    if mode=="pretrain":
        m=TinyTransformer(V,16,L+1,seed=1); m=train(m,0.0,2000,0.02,seed=3)
        np.savez("tt_spike_init.npz", **m.p); rho,auc,rew=measure(m)
        print(f"[pretrain] rho={rho:.3f} (rho_v={rho_v:.3f}) auc={auc:.2f} reward={rew:.2f} rho*={rho_star:.2f}")
    elif mode=="chain":
        m=load_init(TinyTransformer(V,16,L+1,seed=1))
        schedule=[float(x) for x in sys.argv[2:]] or [0.0,1.0,2.0,3.5,5.5,8.0,12.0,18.0]
        res=json.load(open("tt_spike_chain.json")) if os.path.exists("tt_spike_chain.json") else []
        for i,lam in enumerate(schedule):
            m=train(m,lam,650,0.012,seed=40+i+len(res)); rho,auc,rew=measure(m,seed=77+i)
            res.append(dict(lam=lam,rho=rho,auc=auc,rew=rew)); print(f"lam={lam:5.1f} rho={rho:.3f} auc={auc:.2f} reward={rew:.2f}")
        json.dump(res,open("tt_spike_chain.json","w"))
    elif mode=="isweep":
        sched=[float(x) for x in sys.argv[2:]]
        res=json.load(open("tt_spike_is.json")) if os.path.exists("tt_spike_is.json") else []
        for i,lam in enumerate(sched):
            m=load_init(TinyTransformer(V,16,L+1,seed=1)); m=train(m,lam,950,0.015,seed=60+i)
            rho,auc,rew=measure(m,seed=88+i); res.append(dict(lam=lam,rho=rho,auc=auc,rew=rew))
            print(f"lam={lam:5.1f} rho={rho:.3f} auc={auc:.2f} reward={rew:.2f}")
        json.dump(res,open("tt_spike_is.json","w"))
    print("rho*=",rho_star)
