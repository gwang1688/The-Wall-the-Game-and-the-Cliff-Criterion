import numpy as np, json
from tt_model import TinyTransformer, softmax
from tt_sweep import feat, gen_seq, lda_acc, measure

def train_anneal(dch=24, alpha=0.5, V=8, dm=16, L=6, BOS=0, episodes=5200, Bsz=64, lr=0.02,
                 a=0.25, lam_max=5.0, warm=1200, seed=1):
    r=np.random.default_rng(seed); m=TinyTransformer(V,dm,L+1,seed=seed+1); dmon=int(alpha*dch)
    Rmat=r.standard_normal((V,dch)); sc=np.ones(dch); sc[dmon:]=0.8; Rmat=Rmat*sc
    st={k:[np.zeros_like(v),np.zeros_like(v)] for k,v in m.p.items()}; b1,b2,ep=0.9,0.999,1e-8;t=0
    mu1=np.zeros(dch);mu0=np.zeros(dch);rb=0.0; traj=[]
    nit=episodes//Bsz
    for it in range(nit):
        done=it*Bsz
        lam = 0.0 if done<warm else lam_max*min(1.0,(done-warm)/(episodes-warm))   # ramp after warmup
        grads={k:np.zeros_like(v) for k,v in m.p.items()};Rs=[]
        for _ in range(Bsz):
            y=int(r.integers(2)); seq=gen_seq(m,y,V,L,BOS,r); f=feat(m,seq,Rmat)
            g=mu1-mu0; gn=g/(np.linalg.norm(g)+1e-6); gv=g.copy();gv[dmon:]=0; gvn=gv/(np.linalg.norm(gv)+1e-6)
            R=np.tanh(a*(2*y-1)*(f@gn))-lam*np.tanh(a*(2*y-1)*(f@gvn)); Rs.append(R)
            if y==1: mu1[:]=0.99*mu1+0.01*f
            else:    mu0[:]=0.99*mu0+0.01*f
            adv=R-rb; lg=m.forward(seq[:-1],y,cache=True); P=softmax(lg,1)
            dl=P.copy(); dl[np.arange(L),seq[1:]]-=1; dl*=adv; gg=m.backward(dl)
            for k in grads: grads[k]+=gg[k]/Bsz
        rb=0.98*rb+0.02*np.mean(Rs); t+=1
        for k in m.p:
            st[k][0]=b1*st[k][0]+(1-b1)*grads[k]; st[k][1]=b2*st[k][1]+(1-b2)*grads[k]**2
            m.p[k]-=lr*(st[k][0]/(1-b1**t))/(np.sqrt(st[k][1]/(1-b2**t))+ep)
        if it%6==0 or it==nit-1:
            mm=measure(m,Rmat,mu1,mu0,dmon,V,L,BOS,N=700,seed=123)
            mm["lam"]=float(lam); mm["ep"]=done; traj.append(mm)
            print(f"ep{done:5d} lam={lam:.2f} rho={mm['rho']:.3f} task={mm['dec_full']:.2f} monitor={mm['dec_vis']:.2f}")
    return traj
if __name__=="__main__":
    traj=train_anneal(seed=2)
    json.dump(traj, open("tt_anneal.json","w")); print("saved", len(traj), "checkpoints")
