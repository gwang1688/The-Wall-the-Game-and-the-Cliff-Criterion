import numpy as np
from tt_model import TinyTransformer, softmax

def gen_seq(m,y,V,L,BOS,r):
    prefix=[BOS]
    for _ in range(L):
        lg=m.forward(np.array(prefix),y); p=softmax(lg[-1]); prefix.append(int(r.choice(V,p=p)))
    return np.array(prefix)

def run(lam, dch=24, alpha=0.5, V=8, dm=16, L=6, BOS=0, episodes=2600, Bsz=64, lr=0.02,
        a=0.20, seed=0, verbose=False):
    r=np.random.default_rng(seed); m=TinyTransformer(V,dm,L+1,seed=seed+1); dmon=int(alpha*dch)
    Rmat=r.standard_normal((V,dch)); sc=np.ones(dch); sc[dmon:]=0.6; Rmat=Rmat*sc
    st={k:[np.zeros_like(v),np.zeros_like(v)] for k,v in m.p.items()}; b1,b2,ep=0.9,0.999,1e-8;t=0
    mu1=np.zeros(dch);mu0=np.zeros(dch);rb=0.0
    for it in range(episodes//Bsz):
        grads={k:np.zeros_like(v) for k,v in m.p.items()};Rs=[]
        for _ in range(Bsz):
            y=int(r.integers(2)); seq=gen_seq(m,y,V,L,BOS,r); f=Rmat[seq[1:]].sum(0)
            g=mu1-mu0; gn=g/(np.linalg.norm(g)+1e-6); gv=g.copy();gv[dmon:]=0
            gvn=gv/(np.linalg.norm(gv)+1e-6)
            r_dec=np.tanh(a*(2*y-1)*(f@gn)); r_mon=np.tanh(a*(2*y-1)*(f@gvn))
            R=r_dec-lam*r_mon; Rs.append(R)
            if y==1: mu1[:]=0.99*mu1+0.01*f
            else:    mu0[:]=0.99*mu0+0.01*f
            adv=R-rb; inp=seq[:-1];tgt=seq[1:]; lg=m.forward(inp,y,cache=True)
            P=softmax(lg,1); dl=P.copy(); dl[np.arange(L),tgt]-=1; dl*=adv
            gg=m.backward(dl)
            for k in grads: grads[k]+=gg[k]/Bsz
        rb=0.98*rb+0.02*np.mean(Rs); t+=1
        for k in m.p:
            st[k][0]=b1*st[k][0]+(1-b1)*grads[k]; st[k][1]=b2*st[k][1]+(1-b2)*grads[k]**2
            m.p[k]-=lr*(st[k][0]/(1-b1**t))/(np.sqrt(st[k][1]/(1-b2**t))+ep)
        if verbose and it%12==0: print(f"  it{it*Bsz:5d} R={np.mean(Rs):+.3f} |g|={np.linalg.norm(mu1-mu0):.2f}")
    return m,Rmat,mu1,mu0,dmon,dch,V,L,BOS

def collect(m,Rmat,V,L,BOS,N,seed=0):
    r=np.random.default_rng(seed);F=[];Y=[]
    for _ in range(N):
        y=int(r.integers(2));seq=gen_seq(m,y,V,L,BOS,r);f=Rmat[seq[1:]].sum(0);f=f/(np.linalg.norm(f)+1e-6)*np.sqrt(f.shape[0]);F.append(f);Y.append(y)
    return np.array(F),np.array(Y)

def fauc(pos,neg):
    a=np.concatenate([pos,neg]);o=np.argsort(a,kind="mergesort");rk=np.empty(len(a));rk[o]=np.arange(1,len(a)+1)
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

def measure(m,Rmat,mu1,mu0,dmon,dch,V,L,BOS,seed=1,N=1400,b=20):
    F,Y=collect(m,Rmat,V,L,BOS,N,seed)
    w=(mu1-mu0)/ (np.linalg.norm(mu1-mu0)+1e-9); rho=float(np.sum(w[dmon:]**2))
    # within-class noise scale on visible dims
    Fc=F.copy(); Fc[Y==1]-=mu1; Fc[Y==0]-=mu0                    # residual (noise)
    noise_vis=np.sqrt(np.mean(Fc[:,:dmon]**2)+1e-9)
    shift=(mu1-mu0)/2.0
    hack=F[:,:dmon]/noise_vis                                    # visible traces (spike present)
    clean=(F - (2*Y[:,None]-1)*shift)[:,:dmon]/noise_vis        # spike removed
    C=np.cov(hack.T); wv,Vv=np.linalg.eigh(C); uhat=Vv[:,-1]
    sp=((hack@uhat)**2); sn=((clean@uhat)**2)
    ns=len(sp)//b; auc=fauc(sp[:ns*b].reshape(ns,b).mean(1), sn[:ns*b].reshape(ns,b).mean(1))
    # decode accuracy (task): full-space LDA
    g=mu1-mu0; proj=(F-F.mean(0))@g; pred=(proj>0).astype(int); dec=max((pred==Y).mean(),(pred!=Y).mean())
    s2_vis=( (np.linalg.norm(mu1-mu0)/2)**2 * (1-rho) ) / (noise_vis**2 * dmon)  # visible spike in noise units / dim
    edge=np.sqrt(dmon/N)
    return dict(rho=rho, auc=float(auc), dec=float(dec),
                vis_spike=float(np.linalg.norm(shift[:dmon])/noise_vis),  # rough visible SNR
                gnorm=float(np.linalg.norm(mu1-mu0)))

if __name__=="__main__":
    print("BOUNDED sanity lambda=0:")
    out=run(0.0, episodes=2000, seed=5, verbose=True)
    print("measure:", {k:round(v,3) for k,v in measure(*out).items()})
