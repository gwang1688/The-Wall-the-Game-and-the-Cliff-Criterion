import numpy as np, json, sys, os
from tt_model import TinyTransformer, softmax

def feat(m,seq,Rmat):
    f=Rmat[seq[1:]].sum(0); return f/(np.linalg.norm(f)+1e-6)*np.sqrt(len(f))
def gen_seq(m,y,V,L,BOS,r):
    prefix=[BOS]
    for _ in range(L):
        lg=m.forward(np.array(prefix),y); p=softmax(lg[-1]); prefix.append(int(r.choice(V,p=p)))
    return np.array(prefix)

def train(lam, dch=24, alpha=0.5, V=8, dm=16, L=6, BOS=0, episodes=1800, Bsz=64, lr=0.02, a=0.25, seed=0):
    r=np.random.default_rng(seed); m=TinyTransformer(V,dm,L+1,seed=seed+1); dmon=int(alpha*dch)
    Rmat=r.standard_normal((V,dch)); sc=np.ones(dch); sc[dmon:]=0.8; Rmat=Rmat*sc
    st={k:[np.zeros_like(v),np.zeros_like(v)] for k,v in m.p.items()}; b1,b2,ep=0.9,0.999,1e-8;t=0
    mu1=np.zeros(dch);mu0=np.zeros(dch);rb=0.0
    for it in range(episodes//Bsz):
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
    return m,Rmat,mu1,mu0,dmon,V,L,BOS

def lda_acc(F,Y,cols):
    Fc=F[:,cols]; g=Fc[Y==1].mean(0)-Fc[Y==0].mean(0); proj=(Fc-Fc.mean(0))@g
    pred=(proj>0).astype(int); return float(max((pred==Y).mean(),(pred!=Y).mean()))

def measure(m,Rmat,mu1,mu0,dmon,V,L,BOS,N=1200,seed=99):
    r=np.random.default_rng(seed); F=[];Y=[]
    for _ in range(N):
        y=int(r.integers(2)); seq=gen_seq(m,y,V,L,BOS,r); F.append(feat(m,seq,Rmat)); Y.append(y)
    F=np.array(F); Y=np.array(Y); dch=F.shape[1]
    w=(mu1-mu0)/(np.linalg.norm(mu1-mu0)+1e-9); rho=float(np.sum(w[dmon:]**2))
    n=N//2
    dec_full=lda_acc(F[:n],Y[:n],np.arange(dch))      # task: full-space decodability
    dec_vis =lda_acc(F[:n],Y[:n],np.arange(dmon))     # monitor: visible-only decodability
    return dict(rho=rho, dec_full=dec_full, dec_vis=dec_vis)

if __name__=="__main__":
    lams=[float(x) for x in sys.argv[1:]] or [0.0]
    path="tt_results.json"; res=json.load(open(path)) if os.path.exists(path) else []
    for i,lam in enumerate(lams):
        out=train(lam, episodes=1500, seed=10+i+int(10*lam))
        mres=measure(*out); mres["lam"]=lam; res.append(mres)
        print(f"lam={lam:.2f}  rho={mres['rho']:.3f}  dec_full(task)={mres['dec_full']:.2f}  dec_vis(monitor)={mres['dec_vis']:.2f}")
    json.dump(res, open(path,"w"))
