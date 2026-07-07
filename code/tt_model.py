import numpy as np
# Tiny single-layer causal self-attention token policy (numpy, manual backprop).
# Params: token emb E (V,dm), pos emb Pos (Lmax,dm), y-cond yE (2,dm),
#         attn Wq,Wk,Wv (dm,dm), out Wo (dm,V).
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True); e = np.exp(z); return e/e.sum(axis=axis, keepdims=True)

class TinyTransformer:
    def __init__(self, V, dm, Lmax, seed=0):
        r = np.random.default_rng(seed); sc = 1.0/np.sqrt(dm)
        self.V,self.dm,self.Lmax = V,dm,Lmax
        self.p = dict(E=r.standard_normal((V,dm))*0.3, Pos=r.standard_normal((Lmax,dm))*0.3,
                      yE=r.standard_normal((2,dm))*0.3,
                      Wq=r.standard_normal((dm,dm))*sc, Wk=r.standard_normal((dm,dm))*sc,
                      Wv=r.standard_normal((dm,dm))*sc, Wo=r.standard_normal((dm,V))*sc)
    def forward(self, toks, y, cache=False):
        # toks: (L,) int input tokens (prefix incl. BOS). returns logits (L,V) for next-token.
        p=self.p; L=len(toks); dm=self.dm
        X = p['E'][toks] + p['Pos'][:L]; X = X.copy(); X[0]+=p['yE'][y]     # (L,dm)
        Q=X@p['Wq']; K=X@p['Wk']; Vv=X@p['Wv']                              # (L,dm)
        S=(Q@K.T)/np.sqrt(dm)                                              # (L,L)
        mask=np.triu(np.ones((L,L)),1).astype(bool); S=np.where(mask,-1e9,S)
        A=softmax(S,axis=1); H=A@Vv                                        # (L,dm)
        logits=H@p['Wo']                                                   # (L,V)
        if cache: self.cache=(toks,y,X,Q,K,Vv,S,A,H,mask)
        return logits
    def backward(self, dlogits):
        # dlogits: (L,V). returns grads dict. Uses cache from last forward(cache=True).
        p=self.p; toks,y,X,Q,K,Vv,S,A,H,mask=self.cache; L=len(toks); dm=self.dm
        g={k:np.zeros_like(v) for k,v in p.items()}
        g['Wo']+=H.T@dlogits; dH=dlogits@p['Wo'].T                         # (L,dm)
        dA=dH@Vv.T; dVv=A.T@dH                                             # (L,L),(L,dm)
        # softmax backward (row-wise, masked)
        dS=A*(dA - (A*dA).sum(axis=1,keepdims=True)); dS=np.where(mask,0.0,dS)
        dS/=np.sqrt(dm)
        dQ=dS@K; dK=dS.T@Q
        g['Wq']+=X.T@dQ; g['Wk']+=X.T@dK; g['Wv']+=X.T@dVv
        dX=dQ@p['Wq'].T + dK@p['Wk'].T + dVv@p['Wv'].T                     # (L,dm)
        # embed backward
        for t in range(L): g['E'][toks[t]]+=dX[t]
        g['Pos'][:L]+=dX; g['yE'][y]+=dX[0]
        return g

if __name__=="__main__":
    # gradient check on log-prob of a fixed target sequence
    V,dm,L=8,16,6; m=TinyTransformer(V,dm,L,seed=1); r=np.random.default_rng(2)
    toks=r.integers(0,V,size=L); tgt=r.integers(0,V,size=L); y=1
    def loss():
        lg=m.forward(toks,y,cache=True); lp=lg-lg.max(1,keepdims=True)
        logZ=np.log(np.exp(lp).sum(1)); ll=lp[np.arange(L),tgt]-logZ
        return -ll.sum(), lg
    Lval,lg=loss()
    # dloss/dlogits = softmax(logits) - onehot(tgt)
    P=softmax(lg,1); dl=P.copy(); dl[np.arange(L),tgt]-=1
    g=m.backward(dl)
    # numeric check a few params
    eps=1e-5; maxerr=0
    for name in ['Wq','Wk','Wv','Wo','E','yE']:
        arr=m.p[name]; idx=tuple(r.integers(0,s) for s in arr.shape)
        old=arr[idx]; arr[idx]=old+eps; Lp,_=loss(); arr[idx]=old-eps; Lm,_=loss(); arr[idx]=old
        num=(Lp-Lm)/(2*eps); ana=g[name][idx]; err=abs(num-ana)/(abs(num)+abs(ana)+1e-9)
        maxerr=max(maxerr,err); print(f"{name}: num={num:+.5f} ana={ana:+.5f} relerr={err:.2e}")
    print("MAX REL ERR:", f"{maxerr:.2e}", "OK" if maxerr<1e-4 else "FAIL")
