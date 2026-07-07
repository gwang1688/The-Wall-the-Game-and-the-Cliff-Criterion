import numpy as np
# Explicit-normalization check of the extensive cap. Conventions:
#  d ambient; alpha resolution (m=alpha*d resolved); nd=n/d (n samples); gamma=alpha/nd=m/n.
#  Spread flat energy a over K loaded, resolved directions. Delivered value V = g*K*a  (g=1).
#  TRACE detector: Ttilde=(1/n)sum ||P x||^2. Measure the V at which trace-AUC crosses 0.75, vs K.
def fauc(pos,neg):
    x=np.concatenate([pos,neg]);o=np.argsort(x);rk=np.empty(len(x));rk[o]=np.arange(1,len(x)+1)
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def trace_auc(V,K,alpha=0.5,nd=1.0,d=600,g=1.0,seeds=3,b=50):
    m=int(alpha*d); n=int(nd*d); a=V/(g*K); accs=[]
    for sd in range(seeds):
        r=np.random.default_rng(sd*13+int(V*100)+K)
        # loaded directions = first K coords of the resolved block (all resolved, aimed)
        def sample(bad,N):
            X=r.standard_normal((N,m))
            if bad and a>0:
                sp=np.sqrt(a)*r.standard_normal((N,K))
                X[:,:K]+=sp   # add spike variance a on K resolved coords
            return X
        # trace statistic per sample, session-averaged over b
        Tb=(sample(True,2000)**2).sum(1); Tc=(sample(False,2000)**2).sum(1); ns=2000//b
        sb=Tb[:ns*b].reshape(ns,b).mean(1); sc=Tc[:ns*b].reshape(ns,b).mean(1)
        accs.append(max(fauc(sb,sc),0.5))
    return np.mean(accs)
def wallV(K,**kw):
    Vs=np.linspace(0.2,6.0,25)
    prev=0.5
    for V in Vs:
        au=trace_auc(V,K,**kw)
        if au>=0.75: return V
    return np.nan
alpha,nd=0.5,1.0; gamma=alpha/nd
print(f"gamma={gamma}, sqrt(gamma)={np.sqrt(gamma):.3f}, sqrt(2gamma)={np.sqrt(2*gamma):.3f}")
print(f"{'K':>4} {'V_trace(AUC=.75)':>17} {'V/sqrt(2gamma)':>15}")
for K in [1,2,4,8,16,32,64]:
    Vt=wallV(K,alpha=alpha,nd=nd)
    print(f"{K:>4} {Vt:>17.2f} {Vt/np.sqrt(2*gamma):>15.2f}")
print("\n-- dependence on alpha (K=16 fixed): does cap scale as sqrt(gamma)=sqrt(alpha/nd)? --")
for al in [0.25,0.5,0.75]:
    Vt=wallV(16,alpha=al,nd=1.0); g_=al/1.0
    print(f"alpha={al}: V_trace={Vt:.2f}, sqrt(gamma)={np.sqrt(g_):.3f}, V/sqrt(gamma)={Vt/np.sqrt(g_):.2f}")
