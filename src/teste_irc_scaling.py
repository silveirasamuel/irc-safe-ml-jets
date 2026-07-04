#!/usr/bin/env python3
"""
Teste RIGOROSO de IRC safety: leis de escala no limite soft/colinear.

IR safety <=> o efeito de uma particula soft -> 0 quando seu pT -> 0.
   EFN:  <|dm|> propto pT_soft            -> 0     (SAFE)
   PFN:  <|dm|> -> constante != 0                  (UNSAFE: Phi(z=0,theta) fixo)

Colinear safety <=> dividir 1 particula em 2 colineares nao muda nada, p/ QUALQUER fracao z.
   EFN:  <|dm|> = 0  exatamente, para todo z       (SAFE)
   PFN:  <|dm|> != 0                                (UNSAFE)

Reusa os pesos treinados em efn.py (efn_pesos.pt), sem retreinar.
Roda no venv:  ./venv/bin/python teste_irc_scaling.py
"""
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PAD, L, RINV = 60, 128, 2.5
def mlp(s):
    l=[];
    for a,b in zip(s[:-1],s[1:]): l+=[nn.Linear(a,b), nn.ReLU()]
    return nn.Sequential(*l[:-1])
class EFN(nn.Module):
    def __init__(s): super().__init__(); s.Phi=mlp([2,128,128,128,L]); s.F=mlp([L,128,128,1])
    def forward(s,ang,z,m): return s.F(((z.unsqueeze(-1))*s.Phi(ang*RINV)).sum(1)).squeeze(-1)
class PFN(nn.Module):
    def __init__(s): super().__init__(); s.Phi=mlp([3,128,128,128,L]); s.F=mlp([L,128,128,1])
    def forward(s,ang,z,m):
        f=torch.cat([z.unsqueeze(-1),ang*RINV],-1)
        return s.F((s.Phi(f)*m.unsqueeze(-1)).sum(1)).squeeze(-1)

W = torch.load("results/efn_pesos.pt"); mu, sd = W["mu"], W["sd"]
efn=EFN(); efn.load_state_dict(W["efn"]); efn.eval()
pfn=PFN(); pfn.load_state_dict(W["pfn"]); pfn.eval()

def kin(p4):
    px,py,pz,E=p4[...,0],p4[...,1],p4[...,2],p4[...,3]
    pt=np.hypot(px,py); y=0.5*np.log(np.clip((E+pz)/np.clip(E-pz,1e-12,None),1e-12,None))
    return pt,y,np.arctan2(py,px)
def dphi(a,b): return np.arctan2(np.sin(a-b),np.cos(a-b))
def featurize(cp, ay, aphi):
    pt,y,phi=kin(cp); o=np.argsort(-pt)[:PAD]; pt,y,phi=pt[o],y[o],phi[o]
    z=pt/pt.sum(); n=len(pt)
    ang=np.zeros((PAD,2),np.float32); zz=np.zeros(PAD,np.float32); mk=np.zeros(PAD,np.float32)
    ang[:n,0]=y-ay; ang[:n,1]=dphi(phi,aphi); zz[:n]=z; mk[:n]=1.
    return ang,zz,mk
def pred(model, feats):
    A=torch.tensor(np.stack([f[0] for f in feats]))
    Z=torch.tensor(np.stack([f[1] for f in feats]))
    M=torch.tensor(np.stack([f[2] for f in feats]))
    with torch.no_grad(): return (sd*model(A,Z,M)+mu).numpy()

# ---- carrega jatos de teste ----
d=np.load("data/dados_jatos.npz",allow_pickle=True)
P=d["particles"].astype(np.float64); J=d["jets"].astype(np.float64)
JCO,JCI=d["jet_const_offsets"],d["jet_const_index"]
_,jy,jphi=kin(J)
NT=1500
gs=np.arange(len(J))[-NT:]                     # ultimos NT jatos (fora do treino tipico)
consts=[P[JCI[JCO[g]:JCO[g+1]]] for g in gs]
axis=[(jy[g],jphi[g]) for g in gs]
base_f=[featurize(c,*a) for c,a in zip(consts,axis)]
b_efn=pred(efn,base_f); b_pfn=pred(pfn,base_f)

rng=np.random.default_rng(2)
ang_soft=rng.uniform(-0.4,0.4,NT); phi_soft=rng.uniform(-np.pi,np.pi,NT)  # direcao fixa por jato

# ---- (1) escala IR: varia pT da particula soft ----
pts=np.logspace(-6,0,13)
ir_efn, ir_pfn=[],[]
for ptS in pts:
    fe=[]
    for k,(c,a) in enumerate(zip(consts,axis)):
        yy=a[0]+ang_soft[k]; ph=phi_soft[k]
        px,py,pz=ptS*np.cos(ph),ptS*np.sin(ph),ptS*np.sinh(yy)
        cp=np.vstack([c,[px,py,pz,np.sqrt(px*px+py*py+pz*pz)]])
        fe.append(featurize(cp,*a))
    ir_efn.append(np.mean(np.abs(pred(efn,fe)-b_efn)))
    ir_pfn.append(np.mean(np.abs(pred(pfn,fe)-b_pfn)))

# ---- (2) escala COLINEAR EM ANGULO: split da mais dura em 2 a angulo theta ----
#      (analogo fisico do teste soft: efeito -> 0 quando theta -> 0 para safe)
thetas=np.logspace(-4, np.log10(0.4), 13)
th_efn, th_pfn=[],[]
for th in thetas:
    fe=[]
    for c,a in zip(consts,axis):
        pt,y,phi=kin(c); i=int(np.argmax(pt)); pt0,y0,phi0=pt[i],y[i],phi[i]
        def daughter(yy):
            px,py_,pz=0.5*pt0*np.cos(phi0),0.5*pt0*np.sin(phi0),0.5*pt0*np.sinh(yy)
            return [px,py_,pz,np.sqrt(px*px+py_*py_+pz*pz)]
        cp=np.vstack([np.delete(c,i,0), daughter(y0+th/2), daughter(y0-th/2)])
        fe.append(featurize(cp,*a))
    th_efn.append(np.mean(np.abs(pred(efn,fe)-b_efn)))
    th_pfn.append(np.mean(np.abs(pred(pfn,fe)-b_pfn)))

# ---- (3) escala colinear em FRACAO z (split exatamente colinear, theta=0) ----
zs=np.array([0.5,0.2,0.1,0.05,0.02,0.01])
co_efn, co_pfn=[],[]
for zf in zs:
    fe=[]
    for c,a in zip(consts,axis):
        pt,_,_=kin(c); i=int(np.argmax(pt))
        cp=np.vstack([np.delete(c,i,0), zf*c[i], (1-zf)*c[i]])
        fe.append(featurize(cp,*a))
    co_efn.append(np.mean(np.abs(pred(efn,fe)-b_efn)))
    co_pfn.append(np.mean(np.abs(pred(pfn,fe)-b_pfn)))

# ---- ajuste de expoentes (regime de baixa perturbacao) ----
def fit_slope(x, yv, mask):
    lx=np.log(np.array(x)[mask]); ly=np.log(np.clip(np.array(yv)[mask],1e-30,None))
    return float(np.polyfit(lx, ly, 1)[0])
# janela limpa: acima do piso de float32, abaixo da saturacao
soft_slope =fit_slope(pts,    ir_efn, (pts>=1e-5)&(pts<=1e-2))
theta_slope=fit_slope(thetas, th_efn, (thetas>=1e-2)&(thetas<=1e-1))
pfn_soft_plateau =float(np.mean(np.array(ir_pfn)[pts<=1e-3]))
pfn_theta_plateau=float(np.mean(np.array(th_pfn)[thetas<=1e-3]))

# ---- resumo ----
print("== ESCALA IR (pT_soft -> <|dm|> GeV) ==")
for p,e,f in zip(pts,ir_efn,ir_pfn): print(f"  pT={p:.1e}  EFN={e:.2e}  PFN={f:.2e}")
print("== ESCALA COLINEAR-ANGULO (theta -> <|dm|> GeV) ==")
for t,e,f in zip(thetas,th_efn,th_pfn): print(f"  theta={t:.1e}  EFN={e:.2e}  PFN={f:.2e}")
print(f"\n>> EFN soft slope   = {soft_slope:.2f}  (esperado ~1)")
print(f">> EFN theta slope  = {theta_slope:.2f}  (esperado ~1)")
print(f">> PFN soft plateau = {pfn_soft_plateau:.2f} GeV ; PFN theta plateau = {pfn_theta_plateau:.2f} GeV")
import json as _json
_json.dump({"soft_slope":soft_slope,"theta_slope":theta_slope,
            "pfn_soft_plateau":pfn_soft_plateau,"pfn_theta_plateau":pfn_theta_plateau},
           open("results/scaling_fits.json","w"), indent=2)

# ---- figura (3 paineis: escala IR, escala colinear-angulo, exatidao em z) ----
plt.rcParams.update({"font.size":11,"figure.dpi":120})
fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(15,4.6))
FLOOR=1e-12; GRN,ORN="#1b9e77","#d95f02"
# (a1) escala IR
a1.loglog(pts,np.clip(ir_efn,FLOOR,None),"o-",color=GRN,label=f"EFN (safe), slope={soft_slope:.2f}")
a1.loglog(pts,np.clip(ir_pfn,FLOOR,None),"s-",color=ORN,label="PFN (unsafe)")
a1.loglog(pts, ir_efn[-1]*pts/pts[-1],"--",color=GRN,alpha=.5,label="∝ $p_T$ (expected if safe)")
a1.set_xlabel("soft particle $p_T$ [GeV]"); a1.set_ylabel("⟨|Δ predicted mass|⟩ [GeV]")
a1.set_title("IR safety: effect → 0 as $p_T$→0 ?"); a1.grid(alpha=.3,which="both"); a1.legend(frameon=False,fontsize=8.5)
# (a2) escala colinear em angulo
a2.loglog(thetas,np.clip(th_efn,FLOOR,None),"o-",color=GRN,label=f"EFN (safe), slope={theta_slope:.2f}")
a2.loglog(thetas,np.clip(th_pfn,FLOOR,None),"s-",color=ORN,label="PFN (unsafe)")
a2.loglog(thetas, th_efn[-1]*thetas/thetas[-1],"--",color=GRN,alpha=.5,label="∝ $θ$ (expected if safe)")
a2.set_xlabel("collinear opening angle $θ$"); a2.set_ylabel("⟨|Δ predicted mass|⟩ [GeV]")
a2.set_title("Collinear safety: effect → 0 as $θ$→0 ?"); a2.grid(alpha=.3,which="both"); a2.legend(frameon=False,fontsize=8.5)
# (a3) exatidao em fracao z (theta=0)
a3.semilogy(zs,np.clip(co_efn,FLOOR,None),"o-",color=GRN,label="EFN (safe)")
a3.semilogy(zs,np.clip(co_pfn,FLOOR,None),"s-",color=ORN,label="PFN (unsafe)")
a3.set_xlabel("collinear splitting fraction $z$"); a3.set_ylabel("⟨|Δ predicted mass|⟩ [GeV]")
a3.set_title("Exact collinear split ($θ{=}0$): invariant $\\forall z$ ?"); a3.grid(alpha=.3,which="both"); a3.legend(frameon=False,fontsize=8.5)
fig.suptitle("IRC safety as scaling laws: EFN → 0 in both soft ($∝p_T$) and collinear ($∝θ$) limits; PFN saturates at a plateau",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("figures/teste_irc_scaling.png",bbox_inches="tight")
print(">> figura salva em: teste_irc_scaling.png")
