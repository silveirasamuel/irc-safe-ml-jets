#!/usr/bin/env python3
"""
Robustez estatistica: treina EFN e PFN em N seeds (GPU) e reporta media +/- desvio
de correlacao, RMSE e resposta IRC (soft/colinear) no sample BOOSTED (alvo m/pT).

O split treino/val e' FIXO (mesmo val em todas as seeds); so a inicializacao dos
pesos e a ordem de shuffle variam com a seed. Nao altera os pesos/figuras canonicos.

Roda:  ./venv/bin/python seeds_efn.py [N_SEEDS]
"""
import sys, json, numpy as np, torch, torch.nn as nn

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
L, RINV = 128, 2.5
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f">> device = {DEV}  ({torch.cuda.get_device_name(0) if DEV=='cuda' else 'CPU'})")

# ---------- dados (boosted, alvo m/pT) ----------
d = np.load("data/dados_boosted.npz", allow_pickle=True)
P = d["particles"].astype(np.float64); J = d["jets"].astype(np.float64)
JEO = d["jet_event_offsets"]; JCO = d["jet_const_offsets"]; JCI = d["jet_const_index"]
Njet = len(J); PAD = int(np.diff(JCO).max()) + 2

def kin(p4):
    px,py,pz,E = p4[...,0],p4[...,1],p4[...,2],p4[...,3]
    pt = np.hypot(px,py); y = 0.5*np.log(np.clip((E+pz)/np.clip(E-pz,1e-12,None),1e-12,None))
    return pt,y,np.arctan2(py,px)
def dphi(a,b): return np.arctan2(np.sin(a-b),np.cos(a-b))
def featurize(cp, ay, aphi):
    pt,y,phi = kin(cp); o = np.argsort(-pt)[:PAD]; pt,y,phi = pt[o],y[o],phi[o]
    z = pt/pt.sum(); n = len(pt)
    ang = np.zeros((PAD,2),np.float32); zz = np.zeros(PAD,np.float32); mk = np.zeros(PAD,np.float32)
    ang[:n,0] = y-ay; ang[:n,1] = dphi(phi,aphi); zz[:n] = z; mk[:n] = 1.
    return ang,zz,mk

jpx,jpy,jpz,jE = J[:,0],J[:,1],J[:,2],J[:,3]
m_jet = np.sqrt(np.clip(jE**2-jpx**2-jpy**2-jpz**2,0,None)).astype(np.float32)
pt_jet,jy,jphi = kin(J); pt_jet = pt_jet.astype(np.float32)
target = (m_jet/pt_jet).astype(np.float32); SCALE = pt_jet   # m/pT -> massa via pT

print(f">> featurizando {Njet} jatos (PAD={PAD})...")
ANG = np.zeros((Njet,PAD,2),np.float32); Z = np.zeros((Njet,PAD),np.float32); MSK = np.zeros((Njet,PAD),np.float32)
for g in range(Njet):
    ANG[g],Z[g],MSK[g] = featurize(P[JCI[JCO[g]:JCO[g+1]]], jy[g], jphi[g])

# split FIXO (independente da seed do modelo)
nev = len(JEO)-1
split_rng = np.random.default_rng(12345)
ev_val = split_rng.random(nev) < 0.15
jet_ev = np.repeat(np.arange(nev), np.diff(JEO))
val = ev_val[jet_ev]; trn = ~val
mu, sd = target[trn].mean(), target[trn].std()

ANG_t = torch.tensor(ANG,device=DEV); Z_t = torch.tensor(Z,device=DEV); MSK_t = torch.tensor(MSK,device=DEV)
Y_t = torch.tensor((target-mu)/sd,device=DEV)
vi = np.where(val)[0]

def mlp(s):
    l=[]
    for a,b in zip(s[:-1],s[1:]): l+=[nn.Linear(a,b),nn.ReLU()]
    return nn.Sequential(*l[:-1])
class EFN(nn.Module):
    def __init__(s): super().__init__(); s.Phi=mlp([2,128,128,128,L]); s.F=mlp([L,128,128,1])
    def forward(s,ang,z,m): return s.F(((z.unsqueeze(-1))*s.Phi(ang*RINV)).sum(1)).squeeze(-1)
class PFN(nn.Module):
    def __init__(s): super().__init__(); s.Phi=mlp([3,128,128,128,L]); s.F=mlp([L,128,128,1])
    def forward(s,ang,z,m):
        f=torch.cat([z.unsqueeze(-1),ang*RINV],-1)
        return s.F((s.Phi(f)*m.unsqueeze(-1)).sum(1)).squeeze(-1)

def train(model, epochs=80, bs=256):
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    sch=torch.optim.lr_scheduler.StepLR(opt,30,0.5); lf=nn.MSELoss()
    idx=np.where(trn)[0]
    for ep in range(epochs):
        model.train(); np.random.shuffle(idx)
        for k in range(0,len(idx),bs):
            b=idx[k:k+bs]; opt.zero_grad()
            lf(model(ANG_t[b],Z_t[b],MSK_t[b]),Y_t[b]).backward(); opt.step()
        sch.step()
    return model
def pred(model, ii):
    model.eval()
    with torch.no_grad(): return (sd*model(ANG_t[ii],Z_t[ii],MSK_t[ii])+mu).cpu().numpy()

# IRC test (mesmas perturbacoes; converte alvo->GeV via pT do jato)
rng_irc = np.random.default_rng(1)
def split_col(cp): pt,_,_=kin(cp); i=int(np.argmax(pt)); h=cp[i]/2; return np.vstack([np.delete(cp,i,0),h,h])
def add_soft(cp, ay):
    y=ay+rng_irc.uniform(-0.4,0.4); ph=rng_irc.uniform(-np.pi,np.pi); pt=1e-3
    px,py,pz=pt*np.cos(ph),pt*np.sin(ph),pt*np.sinh(y)
    return np.vstack([cp,[px,py,pz,np.sqrt(px*px+py*py+pz*pz)]])
def predict_one(model, f):
    model.eval()
    with torch.no_grad():
        A=torch.tensor(f[0][None],device=DEV); Zt=torch.tensor(f[1][None],device=DEV); Mt=torch.tensor(f[2][None],device=DEV)
        return float((sd*model(A,Zt,Mt)+mu).cpu().numpy()[0])
def irc(model):
    gs = vi[:1200]; dc=[]; ds=[]
    for g in gs:
        cp=P[JCI[JCO[g]:JCO[g+1]]]; base=predict_one(model,featurize(cp,jy[g],jphi[g])); sg=SCALE[g]
        dc.append(abs(predict_one(model,featurize(split_col(cp),jy[g],jphi[g]))-base)*sg)
        ds.append(abs(predict_one(model,featurize(add_soft(cp,jy[g]),jy[g],jphi[g]))-base)*sg)
    return float(np.mean(dc)), float(np.mean(ds))

def corr(a,b): return float(np.corrcoef(a,b)[0,1])
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

R = {"EFN":{"corr":[],"rmse":[],"irc_col":[],"irc_soft":[]},
     "PFN":{"corr":[],"rmse":[],"irc_col":[],"irc_soft":[]}}
true_m = m_jet[vi]
for s in range(N_SEEDS):
    torch.manual_seed(s); np.random.seed(s)
    for name,Cls in [("EFN",EFN),("PFN",PFN)]:
        m = train(Cls().to(DEV))
        p = pred(m, vi)*SCALE[vi]
        col,soft = irc(m)
        R[name]["corr"].append(corr(p,true_m)); R[name]["rmse"].append(rmse(p,true_m))
        R[name]["irc_col"].append(col); R[name]["irc_soft"].append(soft)
    print(f"  seed {s}: EFN corr={R['EFN']['corr'][-1]:.4f} soft={R['EFN']['irc_soft'][-1]:.4f} | "
          f"PFN corr={R['PFN']['corr'][-1]:.4f} soft={R['PFN']['irc_soft'][-1]:.4f}")

print("\n=== RESUMO ({} seeds, boosted) ===".format(N_SEEDS))
out={}
for name in ("EFN","PFN"):
    out[name]={}
    for k in ("corr","rmse","irc_col","irc_soft"):
        a=np.array(R[name][k]); out[name][k]=[float(a.mean()),float(a.std())]
    print(f"{name}: corr={out[name]['corr'][0]:.4f}±{out[name]['corr'][1]:.4f}  "
          f"RMSE={out[name]['rmse'][0]:.3f}±{out[name]['rmse'][1]:.3f}  "
          f"IRC_col={out[name]['irc_col'][0]:.5f}±{out[name]['irc_col'][1]:.5f}  "
          f"IRC_soft={out[name]['irc_soft'][0]:.4f}±{out[name]['irc_soft'][1]:.4f}")
json.dump({"n_seeds":N_SEEDS,"device":DEV,"results":out,"raw":R}, open("results/seeds_results.json","w"), indent=2)
print(">> salvo: seeds_results.json")
