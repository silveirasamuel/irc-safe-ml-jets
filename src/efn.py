#!/usr/bin/env python3
"""
Fase B — Energy Flow Network (EFN): IA IRC-safe por construcao.

Aprende a MASSA de jato do anti-kt a partir dos constituintes, com a arquitetura

        O(jato) = F( sum_i  z_i * Phi(dy_i, dphi_i) )         [EFN]

onde z_i = pT_i / sum pT  (peso de energia) e Phi ve SO angulo relativo ao eixo
do jato. Isso e' IRC-safe por construcao:
  * particula soft  -> z_i -> 0  -> nao contribui;
  * split colinear  -> z_i -> z_i/2 + z_i/2 no MESMO angulo -> soma inalterada.

Contraste: uma PFN (Particle Flow Network) que ve z e soma SEM peso de energia

        O(jato) = F( sum_i  Phi(z_i, dy_i, dphi_i) )          [PFN, NAO IRC-safe]

Ambas ajustam a massa; so a EFN passa no teste de IRC. Mesmo teste do
licao_irc_dbscan.py, agora aplicado a rede.

Roda no venv:  ./venv/bin/python efn.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
PAD = 60                      # max constituintes por jato (p100=59)
L = 128                       # dimensao do espaco latente Phi
RINV = 2.5                    # 1/R: normaliza angulos p/ O(1) na entrada
DEV = "cuda" if torch.cuda.is_available() else "cpu"   # usa a GPU se disponivel

# Config por env var:
#   DATA=dados_boosted.npz TARGET=moverpt ./venv/bin/python efn.py
DATA   = os.environ.get("DATA", "data/dados_jatos.npz")
TARGET = os.environ.get("TARGET", "mass")      # "mass" (GeV) ou "moverpt" (m/pT, so-forma)
_stem  = os.path.splitext(os.path.basename(DATA))[0].replace("dados_jatos", "").replace("dados_", "")
TAG    = f"_{_stem}" if _stem else ""          # "" p/ default; "_boosted" p/ dados_boosted
FIG    = f"figures/efn_resultados{TAG}.png"
WPT    = f"results/efn_pesos{TAG}.pt"

# ----------------------------------------------------------------------
# 1) Carrega dados e monta features por jato
# ----------------------------------------------------------------------
print(f">> carregando {DATA} (alvo={TARGET}) ...")
d = np.load(DATA, allow_pickle=True)
P   = d["particles"].astype(np.float64)          # (Npart,4) px,py,pz,E
J   = d["jets"].astype(np.float64)               # (Njet,4)
JEO = d["jet_event_offsets"]
JCO = d["jet_const_offsets"]; JCI = d["jet_const_index"]
Njet = len(J)
PAD = int(np.diff(JCO).max()) + 2   # cobre TODOS os constituintes (+split) -> sem truncar -> colinear exato
print(f"   PAD adaptativo = {PAD}")

def kin(p4):  # -> pt, y, phi  (vetorizado em (N,4))
    px, py, pz, E = p4[...,0], p4[...,1], p4[...,2], p4[...,3]
    pt = np.hypot(px, py)
    y = 0.5*np.log(np.clip((E+pz)/np.clip(E-pz,1e-12,None),1e-12,None))
    return pt, y, np.arctan2(py, px)

def dphi(a, b):
    return np.arctan2(np.sin(a-b), np.cos(a-b))

def featurize(const_p4, axis_y, axis_phi):
    """constituintes (n,4) + eixo do jato -> (ang(PAD,2), z(PAD), mask(PAD))."""
    pt, y, phi = kin(const_p4)
    order = np.argsort(-pt)[:PAD]                 # mantem os PAD mais duros
    pt, y, phi = pt[order], y[order], phi[order]
    z = pt / pt.sum()
    n = len(pt)
    ang = np.zeros((PAD, 2), np.float32)
    zz  = np.zeros(PAD, np.float32)
    msk = np.zeros(PAD, np.float32)
    ang[:n, 0] = (y - axis_y)
    ang[:n, 1] = dphi(phi, axis_phi)
    zz[:n] = z; msk[:n] = 1.0
    return ang, zz, msk

# cinematica do jato
jpx, jpy, jpz, jE = J[:,0], J[:,1], J[:,2], J[:,3]
m_jet = np.sqrt(np.clip(jE**2 - jpx**2 - jpy**2 - jpz**2, 0, None)).astype(np.float32)
pt_jet, jy, jphi = kin(J); pt_jet = pt_jet.astype(np.float32)

# alvo do treino + fator p/ reconverter em MASSA (GeV): mass_gev = alvo * SCALE
if TARGET == "moverpt":
    target = (m_jet / pt_jet).astype(np.float32)   # razao adimensional (so forma)
    SCALE = pt_jet                                  # pT do jato (IRC-safe) reintroduz a escala
else:
    target = m_jet
    SCALE = np.ones(Njet, np.float32)

print(">> montando features de", Njet, "jatos ...")
ANG = np.zeros((Njet, PAD, 2), np.float32)
Z   = np.zeros((Njet, PAD), np.float32)
MSK = np.zeros((Njet, PAD), np.float32)
for g in range(Njet):
    cp = P[JCI[JCO[g]:JCO[g+1]]]
    ANG[g], Z[g], MSK[g] = featurize(cp, jy[g], jphi[g])

# split treino/val POR EVENTO (evita vazamento entre jatos do mesmo evento)
nev = len(JEO)-1
ev_is_val = np.random.rand(nev) < 0.15
jet_ev = np.repeat(np.arange(nev), np.diff(JEO))
val = ev_is_val[jet_ev]; trn = ~val
print(f"   treino={trn.sum()}  val={val.sum()}")

mu, sd = target[trn].mean(), target[trn].std()   # padroniza o alvo
def t(a): return torch.tensor(a, device=DEV)
ANG_t, Z_t, MSK_t = t(ANG), t(Z), t(MSK)
Y_t = t((target - mu)/sd)

# ----------------------------------------------------------------------
# 2) Modelos
# ----------------------------------------------------------------------
def mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])           # sem ReLU na saida

class EFN(nn.Module):
    """IRC-safe: Phi ve so angulo; agregacao ponderada por z."""
    def __init__(self):
        super().__init__()
        self.Phi = mlp([2, 128, 128, 128, L])
        self.F   = mlp([L, 128, 128, 1])
    def forward(self, ang, z, msk):
        phi = self.Phi(ang * RINV)                # (B,PAD,L)
        agg = (z.unsqueeze(-1) * phi).sum(1)      # z=0 no padding zera contrib.
        return self.F(agg).squeeze(-1)

class PFN(nn.Module):
    """NAO IRC-safe: Phi ve z; agregacao por soma simples (mascarada)."""
    def __init__(self):
        super().__init__()
        self.Phi = mlp([3, 128, 128, 128, L])
        self.F   = mlp([L, 128, 128, 1])
    def forward(self, ang, z, msk):
        feat = torch.cat([z.unsqueeze(-1), ang * RINV], dim=-1)   # (B,PAD,3)
        phi = self.Phi(feat) * msk.unsqueeze(-1)                  # zera padding
        agg = phi.sum(1)
        return self.F(agg).squeeze(-1)

# ----------------------------------------------------------------------
# 3) Treino
# ----------------------------------------------------------------------
def train(model, epochs=80, bs=256):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
    lossf = nn.MSELoss()
    idx = np.where(trn)[0]
    for ep in range(epochs):
        model.train(); np.random.shuffle(idx)
        for k in range(0, len(idx), bs):
            b = idx[k:k+bs]
            opt.zero_grad()
            out = model(ANG_t[b], Z_t[b], MSK_t[b])
            loss = lossf(out, Y_t[b]); loss.backward(); opt.step()
        sched.step()
        if (ep+1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                vi = np.where(val)[0]
                pv = model(ANG_t[vi], Z_t[vi], MSK_t[vi])
                rmse = (sd * (pv - Y_t[vi])).pow(2).mean().sqrt().item()
            print(f"   {model.__class__.__name__} ep{ep+1:2d}  val RMSE(alvo) = {rmse:.4f}")
    return model

if os.environ.get("REPLOT") and os.path.exists(WPT):
    # so re-plota a figura reusando pesos ja treinados (sem as 80 epocas)
    print(f">> REPLOT: carregando pesos de {WPT} (sem treinar)...")
    W = torch.load(WPT, map_location=DEV)
    efn = EFN().to(DEV); efn.load_state_dict(W["efn"])
    pfn = PFN().to(DEV); pfn.load_state_dict(W["pfn"])
    mu, sd = W["mu"], W["sd"]
else:
    print(">> treinando EFN ..."); efn = train(EFN().to(DEV))
    print(">> treinando PFN ..."); pfn = train(PFN().to(DEV))

def predict(model, ang, z, msk):
    model.eval()
    with torch.no_grad():
        return (sd * model(t(ang), t(z), t(msk)) + mu).cpu().numpy()

vi = np.where(val)[0]
pred_efn = predict(efn, ANG[vi], Z[vi], MSK[vi]) * SCALE[vi]   # -> massa em GeV
pred_pfn = predict(pfn, ANG[vi], Z[vi], MSK[vi]) * SCALE[vi]
true_m = m_jet[vi]
def rmse(a, b): return float(np.sqrt(np.mean((a-b)**2)))
def corr(a, b): return float(np.corrcoef(a, b)[0,1])

# ----------------------------------------------------------------------
# 4) Teste de IRC nas redes (mesmas perturbacoes do licao_irc_dbscan.py)
# ----------------------------------------------------------------------
rng = np.random.default_rng(1)
def split_collinear(cp):
    pt,_,_ = kin(cp); i = int(np.argmax(pt)); h = cp[i]/2.0
    return np.vstack([np.delete(cp, i, 0), h, h])
def add_soft(cp, axis_y):
    y = axis_y + rng.uniform(-0.4, 0.4); phi = rng.uniform(-np.pi, np.pi)
    pt = 1e-3
    px, py, pz = pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(y)
    return np.vstack([cp, [px, py, pz, np.sqrt(px*px+py*py+pz*pz)]])

print(">> teste de IRC nas redes ...")
test_g = vi[:3000]                              # subconjunto de val
dEFN = {"colinear": [], "soft": []}; dPFN = {"colinear": [], "soft": []}
for g in test_g:
    cp = P[JCI[JCO[g]:JCO[g+1]]]
    base_f = featurize(cp, jy[g], jphi[g])
    b_efn = predict(efn, base_f[0][None], base_f[1][None], base_f[2][None])[0]
    b_pfn = predict(pfn, base_f[0][None], base_f[1][None], base_f[2][None])[0]
    sg = SCALE[g]                                # converte Δalvo -> Δmassa em GeV
    for pert, fn in [("colinear", split_collinear(cp)), ("soft", add_soft(cp, jy[g]))]:
        f = featurize(fn, jy[g], jphi[g])        # MESMO eixo de referencia
        dEFN[pert].append(abs(predict(efn, f[0][None], f[1][None], f[2][None])[0] - b_efn) * sg)
        dPFN[pert].append(abs(predict(pfn, f[0][None], f[1][None], f[2][None])[0] - b_pfn) * sg)

# ----------------------------------------------------------------------
# 5) Resumo + figura
# ----------------------------------------------------------------------
print("\n" + "="*66)
print(f"{'':14}{'val RMSE [GeV]':>16}{'corr':>10}")
print(f"{'EFN (safe)':14}{rmse(pred_efn,true_m):>16.3f}{corr(pred_efn,true_m):>10.3f}")
print(f"{'PFN (unsafe)':14}{rmse(pred_pfn,true_m):>16.3f}{corr(pred_pfn,true_m):>10.3f}")
print("-"*66)
print(f"{'IRC |Δmassa| [GeV]':22}{'colinear':>14}{'soft':>14}")
for name, dd in [("EFN (safe)", dEFN), ("PFN (unsafe)", dPFN)]:
    print(f"{name:22}{np.mean(dd['colinear']):>14.5f}{np.mean(dd['soft']):>14.5f}")
print("="*66)

torch.save({"efn": efn.state_dict(), "pfn": pfn.state_dict(), "target": TARGET,
            "mu": float(mu), "sd": float(sd)}, WPT)

plt.rcParams.update({"font.size": 11, "figure.dpi": 120})
fig, ax = plt.subplots(2, 2, figsize=(11, 9))
col = {"EFN": "#1b9e77", "PFN": "#d95f02"}
mmax = float(np.percentile(true_m, 99.5))       # range da massa nesta amostra
# (linha 0) regressao massa: predito vs verdadeiro
for a, name, pred in [(ax[0,0], "EFN", pred_efn), (ax[0,1], "PFN", pred_pfn)]:
    a.hexbin(true_m, pred, gridsize=45, extent=(0,mmax,0,mmax), cmap="viridis", bins="log", mincnt=1)
    a.plot([0,mmax],[0,mmax], "w--", lw=1.2)
    a.set_xlim(0,mmax); a.set_ylim(0,mmax)
    a.set_title(f"{name}: jet mass — RMSE={rmse(pred,true_m):.2f} GeV, r={corr(pred,true_m):.3f}",
                color=col[name])
    a.set_xlabel("true anti-kt mass [GeV]"); a.set_ylabel("predicted mass [GeV]")
# (linha 1) teste de IRC
bins = np.logspace(-6, 3, 46)
for a, pert in [(ax[1,0], "colinear"), (ax[1,1], "soft")]:
    a.hist(np.clip(dEFN[pert],1e-6,1e3), bins=bins, color=col["EFN"], alpha=0.5,
           label=f"EFN  ⟨|Δ|⟩={np.mean(dEFN[pert]):.1e}")
    a.hist(np.clip(dPFN[pert],1e-6,1e3), bins=bins, color=col["PFN"], alpha=0.5,
           label=f"PFN  ⟨|Δ|⟩={np.mean(dPFN[pert]):.1e}")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_title(f"IRC test — {'collinear' if pert=='colinear' else 'soft'} perturbation")
    a.set_xlabel("|Δ predicted mass| [GeV]"); a.set_ylabel("jets")
    a.legend(frameon=False, fontsize=9)
fig.suptitle("EFN is IRC-safe by construction (Δ≈0); the PFN fits using IRC-unsafe information and therefore fails the test",
             fontsize=12)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(FIG, bbox_inches="tight")
print(f">> figura salva em: {FIG}   | pesos em: {WPT}")
