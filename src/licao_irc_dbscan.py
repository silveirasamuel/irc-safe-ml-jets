#!/usr/bin/env python3
"""
Figura-licao: por que o DBSCAN NAO e IRC-safe (e o anti-kt e).

Aplica as duas perturbacoes canonicas que definem IRC safety e mede quanto
os jatos mudam:

  * splitting COLINEAR : a particula mais dura -> duas colineares (p -> p/2 + p/2,
                          mesma direcao). Um algoritmo colinear-safe NAO muda.
  * adicao SOFT        : acrescenta 1 particula com pT -> 0 em direcao aleatoria.
                          Um algoritmo infrared-safe NAO muda.

Compara anti-kt (FastJet) vs DBSCAN fisico em (y, phi) com eps ~ R.
Metrica de mudanca: "distancia entre conjuntos de jatos" D (casamento guloso
por dR; jatos nao-casados contribuem com seu proprio pT). D=0 => invariante.

Roda no venv:  ./venv/bin/python licao_irc_dbscan.py [N_EVENTOS]
"""

import sys
import numpy as np
import pythia8
import fastjet as fj
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_EVENTS   = int(sys.argv[1]) if len(sys.argv) > 1 else 500
ECM, PTHAT_MIN = 13600.0, 20.0
JET_R, JET_PT_MIN, JET_Y_MAX = 0.4, 20.0, 2.5
DB_EPS, DB_MINPTS = 0.4, 4          # DBSCAN fisico em (y,phi): eps ~ R
SOFT_PT = 1e-3                       # pT da particula soft [GeV]
rng = np.random.default_rng(0)

# ----------------------------------------------------------------------
# Utilitarios de cinematica
# ----------------------------------------------------------------------
def pt_y_phi(p4):
    px, py, pz, E = p4
    pt = np.hypot(px, py)
    y = 0.5 * np.log((E + pz) / (E - pz)) if abs(E) > abs(pz) else 0.0
    return pt, y, np.arctan2(py, px)

def dphi(a, b):
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))

# ----------------------------------------------------------------------
# Dois clusterizadores -> lista de jatos como (pt, y, phi)
# ----------------------------------------------------------------------
def antikt_jets(P):
    pjs = [fj.PseudoJet(*row) for row in P]
    cs = fj.ClusterSequence(pjs, fj.JetDefinition(fj.antikt_algorithm, JET_R))
    out = []
    for j in fj.sorted_by_pt(cs.inclusive_jets(JET_PT_MIN)):
        if abs(j.rap()) < JET_Y_MAX:
            out.append((j.pt(), j.rap(), j.phi_std()))
    return out

def dbscan_jets(P):
    # coordenadas fisicas (y, phi) e matriz de distancia dR (phi periodico)
    yphi = np.array([pt_y_phi(row)[1:] for row in P])   # (N,2) = (y, phi)
    dy = yphi[:, None, 0] - yphi[None, :, 0]
    dp = dphi(yphi[:, None, 1], yphi[None, :, 1])
    dR = np.sqrt(dy**2 + dp**2)
    labels = DBSCAN(eps=DB_EPS, min_samples=DB_MINPTS, metric="precomputed").fit_predict(dR)
    out = []
    for k in set(labels):
        if k == -1:
            continue
        p4 = P[labels == k].sum(axis=0)     # soma de 4-momentos (fisica correta)
        pt, y, phi = pt_y_phi(p4)
        if pt > JET_PT_MIN and abs(y) < JET_Y_MAX:
            out.append((pt, y, phi))
    return out

# ----------------------------------------------------------------------
# Distancia entre conjuntos de jatos (casamento guloso por dR)
# ----------------------------------------------------------------------
def jetset_distance(A, B):
    A = sorted(A, key=lambda j: -j[0]); B = list(B)
    used = [False] * len(B)
    D = 0.0
    for pt_a, y_a, phi_a in A:
        best, bj = 1e9, -1
        for j, (pt_b, y_b, phi_b) in enumerate(B):
            if used[j]:
                continue
            d = np.hypot(y_a - y_b, dphi(phi_a, phi_b))
            if d < best:
                best, bj = d, j
        if bj >= 0 and best < JET_R:
            used[bj] = True
            D += abs(pt_a - B[bj][0])          # jato casado: diferenca de pT
        else:
            D += pt_a                           # jato de A sem par
    for j, u in enumerate(used):
        if not u:
            D += B[j][0]                        # jato de B sem par
    return D

# ----------------------------------------------------------------------
# Perturbacoes
# ----------------------------------------------------------------------
def split_collinear(P):
    i = int(np.argmax(np.hypot(P[:, 0], P[:, 1])))   # particula mais dura
    half = P[i] / 2.0
    return np.vstack([np.delete(P, i, axis=0), half, half])

def add_soft(P):
    y = rng.uniform(-JET_Y_MAX, JET_Y_MAX)
    phi = rng.uniform(-np.pi, np.pi)
    px, py = SOFT_PT * np.cos(phi), SOFT_PT * np.sin(phi)
    pz = SOFT_PT * np.sinh(y)
    E = np.sqrt(px*px + py*py + pz*pz)               # sem massa
    return np.vstack([P, [px, py, pz, E]])

# ----------------------------------------------------------------------
# Pythia
# ----------------------------------------------------------------------
print(f">> gerando {N_EVENTS} eventos p/ o teste IRC...")
pythia = pythia8.Pythia()
for s in ["Beams:idA = 2212", "Beams:idB = 2212", f"Beams:eCM = {ECM}",
          "HardQCD:all = on", f"PhaseSpace:pTHatMin = {PTHAT_MIN}",
          "Random:setSeed = on", "Random:seed = 7", "Print:quiet = on",
          "Next:numberShowEvent = 0", "Next:numberShowInfo = 0",
          "Next:numberShowProcess = 0"]:
    pythia.readString(s)
if not pythia.init():
    raise RuntimeError("Falha ao inicializar o Pythia.")

# D[algoritmo][perturbacao] = lista de distancias por evento
D = {"anti-kt": {"colinear": [], "soft": []},
     "DBSCAN":  {"colinear": [], "soft": []}}
clusterers = {"anti-kt": antikt_jets, "DBSCAN": dbscan_jets}

n = 0
while n < N_EVENTS:
    if not pythia.next():
        continue
    ev = pythia.event
    rows = []
    for i in range(ev.size()):
        p = ev[i]
        if p.isFinal() and p.isVisible() and abs(p.eta()) < 5.0:
            rows.append((p.px(), p.py(), p.pz(), p.e()))
    if len(rows) < 2:
        continue
    P = np.array(rows, dtype=np.float64)
    Pcol = split_collinear(P)
    Psoft = add_soft(P)
    for name, fun in clusterers.items():
        base = fun(P)
        D[name]["colinear"].append(jetset_distance(base, fun(Pcol)))
        D[name]["soft"].append(jetset_distance(base, fun(Psoft)))
    n += 1
    if n % 100 == 0:
        print(f"  {n}/{N_EVENTS}")

# ----------------------------------------------------------------------
# Resumo numerico
# ----------------------------------------------------------------------
TOL = 1e-3   # GeV; abaixo disso consideramos "invariante"
print("\n" + "=" * 64)
print(f"{'algoritmo':>10} {'perturbacao':>12} {'% eventos alterados':>22} {'<D> [GeV]':>12}")
print("-" * 64)
frac = {}
for name in D:
    for pert in ("colinear", "soft"):
        arr = np.array(D[name][pert])
        f = 100.0 * np.mean(arr > TOL)
        frac[(name, pert)] = f
        print(f"{name:>10} {pert:>12} {f:>21.1f}% {arr.mean():>12.3f}")
print("=" * 64)

# ----------------------------------------------------------------------
# Figura
# ----------------------------------------------------------------------
plt.rcParams.update({"font.size": 11, "figure.dpi": 120})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
col = {"anti-kt": "#1b9e77", "DBSCAN": "#d95f02"}
titulos = {"colinear": "Collinear splitting  (p → p/2 + p/2)",
           "soft": f"Soft addition  (p_T = {SOFT_PT:g} GeV)"}
bins = np.linspace(0, 40, 41)

for ax, pert in zip(axes, ("colinear", "soft")):
    for name in ("anti-kt", "DBSCAN"):
        arr = np.clip(np.array(D[name][pert]), 0, bins[-1] - 1e-9)
        ax.hist(arr, bins=bins, histtype="stepfilled", alpha=0.45,
                color=col[name], label=f"{name}  ({frac[(name,pert)]:.0f}% altered)")
        ax.hist(arr, bins=bins, histtype="step", color=col[name], lw=1.6)
    ax.set_yscale("log")
    ax.set_title(titulos[pert])
    ax.set_xlabel("D = jet-set distance  [GeV]")
    ax.set_ylabel("events")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

fig.suptitle("IRC-safety test:  anti-kt is invariant, DBSCAN is not  "
             f"(pp {ECM/1000:.1f} TeV, {N_EVENTS} events)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("figures/licao_irc_dbscan.png", bbox_inches="tight")
print(">> figura salva em: licao_irc_dbscan.png")
