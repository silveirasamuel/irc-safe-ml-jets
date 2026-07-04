#!/usr/bin/env python3
"""
Gerador de dados para o estudo "IA vs anti-kt" (IRC-safe jet clustering).

Gera colisoes p-p com Pythia, clusteriza com FastJet (anti-kt R=0.4) e
salva TUDO que as duas fases do projeto precisam, num unico .npz:

  Fase B (EFN / observaveis IRC-safe)  -> constituintes + jatos de referencia
  Fase A (clusterizador aprendido)     -> mapa jato -> constituintes

Depende SO de pythia8 + fastjet + numpy (nada de pip extra).

Formato de saida (.npz, "ragged" via offsets estilo CSR)
--------------------------------------------------------
  particles         float32 (Npart, 4)   px, py, pz, E  de todos os constituintes
  particle_pid      int32   (Npart,)      PDG id de cada constituinte
  event_offsets     int64   (Nev+1,)      particulas do evento e: [off[e], off[e+1])
  jets              float32 (Njet, 4)     px, py, pz, E dos jatos anti-kt (referencia)
  jet_event_offsets int64   (Nev+1,)      jatos do evento e: [off[e], off[e+1])
  jet_const_offsets int64   (Njet+1,)     constituintes do jato g: [off[g], off[g+1])
  jet_const_index   int64   (Mconst,)     indice GLOBAL na array `particles`
  meta              str                   json com toda a configuracao + estatisticas

Uso (da raiz do repo):
    python src/gerar_dados.py [N_EVENTOS] [data/SAIDA.npz]
    (padrao: 20000 eventos -> data/dados_jatos.npz)
"""

import os
import sys
import json
import time
import numpy as np
import pythia8
import fastjet as fj

# ----------------------------------------------------------------------
# Parametros (defaults = regiao fiducial do EXP_JATOS/inclusive_jets.py).
# Sobrescreva por env vars: PTHAT_MIN, JET_R, JET_PT_MIN, JET_Y_MAX, ECM, SEED.
# Ex.: PTHAT_MIN=450 JET_R=0.8 JET_PT_MIN=450 python src/gerar_dados.py 50000 data/dados_boosted.npz
# ----------------------------------------------------------------------
def _env(name, default):
    return type(default)(os.environ.get(name, default))

N_EVENTS   = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
OUT_PATH   = sys.argv[2] if len(sys.argv) > 2 else "data/dados_jatos.npz"

ECM              = _env("ECM", 13600.0)       # energia de CM [GeV] — LHC Run 3
PTHAT_MIN        = _env("PTHAT_MIN", 20.0)    # pT minimo do espalhamento duro [GeV]
JET_R            = _env("JET_R", 0.4)         # raio do jato (anti-kt)
JET_PT_MIN       = _env("JET_PT_MIN", 20.0)   # pT minimo do jato [GeV]
JET_Y_MAX        = _env("JET_Y_MAX", 2.5)     # |y| maximo do jato (regiao fiducial)
PARTICLE_ETA_MAX = _env("PARTICLE_ETA_MAX", 5.0)  # aceitacao antes de clusterizar
SEED             = int(_env("SEED", 12345))   # semente do Pythia (reprodutibilidade)

# ----------------------------------------------------------------------
# Pythia
# ----------------------------------------------------------------------
print(f">> Configurando Pythia (eCM={ECM/1000:.1f} TeV, pTHatMin={PTHAT_MIN:.0f} GeV, seed={SEED})...")
pythia = pythia8.Pythia()
pythia.readString("Beams:idA = 2212")
pythia.readString("Beams:idB = 2212")
pythia.readString(f"Beams:eCM = {ECM}")
pythia.readString("HardQCD:all = on")
pythia.readString(f"PhaseSpace:pTHatMin = {PTHAT_MIN}")
pythia.readString("Random:setSeed = on")
pythia.readString(f"Random:seed = {SEED}")
pythia.readString("Next:numberShowEvent = 0")
pythia.readString("Next:numberShowInfo = 0")
pythia.readString("Next:numberShowProcess = 0")
pythia.readString("Print:quiet = on")
if not pythia.init():
    raise RuntimeError("Falha ao inicializar o Pythia.")

jet_def = fj.JetDefinition(fj.antikt_algorithm, JET_R)

# ----------------------------------------------------------------------
# Acumuladores (listas Python; viram arrays numpy no fim)
# ----------------------------------------------------------------------
part_rows     = []   # [px,py,pz,E] por constituinte (todos os eventos, concatenados)
part_pid      = []   # PDG id por constituinte
event_offsets = [0]  # CSR: fronteiras de particulas por evento

jet_rows          = []      # [px,py,pz,E] por jato de referencia
jet_event_offsets = [0]     # CSR: fronteiras de jatos por evento
jet_const_index   = []      # indices GLOBAIS (na array particles) dos constituintes
jet_const_offsets = [0]     # CSR: fronteiras de constituintes por jato

# ----------------------------------------------------------------------
# Loop sobre eventos
# ----------------------------------------------------------------------
print(f">> Gerando {N_EVENTS} eventos...\n")
t0 = time.time()
n_acc = 0
global_part_idx = 0   # contador do indice global de particula (== len(part_rows))

for iev in range(N_EVENTS):
    if not pythia.next():
        continue

    ev = pythia.event

    # ---- constituintes do evento: finais, visiveis, dentro da aceitacao ----
    fj_particles = []
    ev_start = global_part_idx
    for i in range(ev.size()):
        p = ev[i]
        if not (p.isFinal() and p.isVisible()):   # isVisible() ja exclui neutrinos
            continue
        if abs(p.eta()) > PARTICLE_ETA_MAX:
            continue
        part_rows.append((p.px(), p.py(), p.pz(), p.e()))
        part_pid.append(p.id())
        pj = fj.PseudoJet(p.px(), p.py(), p.pz(), p.e())
        pj.set_user_index(global_part_idx)        # ancora o jato ao indice global
        fj_particles.append(pj)
        global_part_idx += 1

    if len(fj_particles) < 2:
        # evento sem constituintes uteis: desfaz o que adicionamos e pula
        del part_rows[ev_start:]
        del part_pid[ev_start:]
        global_part_idx = ev_start
        continue

    n_acc += 1
    event_offsets.append(global_part_idx)

    # ---- clusterizacao anti-kt de referencia ----
    cs = fj.ClusterSequence(fj_particles, jet_def)
    jets = fj.sorted_by_pt(cs.inclusive_jets(JET_PT_MIN))
    jets = [j for j in jets if abs(j.rap()) < JET_Y_MAX]   # regiao fiducial

    for j in jets:
        jet_rows.append((j.px(), j.py(), j.pz(), j.e()))
        for c in j.constituents():
            jet_const_index.append(c.user_index())          # indice global na array particles
        jet_const_offsets.append(len(jet_const_index))
    jet_event_offsets.append(len(jet_rows))

    if (iev + 1) % 5000 == 0:
        dt = time.time() - t0
        rate = (iev + 1) / dt
        eta = (N_EVENTS - iev - 1) / rate
        print(f"  {iev+1:6d}/{N_EVENTS}  ({rate:5.0f} ev/s, ETA {eta:4.0f}s) "
              f"| jatos: {len(jet_rows):7d}  particulas: {len(part_rows):9d}")

pythia.stat()
dt = time.time() - t0

# ----------------------------------------------------------------------
# Converte para arrays e salva
# ----------------------------------------------------------------------
particles         = np.asarray(part_rows, dtype=np.float32)
particle_pid      = np.asarray(part_pid, dtype=np.int32)
event_offsets     = np.asarray(event_offsets, dtype=np.int64)
jets              = np.asarray(jet_rows, dtype=np.float32).reshape(-1, 4)
jet_event_offsets = np.asarray(jet_event_offsets, dtype=np.int64)
jet_const_offsets = np.asarray(jet_const_offsets, dtype=np.int64)
jet_const_index   = np.asarray(jet_const_index, dtype=np.int64)

info = pythia.infoPython()
sigma_pb = info.sigmaGen() * 1e9

meta = {
    "eCM_GeV": ECM, "pTHatMin_GeV": PTHAT_MIN, "process": "HardQCD:all",
    "jet_algo": "antikt", "jet_R": JET_R, "jet_pt_min_GeV": JET_PT_MIN,
    "jet_y_max": JET_Y_MAX, "particle_eta_max": PARTICLE_ETA_MAX, "seed": SEED,
    "n_events_requested": N_EVENTS, "n_events_accepted": n_acc,
    "n_particles": int(particles.shape[0]), "n_jets": int(jets.shape[0]),
    "sigma_pb": sigma_pb, "sigma_err_pb": info.sigmaErr() * 1e9,
    "columns_particles": ["px", "py", "pz", "E"],
    "columns_jets": ["px", "py", "pz", "E"],
    "gen_seconds": dt,
}

np.savez_compressed(
    OUT_PATH,
    particles=particles, particle_pid=particle_pid, event_offsets=event_offsets,
    jets=jets, jet_event_offsets=jet_event_offsets,
    jet_const_offsets=jet_const_offsets, jet_const_index=jet_const_index,
    meta=np.array(json.dumps(meta)),
)

# ----------------------------------------------------------------------
# Resumo
# ----------------------------------------------------------------------
mean_np = particles.shape[0] / max(n_acc, 1)
mean_nj = jets.shape[0] / max(n_acc, 1)
print("\n" + "=" * 68)
print(f"Eventos aceitos     : {n_acc}/{N_EVENTS}   em {dt:.1f}s  ({n_acc/max(dt,1e-9):.0f} ev/s)")
print(f"Particulas totais   : {particles.shape[0]:,}   (~{mean_np:.0f}/evento)")
print(f"Jatos totais        : {jets.shape[0]:,}   (~{mean_nj:.1f}/evento, pT>{JET_PT_MIN:.0f}, |y|<{JET_Y_MAX})")
print(f"Seccao de choque QCD: {sigma_pb:.3e} pb  (+/- {info.sigmaErr()*1e9:.1e})")
print(f"Salvo em            : {OUT_PATH}")
print("=" * 68)
