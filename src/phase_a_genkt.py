#!/usr/bin/env python3
"""
Fase A — Clusterizador de jatos APRENDIDO e IRC-safe por construcao.

Ideia: em vez de uma rede neural nao-diferenciavel que particiona particulas,
aprende-se os parametros da familia generalized-kt

    d_ij = min(p_Ti^{2p}, p_Tj^{2p}) * (dR_ij / R)^2 ,   d_iB = p_Ti^{2p}

que e' recombinacao sequencial => IRC-safe para QUALQUER (p, R).  O expoente de
energia p e o raio R sao "aprendidos" otimizando um objetivo FISICO real:
a resolucao de energia jato-vs-parton  sigma(pT_jato / pT_parton).

Depois:
  (1) prova-se que a familia toda (incl. o otimo aprendido) e' IRC-safe pela
      mesma bateria de perturbacao do licao_irc_dbscan.py, contrastando com DBSCAN;
  (2) compara-se o algoritmo aprendido com o anti-kt padrao.

Roda da raiz do repo:  python src/phase_a_genkt.py
"""
import json
import numpy as np
import pythia8
import fastjet as fj
from sklearn.cluster import DBSCAN
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ECM, PTHAT_MIN = 13600.0, 450.0
NEV      = 2000          # eventos para a otimizacao/comparacao
NIRC     = 900           # subconjunto para a bateria de IRC
JET_PTMIN, JET_YMAX = 200.0, 2.5
MATCH_DR = 0.35
P_GRID = np.array([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])
R_GRID = np.array([0.4,0.6,0.8,1.0,1.2])
rng = np.random.default_rng(3)

def dphi(a,b): return np.arctan2(np.sin(a-b), np.cos(a-b))
def pt_y_phi(p4):
    px,py,pz,E=p4; pt=np.hypot(px,py)
    y=0.5*np.log((E+pz)/(E-pz)) if E>abs(pz) else 0.0
    return pt,y,np.arctan2(py,px)

# ----------------------------------------------------------------------
# 1) Gera eventos: guarda particulas finais (PseudoJet) + 2 partons duros
# ----------------------------------------------------------------------
print(f">> gerando {NEV} eventos (pTHatMin={PTHAT_MIN:.0f})...")
pythia = pythia8.Pythia()
for s in ["Beams:idA = 2212","Beams:idB = 2212",f"Beams:eCM = {ECM}","HardQCD:all = on",
          f"PhaseSpace:pTHatMin = {PTHAT_MIN}","Random:setSeed = on","Random:seed = 91",
          "Print:quiet = on","Next:numberShowEvent = 0","Next:numberShowInfo = 0",
          "Next:numberShowProcess = 0"]:
    pythia.readString(s)
if not pythia.init(): raise RuntimeError("Pythia falhou.")

events = []   # cada item: (P4 ndarray (n,4), partons ndarray (k,3)=(pt,y,phi))
n=0; attempts=0
while n < NEV and attempts < NEV*30:
    attempts += 1
    if not pythia.next(): continue
    ev = pythia.event
    rows=[]; partons=[]
    for i in range(ev.size()):
        p = ev[i]
        if abs(p.status())==23:                              # outgoing do processo duro (2->2); -23 apos chuveirar
            pt,y,ph = pt_y_phi((p.px(),p.py(),p.pz(),p.e())); partons.append((pt,y,ph))
        if p.isFinal() and p.isVisible() and abs(p.eta())<5.0:
            rows.append((p.px(),p.py(),p.pz(),p.e()))
    if len(rows)<2 or len(partons)<2: continue
    events.append((np.array(rows,float), np.array(partons,float)))
    n+=1
print(f"   {len(events)} eventos guardados (~{np.mean([len(e[0]) for e in events]):.0f} part/ev)")

# pre-constroi PseudoJets (reuso entre grades) com user_index p/ overlap
PJ = []
for P4,_ in events:
    lst=[]
    for k,(px,py,pz,E) in enumerate(P4):
        pj=fj.PseudoJet(px,py,pz,E); pj.set_user_index(k); lst.append(pj)
    PJ.append(lst)

# ----------------------------------------------------------------------
# 2) Objetivo fisico: resolucao de resposta sigma(pT_jato/pT_parton)
# ----------------------------------------------------------------------
def cluster(pjs, p, R):
    cs = fj.ClusterSequence(pjs, fj.JetDefinition(fj.genkt_algorithm, R, float(p)))
    return [j for j in fj.sorted_by_pt(cs.inclusive_jets(JET_PTMIN)) if abs(j.rap())<JET_YMAX]

def response_metrics(p, R):
    resp=[]; nmatch=0; npart=0
    for (P4, partons), pjs in zip(events, PJ):
        jets = cluster(pjs, p, R)
        npart += len(partons)
        jy = np.array([j.rap() for j in jets]); jph=np.array([j.phi_std() for j in jets])
        jpt= np.array([j.pt() for j in jets])
        for ppt,py,pph in partons:
            if len(jets)==0: continue
            dR=np.sqrt((jy-py)**2 + dphi(jph,pph)**2)
            k=int(np.argmin(dR))
            if dR[k]<MATCH_DR:
                nmatch+=1; resp.append(jpt[k]/ppt)
    resp=np.array(resp)
    if len(resp)<50: return dict(sigma=np.nan, med=np.nan, eff=nmatch/max(npart,1))
    q16,q50,q84=np.percentile(resp,[15.87,50,84.13])
    return dict(sigma=(q84-q16)/2, med=q50, eff=nmatch/max(npart,1))

print(">> varrendo grade (p, R) — resolucao de resposta...")
SIG=np.full((len(P_GRID),len(R_GRID)),np.nan); MED=np.zeros_like(SIG); EFF=np.zeros_like(SIG)
for ip,p in enumerate(P_GRID):
    for iR,R in enumerate(R_GRID):
        m=response_metrics(p,R); SIG[ip,iR]=m["sigma"]; MED[ip,iR]=m["med"]; EFF[ip,iR]=m["eff"]
    print(f"   p={p:+.1f}: sigma={np.array2string(SIG[ip],precision=3,floatmode='fixed')}"
          f"  bias={np.array2string(MED[ip]-1,precision=3,floatmode='fixed')}  eff~{EFF[ip].mean():.2f}")

# objetivo fisico: RMS da resposta = sqrt(bias^2 + sigma^2)  (escala de energia + resolucao)
# -> otimo INTERIOR: R pequeno perde FSR (bias<0); R grande pega UE/MPI (bias>0)
OBJ = np.sqrt((MED-1.0)**2 + SIG**2)
mask = EFF>=0.80
obj_masked = np.where(mask, OBJ, np.inf)
ip,iR = np.unravel_index(np.argmin(obj_masked), OBJ.shape)
p_star, R_star = float(P_GRID[ip]), float(R_GRID[iR])
print(f"\n>> OTIMO aprendido: p*={p_star:+.1f}  R*={R_star:.1f}  "
      f"(RMS={OBJ[ip,iR]:.3f}, sigma={SIG[ip,iR]:.3f}, bias={MED[ip,iR]-1:+.3f}, eff={EFF[ip,iR]:.2f})")
# referencia anti-kt R=0.8
iap = int(np.where(P_GRID==-1.0)[0][0]); iar=int(np.where(R_GRID==0.8)[0][0])
print(f">> anti-kt R=0.8       : RMS={OBJ[iap,iar]:.3f}, sigma={SIG[iap,iar]:.3f}, bias={MED[iap,iar]-1:+.3f}, eff={EFF[iap,iar]:.2f}")

# ----------------------------------------------------------------------
# 3) Bateria de IRC: genkt (varios p) e DBSCAN, mesmas perturbacoes
# ----------------------------------------------------------------------
def genkt_jets_arr(P4, p, R):
    pjs=[fj.PseudoJet(*r) for r in P4]
    return [(j.pt(),j.rap(),j.phi_std()) for j in cluster(pjs,p,R)]
def dbscan_jets_arr(P4, eps=0.8, mpts=4):
    yphi=np.array([pt_y_phi(r)[1:] for r in P4])
    dy=yphi[:,None,0]-yphi[None,:,0]; dp=dphi(yphi[:,None,1],yphi[None,:,1])
    lab=DBSCAN(eps=eps,min_samples=mpts,metric="precomputed").fit_predict(np.sqrt(dy**2+dp**2))
    out=[]
    for kk in set(lab):
        if kk==-1: continue
        p4=P4[lab==kk].sum(0); pt,y,ph=pt_y_phi(p4)
        if pt>JET_PTMIN and abs(y)<JET_YMAX: out.append((pt,y,ph))
    return out
def jetset_distance(A,B):
    A=sorted(A,key=lambda j:-j[0]); used=[False]*len(B); D=0.0
    for pta,ya,pha in A:
        best,bj=1e9,-1
        for j,(ptb,yb,phb) in enumerate(B):
            if used[j]: continue
            dd=np.hypot(ya-yb,dphi(pha,phb))
            if dd<best: best,bj=dd,j
        if bj>=0 and best<0.8: used[bj]=True; D+=abs(pta-B[bj][0])
        else: D+=pta
    for j,u in enumerate(used):
        if not u: D+=B[j][0]
    return D
def split_collinear(P4):
    i=int(np.argmax(np.hypot(P4[:,0],P4[:,1]))); h=P4[i]/2
    return np.vstack([np.delete(P4,i,0),h,h])
def add_soft(P4):
    y=rng.uniform(-JET_YMAX,JET_YMAX); ph=rng.uniform(-np.pi,np.pi); pt=1e-3
    px,py,pz=pt*np.cos(ph),pt*np.sin(ph),pt*np.sinh(y)
    return np.vstack([P4,[px,py,pz,np.sqrt(px*px+py*py+pz*pz)]])

algos = {f"anti-kt (p=-1)": ("genkt",-1.0), "C/A (p=0)":("genkt",0.0),
         "kt (p=+1)":("genkt",1.0), f"aprendido (p={p_star:+.0f})":("genkt",p_star),
         "DBSCAN":("dbscan",None)}
print(">> bateria de IRC (genkt vs DBSCAN)...")
irc={name:{"colinear":[],"soft":[]} for name in algos}
for (P4,_) in events[:NIRC]:
    Pcol, Psoft = split_collinear(P4), add_soft(P4)
    for name,(kind,p) in algos.items():
        f = (lambda X: genkt_jets_arr(X,p,0.8)) if kind=="genkt" else (lambda X: dbscan_jets_arr(X))
        base=f(P4)
        irc[name]["colinear"].append(jetset_distance(base,f(Pcol)))
        irc[name]["soft"].append(jetset_distance(base,f(Psoft)))
irc_summary={name:{k:(float(np.mean(v)),float(100*np.mean(np.array(v)>1e-3)))
                   for k,v in dd.items()} for name,dd in irc.items()}
print(f"{'algoritmo':>18}{'colinear <D>':>14}{'%alt':>7}{'soft <D>':>12}{'%alt':>7}")
for name,dd in irc_summary.items():
    print(f"{name:>18}{dd['colinear'][0]:>14.4f}{dd['colinear'][1]:>6.0f}%{dd['soft'][0]:>12.4f}{dd['soft'][1]:>6.0f}%")

# ----------------------------------------------------------------------
# 4) Comparacao aprendido vs anti-kt: massa, N, overlap de constituintes
# ----------------------------------------------------------------------
def cluster_full(pjs, p, R):  # devolve jatos com constituintes
    cs = fj.ClusterSequence(pjs, fj.JetDefinition(fj.genkt_algorithm, R, float(p)))
    return [j for j in fj.sorted_by_pt(cs.inclusive_jets(JET_PTMIN)) if abs(j.rap())<JET_YMAX]
def summ(p,R):
    masses=[]; njets=[]
    for pjs in PJ:
        js=cluster_full(pjs,p,R); njets.append(len(js)); masses+=[j.m() for j in js]
    return np.mean(njets), np.mean(masses), np.array(masses)
nj_ak,mm_ak,ms_ak = summ(-1.0,0.8)
nj_st,mm_st,ms_st = summ(p_star,R_star)
# overlap de constituintes (pT compartilhado) entre jatos casados anti-kt <-> aprendido
# (clustering INLINE: o ClusterSequence precisa continuar vivo p/ ler .constituents())
ov=[]
jd_ak = fj.JetDefinition(fj.genkt_algorithm, 0.8, -1.0)
jd_st = fj.JetDefinition(fj.genkt_algorithm, R_star, float(p_star))
try:
    for pjs in PJ[:1200]:
        csA = fj.ClusterSequence(pjs, jd_ak)
        csB = fj.ClusterSequence(pjs, jd_st)
        A=[j for j in fj.sorted_by_pt(csA.inclusive_jets(JET_PTMIN)) if abs(j.rap())<JET_YMAX]
        B=[j for j in fj.sorted_by_pt(csB.inclusive_jets(JET_PTMIN)) if abs(j.rap())<JET_YMAX]
        for ja in A:
            ya,pha=ja.rap(),ja.phi_std()
            best,jb=1e9,None
            for j in B:
                dd=np.hypot(ya-j.rap(),dphi(pha,j.phi_std()))
                if dd<best: best,jb=dd,j
            if jb is None or best>MATCH_DR: continue
            ia={c.user_index():c.pt() for c in ja.constituents()}   # csA ainda vivo
            ib=set(c.user_index() for c in jb.constituents())
            shared=sum(pt for k,pt in ia.items() if k in ib)
            ov.append(shared/max(sum(ia.values()),1e-9))
        del csA, csB
    overlap=float(np.mean(ov)) if ov else float("nan")
except Exception as e:
    print("   (overlap falhou:", e, ")"); overlap=float("nan")
print(f"\n>> COMPARACAO  anti-kt(R0.8) vs aprendido(p{p_star:+.0f},R{R_star:.1f}):")
print(f"   N jatos/ev : {nj_ak:.2f}  vs {nj_st:.2f}")
print(f"   massa media: {mm_ak:.1f}  vs {mm_st:.1f} GeV")
print(f"   overlap de pT de constituintes (casados): {overlap:.3f}")

# ----------------------------------------------------------------------
# 5) Salva resultados + figuras
# ----------------------------------------------------------------------
res=dict(NEV=len(events), p_grid=P_GRID.tolist(), R_grid=R_GRID.tolist(),
         sigma=SIG.tolist(), med=MED.tolist(), eff=EFF.tolist(), obj=OBJ.tolist(),
         p_star=p_star, R_star=R_star,
         obj_star=float(OBJ[ip,iR]), sigma_star=float(SIG[ip,iR]), bias_star=float(MED[ip,iR]-1), eff_star=float(EFF[ip,iR]),
         obj_antikt=float(OBJ[iap,iar]), sigma_antikt=float(SIG[iap,iar]), bias_antikt=float(MED[iap,iar]-1), eff_antikt=float(EFF[iap,iar]),
         irc=irc_summary, njets_antikt=float(nj_ak), njets_star=float(nj_st),
         mass_antikt=float(mm_ak), mass_star=float(mm_st), overlap=overlap)
json.dump(res, open("results/phase_a_results.json","w"), indent=2)

plt.rcParams.update({"font.size":11,"figure.dpi":120})
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.8))
# (a) landscape do objetivo fisico OBJ = RMS da resposta, sobre (p, R)
im=a1.imshow(OBJ, origin="lower", aspect="auto", cmap="viridis_r",
             extent=[R_GRID[0]-.1,R_GRID[-1]+.1,P_GRID[0]-.25,P_GRID[-1]+.25])
a1.scatter([R_star],[p_star], marker="*", s=320, c="#e41a1c", edgecolor="w", label=f"learned (p*={p_star:+.1f}, R*={R_star:.1f})", zorder=5)
a1.scatter([0.8],[-1.0], marker="o", s=90, facecolors="none", edgecolor="w", lw=2, label="anti-kt (p=-1, R=0.8)", zorder=5)
a1.set_xlabel("radius R"); a1.set_ylabel("energy exponent p")
a1.set_title("Physics objective: response RMS $p_T^{jet}/p_T^{parton}$")
fig.colorbar(im, ax=a1, label="RMS = √(bias²+σ²)  (lower = better)")
a1.legend(frameon=True, fontsize=8, loc="upper right")
# (b) bateria IRC: <D> soft por algoritmo (log) — recombinacao vs DBSCAN
names=list(irc_summary.keys()); xs=np.arange(len(names))
soft_D=[max(irc_summary[n]["soft"][0],1e-6) for n in names]
cols=["#1b9e77" if n!="DBSCAN" else "#d95f02" for n in names]
bars=a2.bar(xs, soft_D, 0.6, color=cols)
a2.set_yscale("log"); a2.set_ylim(1e-5, 5)
a2.set_xticks(xs); a2.set_xticklabels([("learned" if n.startswith("aprendido") else n.split(" (")[0]) for n in names], rotation=25, ha="right", fontsize=9)
a2.set_ylabel("⟨D⟩ under soft perturbation  [GeV]")
a2.set_title("IRC safety: recombination (green) vs DBSCAN (orange)")
a2.grid(alpha=.3,axis="y",which="both")
for b,v in zip(bars,soft_D):
    a2.text(b.get_x()+b.get_width()/2, v*1.4, f"{v:.1e}", ha="center", fontsize=7)
fig.suptitle("Phase A — learned clusterer (generalized-kt): IRC-safe by construction, optimized for physics resolution",
             fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("figures/phase_a_genkt.png", bbox_inches="tight")
print(">> salvo: phase_a_genkt.png  +  phase_a_results.json")
