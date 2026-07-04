# Infrared- and Collinear-Safe Machine Learning for Jet Clustering

Failure modes, a quantitative safety battery, and a learned anti-$k_t$.

**Authors:** Samuel Pedro Pereira Silveira (UFTM) · Mauro Rogério Cosentino (UFABC, ALICE/CERN)

📄 **Paper:** [`paper/paper.pdf`](paper/paper.pdf) · interactive web version: [`paper/paper.html`](paper/paper.html)

---

## What this is

Infrared and collinear (IRC) safety — invariance of an observable under soft emissions
and collinear splittings — is what makes a jet definition calculable in perturbative QCD.
It is an **exact symmetry of the physics, not something a neural network learns by training.**
Using ~1.2×10⁵ jets from PYTHIA 8, this project shows three ways an algorithm can relate to it:

1. **Density-based clustering (DBSCAN) is IRC-unsafe** — a single 10⁻³ GeV particle alters the
   reconstructed jets in 6.3 % of events, while anti-$k_t$ is invariant to machine precision.
2. **Energy Flow Networks (EFN) restore exact safety** — the energy-weighted Deep-Sets
   architecture reproduces the anti-$k_t$ jet mass (correlation 0.996 on boosted jets) while
   remaining exactly IRC-safe, with its response to a soft/collinear perturbation *vanishing*
   in the limit (power-law scaling laws), whereas an unconstrained Particle Flow Network (PFN)
   of equal accuracy saturates at a finite plateau.
3. **A learnable, provably safe jet algorithm** — treating the generalized-$k_t$ energy
   exponent and radius as free parameters and optimizing them against jet–parton energy
   response *rediscovers the anti-$k_t$ exponent* $p=-1$; every member of the family is
   IRC-safe by construction.

**Takeaway:** in physics-oriented ML, IRC safety must be imposed as a *design constraint*,
never chased as a training target.

## Repository layout

```
src/         analysis + paper-build scripts (run from the repo root)
figures/     generated figures (PNG) used in the paper
results/     numerical outputs (JSON) and trained network weights (.pt)
paper/       LaTeX source, compiled PDF, and self-contained HTML
data/        PYTHIA/FastJet datasets (.npz) — not tracked (regenerate, see below)
```

## Requirements

**Python packages** (`pip install -r requirements.txt`): NumPy, matplotlib, scikit-learn,
SciPy, PyTorch. A CUDA build of PyTorch is optional but recommended for training
(`--index-url https://download.pytorch.org/whl/cu124`); the scripts auto-detect the GPU.

**High-energy-physics toolchain** (compiled from source, *not* pip-installable):
[PYTHIA 8.3](https://pythia.org/) and [FastJet 3.5](http://fastjet.fr/) with their Python
bindings, importable as `pythia8` and `fastjet`.

## Reproducing the results

All scripts are run **from the repository root**. Generate the data first (this is the only
slow, CPU-bound step; ~5 min each on a modern laptop):

```bash
python src/gerar_dados.py 50000 data/dados_jatos.npz                                  # low-pT, anti-kt R=0.4
PTHAT_MIN=450 JET_R=0.8 JET_PT_MIN=450 SEED=777 python src/gerar_dados.py 50000 data/dados_boosted.npz   # boosted fat jets
```

Then run the analyses (each produces a figure and/or a JSON in `figures/`, `results/`):

```bash
python src/licao_irc_dbscan.py 2000                          # Fig. 1  — DBSCAN is IRC-unsafe
DATA=data/dados_jatos.npz   TARGET=mass    python src/efn.py  # low-pT EFN/PFN
DATA=data/dados_boosted.npz TARGET=moverpt python src/efn.py  # Fig. 2  — boosted EFN/PFN + weights
python src/teste_irc_scaling.py                              # Fig. 3  — soft & collinear scaling laws
python src/phase_a_genkt.py                                  # Fig. 4  — learned generalized-kt
python src/seeds_efn.py 5                                    # 5-seed robustness (uses the GPU)
```

Finally, rebuild the manuscript:

```bash
python src/build_paper.py                                    # -> paper/paper.html
tectonic -X compile paper/paper.tex --outdir paper           # -> paper/paper.pdf
```

Every script fixes its random seeds, so the numbers in the paper are reproducible.
Note: the paper's tabulated values were produced with a CPU build of PyTorch; a GPU rerun
differs negligibly (floating point + library version).

## The IRC-safety test battery

The reusable core of this work is a quantitative test of IRC safety (`src/teste_irc_scaling.py`,
`src/licao_irc_dbscan.py`): apply an explicit **collinear splitting** and **soft addition** to
an event, measure the induced change in the jets, and check its **scaling law** as the
perturbation vanishes. A safe algorithm's response → 0 as a power law; an unsafe one saturates
at a nonzero plateau.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

## Acknowledgement

Analysis pipeline and manuscript developed interactively with an autonomous coding agent
(Claude, Anthropic), 2026.
