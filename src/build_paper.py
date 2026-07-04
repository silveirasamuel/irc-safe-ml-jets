#!/usr/bin/env python3
"""Monta o paper.html autocontido (figuras em base64, equacoes MathML, tabelas)."""
import base64, json

D = json.load(open("results/phase_a_results.json"))
def img(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()

IMG = {
    "DBSCAN":  img("figures/licao_irc_dbscan.png"),
    "EFNBOOST":img("figures/efn_resultados_boosted.png"),
    "SCALING": img("figures/teste_irc_scaling.png"),
    "PHASEA":  img("figures/phase_a_genkt.png"),
}

# numeros da Fase A (do json), formatados
def f(x, n=3): return f"{x:.{n}f}"
PA = dict(
    pstar=f(D["p_star"],1), Rstar=f(D["R_star"],1),
    obj_star=f(D["obj_star"]), sig_star=f(D["sigma_star"]), bias_star=f"{D['bias_star']:+.3f}",
    obj_ak=f(D["obj_antikt"]), sig_ak=f(D["sigma_antikt"]),
    bias_ak=("0.000" if abs(D["bias_antikt"])<5e-4 else f"{D['bias_antikt']:+.3f}"),
    eff=f(D["eff_star"],2), nj_ak=f(D["njets_antikt"],2), nj_st=f(D["njets_star"],2),
    m_ak=f(D["mass_antikt"],1), m_st=f(D["mass_star"],1), overlap=f(D["overlap"],3),
    irc_ak=f(D["irc"]["anti-kt (p=-1)"]["soft"][0],4),
    irc_ca=f(D["irc"]["C/A (p=0)"]["soft"][0],4),
    irc_kt=f(D["irc"]["kt (p=+1)"]["soft"][0],4),
    # chave do algoritmo aprendido: busca por prefixo (o p* exato varia entre runs)
    irc_st=f(D["irc"][next(k for k in D["irc"] if k.startswith("aprendido"))]["soft"][0],4),
    irc_db=f(D["irc"]["DBSCAN"]["soft"][0],4),
)

HTML = r"""<title>IRC-Safe Machine Learning for Jet Clustering</title>
<style>
:root{
  --paper:#f7f4ec; --ink:#1c1e1a; --muted:#6b6f64; --rule:#e0d9c8; --rule2:#eee9dc;
  --head:#153a2e; --safe:#157a5b; --safe-bg:#e6f1ea; --unsafe:#c05617; --unsafe-bg:#f6ebe1;
  --accent:#157a5b; --card:#fbf9f3; --shadow:0 1px 2px rgba(30,40,30,.05);
  --serif:"Charter","Bitstream Charter","Iowan Old Style","Palatino Linotype",Georgia,"Times New Roman",serif;
  --sans:ui-sans-serif,-apple-system,"Segoe UI","Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Code","JetBrains Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root{--paper:#14150f; --ink:#e7e4d8; --muted:#9a9d8f; --rule:#2d2f26; --rule2:#23241d;
    --head:#7fd3b0; --safe:#43c193; --safe-bg:#132a22; --unsafe:#e0894a; --unsafe-bg:#2a1e13;
    --accent:#43c193; --card:#191a12; --shadow:0 1px 2px rgba(0,0,0,.3);}
}
:root[data-theme="light"]{--paper:#f7f4ec; --ink:#1c1e1a; --muted:#6b6f64; --rule:#e0d9c8; --rule2:#eee9dc;
  --head:#153a2e; --safe:#157a5b; --safe-bg:#e6f1ea; --unsafe:#c05617; --unsafe-bg:#f6ebe1; --accent:#157a5b; --card:#fbf9f3;}
:root[data-theme="dark"]{--paper:#14150f; --ink:#e7e4d8; --muted:#9a9d8f; --rule:#2d2f26; --rule2:#23241d;
  --head:#7fd3b0; --safe:#43c193; --safe-bg:#132a22; --unsafe:#e0894a; --unsafe-bg:#2a1e13; --accent:#43c193; --card:#191a12;}

*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);
  font-size:18px;line-height:1.62;font-feature-settings:"kern","liga","onum","pnum";}
.wrap{max-width:940px;margin:0 auto;padding:4.5rem 1.5rem 6rem;}
.col{max-width:720px;margin-inline:auto;}
p{margin:0 0 1.05rem;}
a{color:var(--safe);text-decoration:none;border-bottom:1px solid color-mix(in oklab,var(--safe) 35%,transparent);}
a:hover{border-bottom-color:var(--safe);}
:focus-visible{outline:2px solid var(--safe);outline-offset:2px;border-radius:3px;}

/* header */
.eyebrow{font-family:var(--sans);font-size:.72rem;font-weight:650;letter-spacing:.16em;
  text-transform:uppercase;color:var(--safe);margin:0 0 1.1rem;}
h1.title{font-size:2.55rem;line-height:1.1;letter-spacing:-.015em;margin:0 0 1.1rem;
  font-weight:640;text-wrap:balance;color:var(--head);}
.authors{font-size:1.06rem;margin:0 0 .35rem;}
.affil,.date{font-family:var(--sans);font-size:.82rem;color:var(--muted);margin:0;}
.rule{height:1px;background:var(--rule);border:0;margin:2.4rem 0;}

/* abstract */
.abstract{font-size:.98rem;line-height:1.6;background:var(--card);border:1px solid var(--rule);
  border-left:3px solid var(--accent);border-radius:4px;padding:1.3rem 1.5rem;margin:2.2rem auto 0;box-shadow:var(--shadow);}
.abstract h2{font-family:var(--sans);font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);margin:0 0 .7rem;font-weight:650;}
.abstract p:last-child{margin-bottom:0;}

/* sections */
section{margin-top:3.3rem;}
h2.sec{font-size:1.5rem;letter-spacing:-.01em;color:var(--head);margin:0 0 1.1rem;font-weight:640;text-wrap:balance;}
h2.sec .n{font-family:var(--sans);font-size:.9rem;font-weight:700;color:var(--safe);
  margin-right:.7rem;vertical-align:.12em;}
h3.sub{font-size:1.12rem;color:var(--ink);margin:2rem 0 .6rem;font-weight:640;}
.lead::first-letter{font-size:1.05em;}

/* inline chips + code */
.chip{font-family:var(--sans);font-size:.78em;font-weight:640;padding:.08em .5em;border-radius:100px;
  white-space:nowrap;letter-spacing:.01em;}
.chip.safe{color:var(--safe);background:var(--safe-bg);}
.chip.bad{color:var(--unsafe);background:var(--unsafe-bg);}
code{font-family:var(--mono);font-size:.85em;background:var(--rule2);padding:.1em .38em;border-radius:4px;}
em.v{font-style:italic;}

/* equations */
.eq{display:flex;align-items:center;gap:1rem;margin:1.6rem 0;}
.eq math{font-size:1.18rem;flex:1;overflow-x:auto;}
.eq .num{font-family:var(--sans);font-size:.85rem;color:var(--muted);min-width:2.4rem;text-align:right;}
math{font-family:var(--serif);}

/* figures */
figure{margin:2.4rem 0;}
figure img{width:100%;height:auto;display:block;border:1px solid var(--rule);border-radius:6px;background:#fff;}
figcaption{font-family:var(--sans);font-size:.82rem;line-height:1.5;color:var(--muted);margin-top:.7rem;}
figcaption b{color:var(--ink);font-weight:640;}

/* tables */
.tbl{margin:2.2rem 0;overflow-x:auto;}
.tbl .cap{font-family:var(--sans);font-size:.82rem;color:var(--muted);margin-bottom:.6rem;}
.tbl .cap b{color:var(--ink);font-weight:640;}
table{border-collapse:collapse;width:100%;font-size:.88rem;font-variant-numeric:tabular-nums;}
th,td{padding:.5rem .7rem;text-align:right;border-bottom:1px solid var(--rule2);}
th:first-child,td:first-child{text-align:left;}
thead th{font-family:var(--sans);font-size:.75rem;letter-spacing:.03em;text-transform:uppercase;
  color:var(--muted);border-bottom:1.5px solid var(--rule);font-weight:650;}
tbody tr:hover{background:var(--rule2);}
td.g{color:var(--safe);font-weight:640;} td.o{color:var(--unsafe);font-weight:640;}
.tsec{font-family:var(--sans);font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);padding-top:.9rem!important;}

/* callout */
.key{border:1px solid var(--rule);border-radius:6px;background:var(--card);padding:1.2rem 1.4rem;margin:2.2rem 0;
  display:grid;grid-template-columns:auto 1fr;gap:1rem;align-items:start;box-shadow:var(--shadow);}
.key .mark{font-family:var(--sans);font-weight:800;color:var(--safe);font-size:1.4rem;line-height:1;}
.key p{margin:0;font-size:.96rem;}

/* references */
ol.refs{counter-reset:r;list-style:none;padding:0;margin:1.2rem 0 0;font-size:.85rem;line-height:1.5;}
ol.refs li{counter-increment:r;padding-left:2.4rem;position:relative;margin-bottom:.7rem;color:var(--ink);}
ol.refs li::before{content:"["counter(r)"]";position:absolute;left:0;font-family:var(--sans);
  color:var(--muted);font-size:.8rem;}
.foot{font-family:var(--sans);font-size:.78rem;color:var(--muted);margin-top:3rem;border-top:1px solid var(--rule);padding-top:1.2rem;}
@media (max-width:640px){.wrap{padding:3rem 1.1rem 4rem}h1.title{font-size:2rem}body{font-size:17px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="wrap">
<header class="col">
  <p class="eyebrow">Preprint &middot; Jet Physics &times; Machine Learning</p>
  <h1 class="title">Infrared- and Collinear-Safe Machine Learning for Jet Clustering: Failure Modes, a Quantitative Safety Battery, and a Learned Anti-k<sub>t</sub></h1>
  <p class="authors">Samuel Pedro Pereira Silveira<sup>1</sup> &nbsp;&middot;&nbsp; Mauro Rog&eacute;rio Cosentino<sup>2,3</sup></p>
  <p class="affil"><sup>1</sup>&#8202;Universidade Federal do Tri&acirc;ngulo Mineiro (UFTM) &nbsp;&middot;&nbsp; <sup>2</sup>&#8202;Universidade Federal do ABC (UFABC) &nbsp;&middot;&nbsp; <sup>3</sup>&#8202;ALICE Collaboration, CERN</p>
  <p class="affil">Analysis and manuscript prepared with an autonomous coding agent (Claude, Anthropic)</p>
  <p class="date">July 4, 2026</p>

  <div class="abstract">
    <h2>Abstract</h2>
    <p>Infrared and collinear (IRC) safety &mdash; invariance of an observable under the emission of soft quanta and under collinear splittings &mdash; is the property that makes a jet definition calculable in perturbative QCD. It is a hard, exact symmetry of the physics, and it is <em>not</em> something a neural network acquires by training. We study, on <math><mrow><mn>1.2</mn><mo>&#215;</mo><msup><mn>10</mn><mn>5</mn></msup></mrow></math> jets from PYTHIA&nbsp;8, three ways an algorithm can relate to IRC safety. (i) We show that density-based clustering (DBSCAN) in the physical <math><mrow><mo>(</mo><mi>y</mi><mo>,</mo><mi>&#966;</mi><mo>)</mo></mrow></math> plane is <span class="chip bad">IRC-unsafe</span>: a single <math><mrow><msub><mi>p</mi><mi>T</mi></msub><mo>=</mo><msup><mn>10</mn><mrow><mo>&#8722;</mo><mn>3</mn></mrow></msup></mrow></math>&nbsp;GeV particle alters the reconstructed jets in 6.3% of events. (ii) We show that an Energy Flow Network (EFN), whose energy-weighted Deep-Sets architecture enforces IRC safety <em>by construction</em>, reproduces the anti-k<sub>t</sub> jet mass (correlation 0.996 on boosted jets) while remaining <span class="chip safe">exactly IRC-safe</span> &mdash; its response to a soft or collinear perturbation vanishes as a power law in the limit, whereas an unconstrained Particle Flow Network of equal accuracy saturates at a finite plateau. (iii) We show that the clustering itself can be made learnable while keeping safety exact: optimizing the generalized-k<sub>t</sub> exponent and radius &mdash; a low-dimensional but provably safe family &mdash; against a genuine physics objective (jet&ndash;parton energy response) rediscovers the anti-k<sub>t</sub> exponent <math><mrow><mi>p</mi><mo>=</mo><mo>&#8722;</mo><mn>1</mn></mrow></math> at a mildly enlarged radius. The unifying lesson: in physics-oriented ML, IRC safety must be imposed as a design constraint, never chased as a training target.</p>
  </div>
</header>

<div class="col">

<section>
  <h2 class="sec"><span class="n">1</span>Introduction</h2>
  <p class="lead">Jets &mdash; collimated sprays of hadrons &mdash; are the experimental proxy for the quarks and gluons produced in high-energy collisions. A <em>jet algorithm</em> maps a set of final-state particles onto a set of jets, and the value of any such map hinges on one property: <b>infrared and collinear (IRC) safety</b>. An observable is collinear-safe if replacing a particle by two collinear particles carrying its momentum leaves it unchanged, and infrared-safe if adding an arbitrarily soft particle leaves it unchanged [<a href="#r1">1</a>,&#8202;<a href="#r9">9</a>]. These invariances are exactly the conditions under which the real and virtual divergences of QCD cancel, so an IRC-unsafe quantity is not merely noisy &mdash; it is formally incalculable order by order in perturbation theory.</p>
  <p>The de facto standard at the LHC is the anti-k<sub>t</sub> algorithm [<a href="#r1">1</a>], a sequential-recombination method that is IRC-safe by construction and produces geometrically regular jets. As machine learning permeates jet physics, a tempting shortcut is to replace the algorithm with a generic clustering or neural network trained on data. The central point of this work is that such a substitution silently forfeits IRC safety unless the safety is <em>built into the architecture</em>: no amount of training data confers an exact symmetry that the function class does not possess.</p>
  <p>That energy-weighted, permutation-invariant networks can encode IRC-safe observables is established &mdash; Energy Flow Networks [<a href="#r6">6</a>] and the complete linear basis of energy flow polynomials [<a href="#r11">11</a>] are the known constructions. Our aim is not a new safe observable but a sharp, reusable <em>diagnostic</em> of safety. We contribute (a) a quantitative <em>IRC-safety test battery</em>, built from the explicit soft/collinear perturbations and their scaling laws; (b) using it, a controlled comparison in which an energy-weighted network is safe to machine precision while an otherwise-identical unconstrained network is not &mdash; isolating how energy enters the aggregation, as a multiplicative weight rather than a summed feature, as the decisive factor; and (c) a learnable-yet-provably-safe jet <em>algorithm</em>, obtained by optimizing the generalized-k<sub>t</sub> family against a physics objective.</p>
</section>

<section>
  <h2 class="sec"><span class="n">2</span>Simulation and data</h2>
  <p>Proton&ndash;proton collisions at <math><mrow><msqrt><mi>s</mi></msqrt><mo>=</mo><mn>13.6</mn></mrow></math>&nbsp;TeV were generated with <b>PYTHIA&nbsp;8.317</b> [<a href="#r8">8</a>] (inclusive hard QCD, full parton shower, hadronization and multiparton interactions). Final-state visible particles within <math><mrow><mo>|</mo><mi>&#951;</mi><mo>|</mo><mo>&lt;</mo><mn>5</mn></mrow></math> were clustered with <b>FastJet&nbsp;3.5.1</b> [<a href="#r2">2</a>]. We use two complementary samples: a <em>low-p<sub>T</sub></em> sample (<math><mrow><msub><mover><mi>p</mi><mo>^</mo></mover><mi>T</mi></msub><mo>&gt;</mo><mn>20</mn></mrow></math>&nbsp;GeV, anti-k<sub>t</sub> <em class="v">R</em>&#8202;=&#8202;0.4), where jet mass is dominated by fluctuations, and a <em>boosted</em> sample (<math><mrow><msub><mover><mi>p</mi><mo>^</mo></mover><mi>T</mi></msub><mo>&gt;</mo><mn>450</mn></mrow></math>&nbsp;GeV, <em class="v">R</em>&#8202;=&#8202;0.8 fat jets), where jet mass is a genuine physical scale. Jets are retained with <math><mrow><msub><mi>p</mi><mi>T</mi></msub><mo>&gt;</mo></mrow></math> the generation threshold and <math><mrow><mo>|</mo><mi>y</mi><mo>|</mo><mo>&lt;</mo><mn>2.5</mn></mrow></math>. Each jet stores its constituents, so particle-level perturbations can be re-clustered exactly.</p>

  <div class="tbl">
    <div class="cap"><b>Table 1.</b> The two simulated samples. Cross sections are the PYTHIA-reported generated values.</div>
    <table>
      <thead><tr><th>Sample</th><th>p&#770;<sub>T</sub><sup>min</sup> [GeV]</th><th>Algo / R</th><th>Events</th><th>Jets</th><th>median p<sub>T</sub> [GeV]</th><th>median mass [GeV]</th><th>&#963; [pb]</th></tr></thead>
      <tbody>
        <tr><td>Low-p<sub>T</sub></td><td>20</td><td>anti-k<sub>t</sub> / 0.4</td><td>50&#8202;000</td><td>40&#8202;169</td><td>26.8</td><td>6.4</td><td>7.7&#8202;&#215;&#8202;10<sup>8</sup></td></tr>
        <tr><td>Boosted</td><td>450</td><td>anti-k<sub>t</sub> / 0.8</td><td>50&#8202;000</td><td>80&#8202;610</td><td>539</td><td>82.8</td><td>1.3&#8202;&#215;&#8202;10<sup>3</sup></td></tr>
      </tbody>
    </table>
  </div>
</section>

<section>
  <h2 class="sec"><span class="n">3</span>A quantitative IRC-safety test battery</h2>
  <p>We define two elementary perturbations of an event and a scalar that measures how much the jets move. A <b>collinear splitting</b> replaces the hardest particle <math><mrow><msub><mi>p</mi><mi>i</mi></msub></mrow></math> by two particles <math><mrow><mi>z</mi><msub><mi>p</mi><mi>i</mi></msub></mrow></math> and <math><mrow><mo>(</mo><mn>1</mn><mo>&#8722;</mo><mi>z</mi><mo>)</mo><msub><mi>p</mi><mi>i</mi></msub></mrow></math> pointing in the same direction; a <b>soft addition</b> inserts one particle of transverse momentum <math><mrow><msubsup><mi>p</mi><mi>T</mi><mrow><mi>soft</mi></mrow></msubsup></mrow></math> at a random angle. An IRC-safe algorithm is invariant under the first for every <em class="v">z</em>, and under the second in the limit <math><mrow><msubsup><mi>p</mi><mi>T</mi><mrow><mi>soft</mi></mrow></msubsup><mo>&#8594;</mo><mn>0</mn></mrow></math>.</p>
  <p>To compare the jet content before and after a perturbation we use a transport-style <b>jet-set distance</b>: leading jets are greedily matched within <math><mrow><mi>&#916;</mi><mi>R</mi><mo>&lt;</mo><mi>R</mi></mrow></math> and</p>
  <div class="eq">
    <math display="block"><mrow><mi>D</mi><mo>=</mo><munder><mo>&#8721;</mo><mrow><mi>matched</mi></mrow></munder><mo>|</mo><mi>&#916;</mi><msub><mi>p</mi><mi>T</mi></msub><mo>|</mo><mo>+</mo><munder><mo>&#8721;</mo><mrow><mi>unmatched</mi></mrow></munder><msub><mi>p</mi><mi>T</mi></msub></mrow></math>
    <span class="num">(1)</span>
  </div>
  <p>so that <math><mrow><mi>D</mi><mo>=</mo><mn>0</mn></mrow></math> if and only if the jets are unchanged. The decisive test is not a single number but the <b>scaling law</b> of <math><mrow><mo>&#10216;</mo><mo>|</mo><mi>&#916;</mi><mi>O</mi><mo>|</mo><mo>&#10217;</mo></mrow></math> as <math><mrow><msubsup><mi>p</mi><mi>T</mi><mrow><mi>soft</mi></mrow></msubsup><mo>&#8594;</mo><mn>0</mn></mrow></math>: a safe algorithm's response vanishes (linearly in <math><mrow><msubsup><mi>p</mi><mi>T</mi><mrow><mi>soft</mi></mrow></msubsup></mrow></math>), while an unsafe one approaches a nonzero constant.</p>
</section>

<section>
  <h2 class="sec"><span class="n">4</span>Failure mode: density-based clustering</h2>
  <p>A natural ML-flavoured jet finder is DBSCAN [<a href="#r10">10</a>], which groups particles by density in the physical rapidity&ndash;azimuth plane (we use <math><mrow><mi>&#949;</mi><mo>&#8776;</mo><mi>R</mi></mrow></math>, minimum 4 neighbours). Density is intrinsically infrared-sensitive: an extra soft particle changes the local neighbour counts that define cluster cores, so cluster membership &mdash; and therefore the jets &mdash; can change. Figure&nbsp;1 quantifies this against anti-k<sub>t</sub> on the low-p<sub>T</sub> sample. Anti-k<sub>t</sub> is invariant to machine precision under both perturbations (<math><mrow><mi>D</mi><mo>=</mo><mn>0</mn></mrow></math> in 100% of events); DBSCAN changes the jets in <b>6.3% of events</b> under a single <math><mrow><msup><mn>10</mn><mrow><mo>&#8722;</mo><mn>3</mn></mrow></msup></mrow></math>&nbsp;GeV addition, with a tail of <math><mrow><mi>D</mi></mrow></math> reaching tens of GeV where a soft particle bridges two jets. The precise fraction depends on the DBSCAN neighbourhood parameters (&epsilon; and the minimum number of neighbours); the qualitative failure &mdash; a nonzero, unbounded response to an infinitesimal input &mdash; does not.</p>
  <figure>
    <img alt="IRC-safety test of anti-kt versus DBSCAN" src="@@IMG_DBSCAN@@">
    <figcaption><b>Figure 1.</b> Jet-set distance <em class="v">D</em> under a collinear splitting (left) and a soft addition (right), for anti-k<sub>t</sub> (green) and physical <math><mrow><mo>(</mo><mi>y</mi><mo>,</mo><mi>&#966;</mi><mo>)</mo></mrow></math> DBSCAN (orange), over 2000 events. Anti-k<sub>t</sub> is a delta function at <math><mrow><mi>D</mi><mo>=</mo><mn>0</mn></mrow></math>; DBSCAN develops a tail, most severely in the infrared.</figcaption>
  </figure>
</section>

<section>
  <h2 class="sec"><span class="n">5</span>Restoring exact safety with energy weighting</h2>
  <p>The generalized-k<sub>t</sub> family underlying anti-k<sub>t</sub> clusters particles by the measure</p>
  <div class="eq">
    <math display="block"><mrow>
      <msub><mi>d</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>=</mo><mi>min</mi><mo>(</mo>
      <msubsup><mi>p</mi><mrow><mi>T</mi><mi>i</mi></mrow><mrow><mn>2</mn><mi>p</mi></mrow></msubsup><mo>,</mo>
      <msubsup><mi>p</mi><mrow><mi>T</mi><mi>j</mi></mrow><mrow><mn>2</mn><mi>p</mi></mrow></msubsup><mo>)</mo>
      <mfrac><msubsup><mi>&#916;</mi><mrow><mi>i</mi><mi>j</mi></mrow><mn>2</mn></msubsup><msup><mi>R</mi><mn>2</mn></msup></mfrac>
      <mo>,</mo><mspace width="1.4em"/>
      <msub><mi>d</mi><mrow><mi>i</mi><mi>B</mi></mrow></msub><mo>=</mo><msubsup><mi>p</mi><mrow><mi>T</mi><mi>i</mi></mrow><mrow><mn>2</mn><mi>p</mi></mrow></msubsup>
    </mrow></math>
    <span class="num">(2)</span>
  </div>
  <p>with <math><mrow><msubsup><mi>&#916;</mi><mrow><mi>i</mi><mi>j</mi></mrow><mn>2</mn></msubsup><mo>=</mo><msup><mrow><mo>(</mo><msub><mi>y</mi><mi>i</mi></msub><mo>&#8722;</mo><msub><mi>y</mi><mi>j</mi></msub><mo>)</mo></mrow><mn>2</mn></msup><mo>+</mo><msup><mrow><mo>(</mo><msub><mi>&#966;</mi><mi>i</mi></msub><mo>&#8722;</mo><msub><mi>&#966;</mi><mi>j</mi></msub><mo>)</mo></mrow><mn>2</mn></msup></mrow></math> and <math><mrow><mi>p</mi><mo>=</mo><mo>&#8722;</mo><mn>1</mn></mrow></math> for anti-k<sub>t</sub>. Its IRC safety is structural: soft particles carry vanishing weight and collinear pairs are recombined first. To ask whether a <em>network</em> can be safe, we compare two Deep-Sets architectures [<a href="#r7">7</a>] trained to regress the anti-k<sub>t</sub> jet mass from the constituents.</p>
  <p>The <b>Energy Flow Network</b> (EFN) [<a href="#r6">6</a>] restricts the per-particle map <math><mrow><mi>&#934;</mi></mrow></math> to <em>angular</em> information and weights it by the energy fraction:</p>
  <div class="eq">
    <math display="block"><mrow>
      <mi>O</mi><mo>=</mo><mi>F</mi><mo>&#8289;</mo><mrow><mo>(</mo>
      <munder><mo>&#8721;</mo><mi>i</mi></munder><msub><mi>z</mi><mi>i</mi></msub>
      <mi>&#934;</mi><mo>&#8289;</mo><mrow><mo>(</mo><msub><mover><mi>y</mi><mo>^</mo></mover><mi>i</mi></msub><mo>,</mo><msub><mover><mi>&#966;</mi><mo>^</mo></mover><mi>i</mi></msub><mo>)</mo></mrow>
      <mo>)</mo></mrow><mo>,</mo><mspace width="1.2em"/>
      <msub><mi>z</mi><mi>i</mi></msub><mo>=</mo><mfrac><msub><mi>p</mi><mrow><mi>T</mi><mi>i</mi></mrow></msub><mrow><munder><mo>&#8721;</mo><mi>j</mi></munder><msub><mi>p</mi><mrow><mi>T</mi><mi>j</mi></mrow></msub></mrow></mfrac>
    </mrow></math>
    <span class="num">(3)</span>
  </div>
  <p>A soft particle has <math><mrow><msub><mi>z</mi><mi>i</mi></msub><mo>&#8594;</mo><mn>0</mn></mrow></math> and contributes nothing; a collinear splitting shares <math><mrow><msub><mi>z</mi><mi>i</mi></msub></mrow></math> between two identical angular points and leaves the sum invariant. As a deliberately unsafe control, the <b>Particle Flow Network</b> (PFN) lets <math><mrow><mi>&#934;</mi></mrow></math> see <math><mrow><msub><mi>z</mi><mi>i</mi></msub></mrow></math> and aggregates by a plain sum, <math><mrow><mi>O</mi><mo>=</mo><mi>F</mi><mo>(</mo><msub><mo>&#8721;</mo><mi>i</mi></msub><mi>&#934;</mi><mo>(</mo><msub><mi>z</mi><mi>i</mi></msub><mo>,</mo><msub><mover><mi>y</mi><mo>^</mo></mover><mi>i</mi></msub><mo>,</mo><msub><mover><mi>&#966;</mi><mo>^</mo></mover><mi>i</mi></msub><mo>)</mo><mo>)</mo></mrow></math>, so that <math><mrow><mi>&#934;</mi><mo>(</mo><mn>0</mn><mo>,</mo><mo>&#8901;</mo><mo>)</mo><mo>&#8800;</mo><mn>0</mn></mrow></math> injects a fixed soft contribution. The EFN and PFN are the two canonical Deep-Sets variants of Ref.&nbsp;[<a href="#r6">6</a>]; they differ precisely in whether energy is the aggregation <em>weight</em> or a summed per-particle <em>feature</em>, and it is that choice &mdash; not depth, width or training &mdash; that decides safety.</p>

  <h3 class="sub">5.1&ensp;Equal accuracy, unequal safety</h3>
  <p>On boosted jets the two networks are statistically indistinguishable as mass regressors (correlation 0.996 vs 0.995; Fig.&nbsp;2, Table&nbsp;2), yet their behaviour under perturbation could not be more different. The EFN is invariant to machine precision under collinear splitting in <em>both</em> samples, and its soft response is smaller than the PFN's by a factor of about 20 on low-p<sub>T</sub> jets and ~600 on boosted jets. Crucially, the PFN's marginally better fit on the low-p<sub>T</sub> sample is obtained precisely by exploiting the IRC-unsafe information (soft counts, collinear multiplicity) that its safe counterpart is forbidden to use. The contrast is stable: over five random initializations (boosted sample) the EFN correlation is 0.9959&#8202;&plusmn;&#8202;0.0001 and its soft response 0.010&#8202;&plusmn;&#8202;0.000&nbsp;GeV, with the collinear response identically zero in every run, whereas the PFN&rsquo;s soft response is not only large but erratic, 9.1&#8202;&plusmn;&#8202;2.1&nbsp;GeV (collinear 0.27&#8202;&plusmn;&#8202;0.03&nbsp;GeV). The safe network is reproducible in both accuracy and safety; the unsafe one is unstable in its very unsafety.</p>
  <figure>
    <img alt="EFN and PFN jet-mass regression and IRC test, boosted sample" src="@@IMG_EFNBOOST@@">
    <figcaption><b>Figure 2.</b> Boosted sample. Top: predicted vs. true anti-k<sub>t</sub> jet mass for the EFN (left) and PFN (right); both reach correlation <math><mrow><mi>r</mi><mo>&#8776;</mo><mn>0.996</mn></mrow></math>. Bottom: distribution of the induced change <math><mrow><mo>|</mo><mi>&#916;</mi><mi>m</mi><mo>|</mo></mrow></math> under collinear (left) and soft (right) perturbations. The EFN piles up at machine zero; the PFN spreads to several GeV.</figcaption>
  </figure>

  <div class="tbl">
    <div class="cap"><b>Table 2.</b> Mass-regression accuracy and IRC response of the two networks. <math><mrow><mo>|</mo><mi>&#916;</mi><mi>m</mi><mo>|</mo></mrow></math> is the mean induced mass change [GeV]; collinear uses <math><mrow><mi>z</mi><mo>=</mo><mfrac><mn>1</mn><mn>2</mn></mfrac></mrow></math>, soft uses <math><mrow><msubsup><mi>p</mi><mi>T</mi><mrow><mi>soft</mi></mrow></msubsup><mo>=</mo><msup><mn>10</mn><mrow><mo>&#8722;</mo><mn>3</mn></mrow></msup></mrow></math>&nbsp;GeV.</div>
    <table>
      <thead><tr><th>Model</th><th>corr</th><th>RMSE [GeV]</th><th>|&#916;m| collinear</th><th>|&#916;m| soft</th></tr></thead>
      <tbody>
        <tr><td class="tsec" colspan="5">Low-p<sub>T</sub> sample &mdash; jet mass in GeV</td></tr>
        <tr><td>EFN <span class="chip safe">safe</span></td><td>0.37</td><td>2.41</td><td class="g">0.00000</td><td class="g">0.133</td></tr>
        <tr><td>PFN <span class="chip bad">unsafe</span></td><td>0.70</td><td>1.85</td><td class="o">0.296</td><td class="o">2.744</td></tr>
        <tr><td class="tsec" colspan="5">Boosted sample &mdash; jet mass in GeV</td></tr>
        <tr><td>EFN <span class="chip safe">safe</span></td><td>0.996</td><td>4.59</td><td class="g">0.00000</td><td class="g">0.011</td></tr>
        <tr><td>PFN <span class="chip bad">unsafe</span></td><td>0.995</td><td>5.25</td><td class="o">0.225</td><td class="o">6.429</td></tr>
      </tbody>
    </table>
  </div>

  <h3 class="sub">5.2&ensp;The scaling law is the proof</h3>
  <p>A single perturbation size is suggestive; the limit is decisive. Figure&nbsp;3 probes both limits over many decades, in double precision to expose the true power law. Under a soft emission (left) the EFN response falls linearly (fitted slope 0.98), reaching 1.4&#215;10<sup>&#8722;4</sup>&nbsp;GeV at p<sub>T</sub><sup>soft</sup>&#8202;=&#8202;10<sup>&#8722;6</sup>&nbsp;GeV, while the PFN saturates at a plateau of &#8776;2.6&nbsp;GeV independent of p<sub>T</sub><sup>soft</sup>: the contribution &Phi;(0,&middot;) of a zero-energy particle does not disappear. Under a collinear splitting of opening angle &theta; (centre) the same pattern recurs &mdash; the EFN again vanishes as a power law (fitted slope 0.96); the linear rather than quadratic scaling reflects the piecewise-linear ReLU map &Phi; &mdash; a smooth activation would give the &theta;<sup>2</sup> of a symmetric split &mdash; whereas the PFN plateaus at &#8776;0.29&nbsp;GeV. For an exactly collinear split (&theta;&#8202;=&#8202;0, right) the EFN is invariant to machine precision (&#8818;10<sup>&#8722;7</sup>&nbsp;GeV) for every momentum fraction <em class="v">z</em>; the PFN is not (&#8776;0.3&nbsp;GeV). The vanishing of the EFN response in <em>both</em> limits is the operational content of IRC safety; the PFN’s two finite plateaus are the receipts of its unsafety.</p>
  <figure>
    <img alt="Infrared and collinear scaling laws for EFN and PFN" src="@@IMG_SCALING@@">
    <figcaption><b>Figure 3.</b> IRC safety as scaling laws. Left: mean predicted-mass shift vs. the soft particle’s p<sub>T</sub> (log&ndash;log); the EFN vanishes as a power law, the PFN plateaus at &#8776;2.6&nbsp;GeV. Centre: shift vs. the collinear opening angle &theta;; the EFN vanishes linearly (the ReLU map gives &theta;<sup>1</sup> rather than &theta;<sup>2</sup>), the PFN plateaus at &#8776;0.29&nbsp;GeV. Right: shift vs. the exactly-collinear momentum fraction <em class="v">z</em> (&theta;&#8202;=&#8202;0); the EFN is flat at machine precision, the PFN is not.</figcaption>
  </figure>
</section>

<section>
  <h2 class="sec"><span class="n">6</span>A learned, provably safe jet algorithm</h2>
  <p>Sections 4&ndash;5 concern observables computed <em>on</em> jets. We now make the clustering itself learnable while keeping IRC safety exact, as a proof of concept within a low-dimensional but provably safe family. Rather than a non-differentiable neural partition, we treat the generalized-k<sub>t</sub> measure of Eq.&nbsp;(2) as a two-parameter model &mdash; energy exponent <math><mrow><mi>p</mi></mrow></math> and radius <math><mrow><mi>R</mi></mrow></math> &mdash; and optimize it against a real physics objective on the boosted sample: the root-mean-square of the jet&ndash;parton energy response,</p>
  <div class="eq">
    <math display="block"><mrow>
      <mi>&#8499;</mi><mo>(</mo><mi>p</mi><mo>,</mo><mi>R</mi><mo>)</mo><mo>=</mo>
      <msqrt><mrow><msup><mi>b</mi><mn>2</mn></msup><mo>+</mo><msup><mi>&#963;</mi><mn>2</mn></msup></mrow></msqrt><mo>,</mo><mspace width="1em"/>
      <mi>r</mi><mo>=</mo><mfrac><msubsup><mi>p</mi><mi>T</mi><mrow><mi>jet</mi></mrow></msubsup><msubsup><mi>p</mi><mi>T</mi><mrow><mi>parton</mi></mrow></msubsup></mfrac>
    </mrow></math>
    <span class="num">(4)</span>
  </div>
  <p>where <math><mrow><mi>b</mi><mo>=</mo><mi>median</mi><mo>(</mo><mi>r</mi><mo>)</mo><mo>&#8722;</mo><mn>1</mn></mrow></math> is the energy-scale bias and <math><mrow><mi>&#963;</mi></mrow></math> the resolution, each measured against the outgoing hard partons (PYTHIA status <math><mrow><mo>&#177;</mo><mn>23</mn></mrow></math>). This objective has a genuine interior optimum: too small a radius misses final-state radiation (<math><mrow><mi>b</mi><mo>&lt;</mo><mn>0</mn></mrow></math>), too large a radius absorbs the underlying event (<math><mrow><mi>b</mi><mo>&gt;</mo><mn>0</mn></mrow></math>). Every point of the scanned family is a sequential-recombination algorithm and hence IRC-safe by construction &mdash; the safety is a hard constraint on the search, not a term in the loss.</p>
  <p>The optimum (Fig.&nbsp;4, left) lands at <math><mrow><msup><mi>p</mi><mo>&#8727;</mo></msup><mo>=</mo><mo>&#8722;</mo><mn>1.0</mn></mrow></math>, <math><mrow><msup><mi>R</mi><mo>&#8727;</mo></msup><mo>=</mo><mn>1.0</mn></mrow></math>: the search recovers the anti-k<sub>t</sub> energy exponent and prefers a radius only slightly larger than the conventional <math><mrow><mi>R</mi><mo>=</mo><mn>0.8</mn></mrow></math>. The improvement is marginal, however (@@OBJ_AK@@ &rarr; @@OBJ_STAR@@), and the exponent <math><mrow><mi>p</mi></mrow></math> is weakly constrained &mdash; the objective is shallow across the whole family. That shallowness is itself the message: it quantifies why anti-k<sub>t</sub>'s soft-resilient, rigid-cone behaviour is such a robust default. The learned algorithm is best read not as <em>beating</em> anti-k<sub>t</sub> but as <em>rediscovering</em> it from a physics objective while never leaving the IRC-safe family. The IRC battery (Fig.&nbsp;4, right) confirms that the learned algorithm, together with the whole recombination family, has a soft response of <math><mrow><mo>&#8776;</mo><msup><mn>10</mn><mrow><mo>&#8722;</mo><mn>4</mn></mrow></msup></mrow></math>&nbsp;GeV &mdash; consistent with floating-point noise &mdash; while DBSCAN sits four orders of magnitude higher.</p>
  <figure>
    <img alt="Learned generalized-kt: objective landscape and IRC battery" src="@@IMG_PHASEA@@">
    <figcaption><b>Figure 4.</b> Left: the physics objective <math><mrow><mi>&#8499;</mi><mo>(</mo><mi>p</mi><mo>,</mo><mi>R</mi><mo>)</mo></mrow></math> over the generalized-k<sub>t</sub> family; the star marks the learned optimum <math><mrow><mo>(</mo><msup><mi>p</mi><mo>&#8727;</mo></msup><mo>,</mo><msup><mi>R</mi><mo>&#8727;</mo></msup><mo>)</mo><mo>=</mo><mo>(</mo><mo>&#8722;</mo><mn>1</mn><mo>,</mo><mn>1.0</mn><mo>)</mo></mrow></math>, the circle the anti-k<sub>t</sub> reference. Right: mean soft-perturbation distance <math><mrow><mo>&#10216;</mo><mi>D</mi><mo>&#10217;</mo></mrow></math> per algorithm (log scale); all recombination algorithms (green) are safe, DBSCAN (orange) is not.</figcaption>
  </figure>

  <div class="tbl">
    <div class="cap"><b>Table 3.</b> Learned generalized-k<sub>t</sub> vs. the anti-k<sub>t</sub> <math><mrow><mi>R</mi><mo>=</mo><mn>0.8</mn></mrow></math> reference on boosted jets. The learned jets fully contain the anti-k<sub>t</sub> jets (constituent-p<sub>T</sub> overlap = 1.00), being the same exponent at a larger radius.</div>
    <table>
      <thead><tr><th>Algorithm</th><th>RMS resp.</th><th>bias b</th><th>&#963;</th><th>jets/event</th><th>mean mass [GeV]</th><th>&#10216;D&#10217; soft [GeV]</th></tr></thead>
      <tbody>
        <tr><td>anti-k<sub>t</sub> (p=&#8722;1, R=0.8)</td><td>@@OBJ_AK@@</td><td>@@BIAS_AK@@</td><td>@@SIG_AK@@</td><td>@@NJ_AK@@</td><td>@@M_AK@@</td><td class="g">@@IRC_AK@@</td></tr>
        <tr><td>learned (p*=&#8722;1, R*=1.0)</td><td class="g">@@OBJ_STAR@@</td><td>@@BIAS_STAR@@</td><td class="g">@@SIG_STAR@@</td><td>@@NJ_ST@@</td><td>@@M_ST@@</td><td class="g">@@IRC_ST@@</td></tr>
        <tr><td>DBSCAN (&#949;=0.8)</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td><td class="o">@@IRC_DB@@</td></tr>
      </tbody>
    </table>
  </div>
</section>

<section>
  <h2 class="sec"><span class="n">7</span>Discussion</h2>
  <p>The three studies form a single argument. DBSCAN shows that adopting a generic clustering method imports its symmetries &mdash; and density has the wrong ones. The EFN&ndash;PFN pair isolates the mechanism: two networks of equal expressive power and near-identical accuracy diverge completely in safety, decided entirely by whether energy enters as a linear weight (safe) or as a feature summed with unit weight (unsafe). The scaling law elevates this from "small vs. large" to the exact limit that defines the property. Finally, the learned algorithm shows the constructive route: parametrize <em>within</em> a provably safe family and let optimization choose &mdash; here recovering anti-k<sub>t</sub>'s exponent and quantifying the mild preference for a larger radius once the underlying event is present.</p>
  <p>Limitations are worth stating plainly. The learnable clustering explores a two-parameter family, not an arbitrary neural partition; extending learnable safe clustering to a richer, still-safe hypothesis class (for instance an energy-weighted, permutation-equivariant metric inside the recombination loop) is the natural next step. The jet-mass regression targets a single observable; the EFN's low correlation on low-p<sub>T</sub> jets is itself physical &mdash; at that scale the IRC-safe information content of the mass is genuinely small &mdash; and it rises to 0.996 once the mass becomes a real scale. All results are at particle level without detector effects or pileup, whose interaction with IRC safety is a known and important separate axis. Finally, all quoted numbers derive from the finite samples of Table&nbsp;1; the sensitivity of the network results to initialization is quantified in Sec.&nbsp;5.1, and the central claim of the scaling laws &mdash; exact vanishing versus a finite plateau &mdash; does not depend on sample size. All results also use a single event generator (PYTHIA&nbsp;8); a cross-check against an independent parton-shower and hadronization model would probe generator dependence.</p>
  <div class="key">
    <div class="mark">&#10003;</div>
    <p><b>Takeaway.</b> IRC safety is an exact symmetry of the physics, not a statistic. A model is safe only if its <em>architecture</em> makes it so &mdash; energy weighting for observables, sequential recombination for clustering. Imposed as a constraint it costs essentially nothing; chased as a training target it is never actually reached, and the plateau in Fig.&nbsp;3 is the receipt.</p>
  </div>
</section>

<section>
  <h2 class="sec"><span class="n">8</span>Conclusions</h2>
  <p>We provided a quantitative, reusable battery for testing infrared and collinear safety of jet algorithms and machine-learning models, and used it to (i) diagnose density-based clustering as IRC-unsafe, (ii) demonstrate that energy-weighted Deep-Sets restore exact safety at no cost in accuracy, with the soft-momentum scaling law as rigorous proof, and (iii) construct a learnable jet algorithm that is IRC-safe by construction and rediscovers the anti-k<sub>t</sub> exponent when optimized against jet&ndash;parton energy response. The through-line for physics-oriented machine learning is that exact symmetries belong in the function class, not in the loss.</p>
</section>

<section>
  <h2 class="sec"><span class="n">9</span>Data, code and reproducibility</h2>
  <p>All results were produced on a single CPU (Intel i5-13450HX) with PYTHIA&nbsp;8.317, FastJet&nbsp;3.5.1, PyTorch&nbsp;2.12.1 (CPU), scikit-learn&nbsp;1.9.0, NumPy&nbsp;2.2.4 and Python&nbsp;3.13. The pipeline is fully scripted and seeded: <code>gerar_dados.py</code> (event generation and anti-k<sub>t</sub> reference), <code>licao_irc_dbscan.py</code> (Fig.&nbsp;1), <code>efn.py</code> (EFN/PFN training, Fig.&nbsp;2, Table&nbsp;2), <code>teste_irc_scaling.py</code> (Fig.&nbsp;3), and <code>phase_a_genkt.py</code> (Fig.&nbsp;4, Table&nbsp;3). Networks use a shared Deep-Sets backbone (per-particle MLP <math><mrow><mn>2</mn></mrow></math>/<math><mrow><mn>3</mn></mrow></math>&#8202;&#8594;&#8202;128&#8202;&#8594;&#8202;128&#8202;&#8594;&#8202;128, latent dimension 128, aggregation, MLP head), Adam, 80 epochs; the EFN predicts the dimensionless ratio <math><mrow><mi>m</mi><mo>/</mo><msub><mi>p</mi><mi>T</mi></msub></mrow></math> on boosted jets and is rescaled by the (IRC-safe) jet <math><mrow><msub><mi>p</mi><mi>T</mi></msub></mrow></math>.</p>
</section>

<section>
  <h2 class="sec"><span class="n">&nbsp;</span>References</h2>
  <ol class="refs">
    <li id="r1">M. Cacciari, G. P. Salam, G. Soyez. <em>The anti-k<sub>t</sub> jet clustering algorithm.</em> JHEP <b>04</b> (2008) 063. arXiv:0802.1189.</li>
    <li id="r2">M. Cacciari, G. P. Salam, G. Soyez. <em>FastJet user manual.</em> Eur. Phys. J. C <b>72</b> (2012) 1896. arXiv:1111.6097.</li>
    <li id="r3">S. D. Ellis, D. E. Soper. <em>Successive combination jet algorithm for hadron collisions.</em> Phys. Rev. D <b>48</b> (1993) 3160. arXiv:hep-ph/9305266.</li>
    <li id="r4">S. Catani, Yu. L. Dokshitzer, M. H. Seymour, B. R. Webber. <em>Longitudinally-invariant k<sub>&#8869;</sub>-clustering algorithms for hadron&ndash;hadron collisions.</em> Nucl. Phys. B <b>406</b> (1993) 187.</li>
    <li id="r5">Y. L. Dokshitzer, G. D. Leder, S. Moretti, B. R. Webber. <em>Better jet clustering algorithms.</em> JHEP <b>08</b> (1997) 001. arXiv:hep-ph/9707323.</li>
    <li id="r6">P. T. Komiske, E. M. Metodiev, J. Thaler. <em>Energy Flow Networks: Deep Sets for particle jets.</em> JHEP <b>01</b> (2019) 121. arXiv:1810.05165.</li>
    <li id="r7">M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. Salakhutdinov, A. Smola. <em>Deep Sets.</em> NeurIPS <b>30</b> (2017). arXiv:1703.06114.</li>
    <li id="r8">C. Bierlich et al. <em>A comprehensive guide to the physics and usage of PYTHIA 8.3.</em> SciPost Phys. Codebases <b>8</b> (2022). arXiv:2203.11601.</li>
    <li id="r9">G. P. Salam. <em>Towards jetography.</em> Eur. Phys. J. C <b>67</b> (2010) 637. arXiv:0906.1833.</li>
    <li id="r10">M. Ester, H.-P. Kriegel, J. Sander, X. Xu. <em>A density-based algorithm for discovering clusters in large spatial databases with noise.</em> Proc. KDD-96 (1996) 226.</li>
    <li id="r11">P. T. Komiske, E. M. Metodiev, J. Thaler. <em>Energy flow polynomials: a complete linear basis for jet substructure.</em> JHEP <b>04</b> (2018) 013. arXiv:1712.07124.</li>
    <li id="r12">G. Sterman, S. Weinberg. <em>Jets from quantum chromodynamics.</em> Phys. Rev. Lett. <b>39</b> (1977) 1436.</li>
    <li id="r13">A. Paszke et al. <em>PyTorch: an imperative style, high-performance deep learning library.</em> NeurIPS <b>32</b> (2019). arXiv:1912.01703.</li>
    <li id="r14">F. Pedregosa et al. <em>Scikit-learn: machine learning in Python.</em> JMLR <b>12</b> (2011) 2825.</li>
  </ol>
  <p class="foot">Manuscript and analysis pipeline developed interactively with an autonomous coding agent (Claude, Anthropic), July 2026. Figures generated directly from the scripts above; every number in the tables is reproduced by rerunning the pipeline with the fixed random seeds.</p>
</section>

</div>
</div>
"""

REPL = {
    "@@IMG_DBSCAN@@": IMG["DBSCAN"],
    "@@IMG_EFNBOOST@@": IMG["EFNBOOST"], "@@IMG_SCALING@@": IMG["SCALING"],
    "@@IMG_PHASEA@@": IMG["PHASEA"],
    "@@OBJ_AK@@": PA["obj_ak"], "@@OBJ_STAR@@": PA["obj_star"],
    "@@SIG_AK@@": PA["sig_ak"], "@@SIG_STAR@@": PA["sig_star"],
    "@@BIAS_AK@@": PA["bias_ak"], "@@BIAS_STAR@@": PA["bias_star"],
    "@@NJ_AK@@": PA["nj_ak"], "@@NJ_ST@@": PA["nj_st"],
    "@@M_AK@@": PA["m_ak"], "@@M_ST@@": PA["m_st"],
    "@@IRC_AK@@": PA["irc_ak"], "@@IRC_ST@@": PA["irc_st"], "@@IRC_DB@@": PA["irc_db"],
}
for k, v in REPL.items():
    HTML = HTML.replace(k, v)

open("paper/paper.html", "w").write(HTML)
print(f">> paper.html escrito ({len(HTML)/1024:.0f} KB, {len(HTML)} chars)")
# checagem: nenhum token @@ sobrou
import re
left = re.findall(r"@@[A-Z_]+@@", HTML)
print("tokens nao-substituidos:", set(left) if left else "nenhum")
