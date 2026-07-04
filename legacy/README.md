# Legacy

`aijet.py` is the **original exploratory script** that started this project. It is kept
here for provenance only and is **not part of the analysis pipeline**.

It clusters with DBSCAN in the raw `(px, py, pz)` momentum space and approximates the jet
energy as `|p|` (massless) — both physically wrong for jets. The correct, IRC-focused study
that superseded it is [`src/licao_irc_dbscan.py`](../src/licao_irc_dbscan.py), which uses the
physical `(y, φ)` plane and proper four-momentum recombination.
