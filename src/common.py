#!/usr/bin/env python3
"""
Utilitarios compartilhados: cinematica, featurizacao e os modelos Deep-Sets
(EFN / PFN). Importado por efn.py, teste_irc_scaling.py e seeds_efn.py para
evitar duplicacao (os tres treinam/avaliam exatamente a mesma arquitetura).
"""
import numpy as np
import torch
import torch.nn as nn

RINV = 2.5   # 1/R: normaliza os angulos de entrada para O(1)


# ---------------------------------------------------------------- cinematica
def kin(p4):
    """(...,4) px,py,pz,E -> (pT, y, phi), vetorizado (aceita (N,4) e (4,))."""
    px, py, pz, E = p4[..., 0], p4[..., 1], p4[..., 2], p4[..., 3]
    pt = np.hypot(px, py)
    y = 0.5 * np.log(np.clip((E + pz) / np.clip(E - pz, 1e-12, None), 1e-12, None))
    return pt, y, np.arctan2(py, px)


def dphi(a, b):
    """Diferenca de azimute com wrap para (-pi, pi]."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def featurize(const_p4, axis_y, axis_phi, PAD):
    """Constituintes (n,4) + eixo do jato -> (ang(PAD,2), z(PAD), mask(PAD)).

    z_i = pT_i / sum pT  (fracao de energia); ang = (y - y_eixo, dphi(phi, phi_eixo)).
    Os constituintes sao ordenados por pT e cortados em PAD; para PAD >= n nao ha
    truncamento (necessario para exatidao colinear)."""
    pt, y, phi = kin(const_p4)
    order = np.argsort(-pt)[:PAD]
    pt, y, phi = pt[order], y[order], phi[order]
    z = pt / pt.sum()
    n = len(pt)
    ang = np.zeros((PAD, 2), np.float32)
    zz = np.zeros(PAD, np.float32)
    msk = np.zeros(PAD, np.float32)
    ang[:n, 0] = y - axis_y
    ang[:n, 1] = dphi(phi, axis_phi)
    zz[:n] = z
    msk[:n] = 1.0
    return ang, zz, msk


# ------------------------------------------------------------------- modelos
def mlp(sizes):
    """MLP com ReLU entre as camadas e SEM ativacao na saida."""
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class EFN(nn.Module):
    """Energy Flow Network — IRC-safe: Phi ve so o angulo, agregacao ponderada
    por z. O(jato) = F( sum_i z_i * Phi(dy_i, dphi_i) )."""
    def __init__(self, L=128):
        super().__init__()
        self.Phi = mlp([2, 128, 128, 128, L])
        self.F = mlp([L, 128, 128, 1])

    def forward(self, ang, z, msk=None):
        phi = self.Phi(ang * RINV)                 # (B, PAD, L)
        agg = (z.unsqueeze(-1) * phi).sum(1)       # z=0 no padding zera a contribuicao
        return self.F(agg).squeeze(-1)


class PFN(nn.Module):
    """Particle Flow Network — NAO IRC-safe: Phi ve z e a agregacao e' soma
    simples (mascarada). O(jato) = F( sum_i Phi(z_i, dy_i, dphi_i) )."""
    def __init__(self, L=128):
        super().__init__()
        self.Phi = mlp([3, 128, 128, 128, L])
        self.F = mlp([L, 128, 128, 1])

    def forward(self, ang, z, msk):
        feat = torch.cat([z.unsqueeze(-1), ang * RINV], dim=-1)   # (B, PAD, 3)
        phi = self.Phi(feat) * msk.unsqueeze(-1)                  # zera o padding
        agg = phi.sum(1)
        return self.F(agg).squeeze(-1)
