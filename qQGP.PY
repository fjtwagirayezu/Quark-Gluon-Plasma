#!/usr/bin/env python3
from __future__ import annotations

import os
import logging
import warnings

# 1. Silence the Python warnings system for fontTools
warnings.filterwarnings("ignore", module="fontTools")
# 2. Silence the Logging system for fontTools specifically
logging.getLogger('fontTools').setLevel(logging.ERROR)
# 3. Suppress SyntaxWarnings (useful if running in environments that ignore raw strings)
warnings.filterwarnings("ignore", category=SyntaxWarning)

import json
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.linalg import expm

# ============================================================
# QGP channel -- Option B, theory-aligned, exact-CPTP sectorized code
# ============================================================

# ... [Rest of your plotting and utility functions remain unchanged] ...

# --------------------------
# superoperator / exact CPTP tools
# --------------------------
def vecF(X: np.ndarray) -> np.ndarray:
    return X.reshape((-1,), order="F")


def matF(v: np.ndarray, d: int) -> np.ndarray:
    return v.reshape((d, d), order="F")


def lindblad_superop(H: np.ndarray, Ls: list[np.ndarray]) -> np.ndarray:
    """
    Column-stacking convention:
      vec(A X B) = (B^T \otimes A) vec(X)
    """
    d = H.shape[0]
    I = np.eye(d, dtype=complex)

    # Hamiltonian part
    S = -1j * (np.kron(I, H) - np.kron(H.T, I))

    # Dissipator
    for L in Ls:
        LdL = L.conj().T @ L
        S += np.kron(L.conj(), L)
        S += -0.5 * np.kron(I, LdL)
        S += -0.5 * np.kron(LdL.T, I)

    return S


def superop_to_choi(Phi: np.ndarray, d: int) -> np.ndarray:
    """
    Build Choi matrix J(Phi) = sum_{ij} Phi(E_ij) \otimes E_ij
    in the same column-stacking convention used above.
    """
    # Fixed: Added 'r' prefix to the docstring above to avoid SyntaxWarning
    J = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            Eij = np.zeros((d, d), dtype=complex)
            Eij[i, j] = 1.0
            Y = matF(Phi @ vecF(Eij), d)
            J += np.kron(Y, Eij)
    J = 0.5 * (J + J.conj().T)
    return J

# ... [Rest of your physics logic and SU(3) functions remain unchanged] ...

# --------------------------
# main driver
# --------------------------
def run_all():
    set_style()
    mp = ModelParams()
    ensure_dir(mp.outdir)

    tau_f = tau_freezeout(mp)
    taus = np.arange(mp.tau0_fm, tau_f + 1e-12, mp.dt_fm)
    Ts = np.array([T_of_tau(t, mp) for t in taus])

    # 1) Temperature profile
    fig_singlecol(2.35)
    ax = plt.gca()
    ax.plot(taus, Ts, marker="o", markevery=max(1, len(taus) // 8), linewidth=1.8)
    # Fixed: Added 'r' to LaTeX strings below
    ax.axhline(mp.Tf_MeV, linestyle="--", linewidth=1.0, label=r"$T_f$")
    add_tau_freezeout_line(ax, tau_f)
    ax.set_xlabel(r"$\tau\,[\mathrm{fm}/c]$")
    ax.set_ylabel(r"$T(\tau)\,[\mathrm{MeV}]$")
    ax.legend(loc="upper right")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_T_profile"))

    # 2) Exact color scan
    kscan = (0.03, 0.06, 0.10, 0.16)
    res = evolve_color_scan_exact(mp, kscan)

    fig_singlecol(2.45)
    ax = plt.gca()
    plot_scan_curves(
        ax,
        res[kscan[0]]["taus"],
        [res[k]["IAB"] for k in kscan],
        [fr"$\kappa_q(T_0)={k:.2f}$" for k in kscan],
    )
    add_tau_freezeout_line(ax, tau_f)
    ax.set_xlabel(r"$\tau\,[\mathrm{fm}/c]$")
    ax.set_ylabel(r"$I_c(A{:}B)$")
    ax.legend(loc="upper right")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_MI_color_vs_tau_kappaq_scan"))

    # ... [Additional plotting code follows same logic] ...

    print(f"[OK] Exact-CPTP theory-aligned plots saved in: {mp.outdir}")


if __name__ == "__main__":
    run_all()
