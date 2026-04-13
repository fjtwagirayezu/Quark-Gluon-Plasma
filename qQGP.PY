#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.linalg import expm

# ============================================================
# QGP channel -- Option B, theory-aligned, exact-CPTP sectorized code
# ============================================================
#
# What this version fixes relative to the earlier plotting code:
#   1) exact local GKSL channels via matrix exponentials
#   2) no Euler+renormalize step
#   3) explicit ancilla-based color recombination at freeze-out
#   4) minimal explicit flavor/hadron sector for K/pi
#
# Practical note:
#   Exact dense evolution on the full Hilbert space of the paper is too large
#   for a tractable single-file script. This code therefore implements the
#   exact-CPTP dynamics sector-by-sector:
#       - color sector: 3 x 3bar x 8
#       - energy sector: NE x NE
#       - minimal flavor/hadron freeze-out sector
#   This is the exact sectorized implementation relevant to the plotted figures.
# ============================================================


# --------------------------
# plotting utilities
# --------------------------
def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def fig_singlecol(height=2.25):
    return plt.figure(figsize=(3.6, height))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def finalize_ax(ax, add_grid=True):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if add_grid:
        ax.grid(alpha=0.20, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path + ".pdf")
    plt.savefig(path + ".png")
    plt.close()


# --------------------------
# linear algebra
# --------------------------
def basis(d: int, i: int) -> np.ndarray:
    v = np.zeros((d, 1), dtype=complex)
    v[i, 0] = 1.0
    return v


def density_from_state(psi: np.ndarray) -> np.ndarray:
    psi = psi.reshape((-1, 1))
    return psi @ psi.conj().T


def kronN(*ops) -> np.ndarray:
    out = np.array([[1.0]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out


def partial_trace(rho: np.ndarray, dims: tuple[int, ...], keep: tuple[int, ...]) -> np.ndarray:
    dims = tuple(dims)
    keep = tuple(keep)
    n = len(dims)
    reshaped = rho.reshape(dims + dims)
    trace_out = [k for k in range(n) if k not in keep]
    cur_n = n
    for k in sorted(trace_out, reverse=True):
        reshaped = np.trace(reshaped, axis1=k, axis2=k + cur_n)
        cur_n -= 1
    kept_dims = [dims[k] for k in keep]
    dkeep = int(np.prod(kept_dims))
    return reshaped.reshape((dkeep, dkeep))


def von_neumann_entropy(rho: np.ndarray, tol: float = 1e-12) -> float:
    rhoH = 0.5 * (rho + rho.conj().T)
    vals = np.linalg.eigvalsh(rhoH)
    vals = np.real(vals)
    vals = vals[vals > tol]
    if vals.size == 0:
        return 0.0
    return float(-np.sum(vals * np.log2(vals)))


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def mutual_information(rho: np.ndarray, dims: tuple[int, ...], keepA: tuple[int, ...], keepB: tuple[int, ...]) -> float:
    rhoA = partial_trace(rho, dims, keepA)
    rhoB = partial_trace(rho, dims, keepB)
    keepAB = tuple(sorted(set(keepA + keepB)))
    rhoAB = partial_trace(rho, dims, keepAB)
    return von_neumann_entropy(rhoA) + von_neumann_entropy(rhoB) - von_neumann_entropy(rhoAB)


def partial_transpose_two_party(rhoAB: np.ndarray, dA: int, dB: int, sys: int = 1) -> np.ndarray:
    arr = rhoAB.reshape((dA, dB, dA, dB))
    if sys == 1:
        arr = arr.transpose(0, 3, 2, 1)
    else:
        arr = arr.transpose(2, 1, 0, 3)
    return arr.reshape((dA * dB, dA * dB))


def log_negativity(rhoAB: np.ndarray, dA: int, dB: int) -> float:
    rhoPT = partial_transpose_two_party(rhoAB, dA, dB, sys=1)
    svals = np.linalg.svd(rhoPT, compute_uv=False)
    return max(0.0, float(np.log2(np.sum(svals))))


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
    J = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            Eij = np.zeros((d, d), dtype=complex)
            Eij[i, j] = 1.0
            Y = matF(Phi @ vecF(Eij), d)
            J += np.kron(Y, Eij)
    J = 0.5 * (J + J.conj().T)
    return J


def choi_to_kraus(J: np.ndarray, d: int, tol: float = 1e-12) -> list[np.ndarray]:
    vals, vecs = np.linalg.eigh(J)
    kraus = []
    for lam, v in zip(vals, vecs.T):
        if lam > tol:
            K = np.sqrt(lam) * matF(v, d)
            kraus.append(K)
    return kraus


def exact_local_channel_kraus(H: np.ndarray, Ls: list[np.ndarray], dt: float) -> list[np.ndarray]:
    d = H.shape[0]
    S = lindblad_superop(H, Ls)
    Phi = expm(dt * S)
    J = superop_to_choi(Phi, d)
    return choi_to_kraus(J, d)


def apply_local_channel(rho: np.ndarray, dims: tuple[int, ...], target: int, kraus_local: list[np.ndarray]) -> np.ndarray:
    d_target = dims[target]
    out = np.zeros_like(rho)
    for K in kraus_local:
        ops = []
        for i, d in enumerate(dims):
            ops.append(K if i == target else np.eye(d, dtype=complex))
        Kbig = kronN(*ops)
        out += Kbig @ rho @ Kbig.conj().T
    out = 0.5 * (out + out.conj().T)
    out /= np.trace(out)
    return out


def permute_systems(rho: np.ndarray, dims: tuple[int, ...], perm: tuple[int, ...]) -> np.ndarray:
    """
    Return rho with subsystem order permuted.
    """
    n = len(dims)
    rho_t = rho.reshape(dims + dims)
    axes = tuple(list(perm) + [n + p for p in perm])
    rho_p = np.transpose(rho_t, axes=axes)
    new_dims = tuple(dims[p] for p in perm)
    return rho_p.reshape((int(np.prod(new_dims)), int(np.prod(new_dims))))


def inverse_permutation(perm: tuple[int, ...]) -> tuple[int, ...]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


def apply_two_body_channel(rho: np.ndarray, dims: tuple[int, ...], targets: tuple[int, int], kraus_pair: list[np.ndarray]) -> np.ndarray:
    """
    Apply a two-body channel on arbitrary subsystem targets.
    """
    i, j = targets
    n = len(dims)
    rest = tuple(k for k in range(n) if k not in targets)
    perm = (i, j) + rest
    invp = inverse_permutation(perm)

    rho_p = permute_systems(rho, dims, perm)
    dims_p = tuple(dims[p] for p in perm)
    d1, d2 = dims_p[0], dims_p[1]
    drest = int(np.prod(dims_p[2:]))

    out_p = np.zeros_like(rho_p)
    Irest = np.eye(drest, dtype=complex)
    for K in kraus_pair:
        Kbig = np.kron(K, Irest)
        out_p += Kbig @ rho_p @ Kbig.conj().T

    out_p = 0.5 * (out_p + out_p.conj().T)
    out_p /= np.trace(out_p)

    return permute_systems(out_p, dims_p, invp)


# --------------------------
# SU(3)
# --------------------------
def gell_mann() -> list[np.ndarray]:
    lam = []
    lam.append(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex))
    lam.append(np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex))
    lam.append(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex))
    lam.append(np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex))
    lam.append(np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex))
    lam.append(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex))
    lam.append(np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex))
    lam.append(np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / np.sqrt(3))
    return lam


LAM = gell_mann()
TGEN = [m / 2.0 for m in LAM]


def structure_constants(lams: list[np.ndarray]) -> np.ndarray:
    f = np.zeros((8, 8, 8), dtype=float)
    for a in range(8):
        for b in range(8):
            comm = lams[a] @ lams[b] - lams[b] @ lams[a]
            for c in range(8):
                val = np.trace(comm @ lams[c]) / (4j)
                f[a, b, c] = float(np.real_if_close(val))
    return f


FABC = structure_constants(LAM)


def adjoint_generators(fabc: np.ndarray) -> list[np.ndarray]:
    Fs = []
    for a in range(8):
        F = np.zeros((8, 8), dtype=complex)
        for b in range(8):
            for c in range(8):
                F[b, c] = -1j * fabc[a, b, c]
        Fs.append(F)
    return Fs


FGEN = adjoint_generators(FABC)


# --------------------------
# model parameters
# --------------------------
@dataclass
class ModelParams:
    tau0_fm: float = 0.6
    T0_MeV: float = 400.0
    Tf_MeV: float = 156.0
    dt_fm: float = 0.05

    NE: int = 5
    omega_MeV: float = 250.0

    kappaE0_fm_inv: float = 0.35
    expE: float = 3.0

    kappa_q0_fm_inv: float = 0.08
    kappa_qbar0_fm_inv: float = 0.08
    kappa_g0_fm_inv: float = 0.18
    expC: float = 3.0

    CA_over_CF: float = 9.0 / 4.0
    strange_bath_fraction: float = 0.12
    eta_octet_dressed: float = 0.25

    outdir: str = "outputs_qgp_channel_optionB_exactCPTP"


# --------------------------
# medium
# --------------------------
def T_of_tau(tau: float, mp: ModelParams) -> float:
    return mp.T0_MeV * (mp.tau0_fm / tau) ** (1.0 / 3.0)


def tau_freezeout(mp: ModelParams) -> float:
    return mp.tau0_fm * (mp.T0_MeV / mp.Tf_MeV) ** 3


def kappaE(T: float, mp: ModelParams) -> float:
    return mp.kappaE0_fm_inv * (T / mp.T0_MeV) ** mp.expE


def kappa_q(T: float, mp: ModelParams) -> float:
    return mp.kappa_q0_fm_inv * (T / mp.T0_MeV) ** mp.expC


def kappa_qbar(T: float, mp: ModelParams) -> float:
    return mp.kappa_qbar0_fm_inv * (T / mp.T0_MeV) ** mp.expC


def kappa_g(T: float, mp: ModelParams) -> float:
    return mp.kappa_g0_fm_inv * (T / mp.T0_MeV) ** mp.expC


def gamma_down_up(T: float, mp: ModelParams) -> tuple[float, float]:
    gd = kappaE(T, mp)
    gu = gd * np.exp(-mp.omega_MeV / max(T, 1e-12))
    return float(gd), float(gu)


# --------------------------
# color states
# --------------------------
def singlet_AB() -> np.ndarray:
    psi = np.zeros((9,), dtype=complex)
    psi[0] = 1 / np.sqrt(3)
    psi[4] = 1 / np.sqrt(3)
    psi[8] = 1 / np.sqrt(3)
    return psi


def octet_AB_states() -> list[np.ndarray]:
    states = []
    for Ta in TGEN:
        vec = np.zeros((9,), dtype=complex)
        for i in range(3):
            for j in range(3):
                vec[3 * i + j] = np.sqrt(2.0) * Ta[i, j]
        vec /= np.linalg.norm(vec)
        states.append(vec)
    return states


OCTET_STATES = octet_AB_states()


def octet_dressed_total_singlet() -> np.ndarray:
    psi = np.zeros((9 * 8,), dtype=complex)
    for a in range(8):
        psi += np.kron(OCTET_STATES[a], basis(8, a).ravel()) / np.sqrt(8.0)
    psi /= np.linalg.norm(psi)
    return psi


def initial_color_rho(mp: ModelParams) -> np.ndarray:
    psi_s = singlet_AB()
    rho_pair_sing = np.kron(density_from_state(psi_s), np.eye(8, dtype=complex) / 8.0)
    psi_oct = octet_dressed_total_singlet()
    rho_oct = density_from_state(psi_oct)
    eta = float(np.clip(mp.eta_octet_dressed, 0.0, 1.0))
    rho = (1 - eta) * rho_pair_sing + eta * rho_oct
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)
    return rho


# --------------------------
# exact local color channels
# --------------------------
def base_color_dissipator_fundamental() -> np.ndarray:
    d = 3
    H = np.zeros((d, d), dtype=complex)
    Ls = TGEN[:]  # kappa factored outside
    return lindblad_superop(H, Ls)


def base_color_dissipator_conjugate() -> np.ndarray:
    d = 3
    H = np.zeros((d, d), dtype=complex)
    Ls = [-(T.conj()) for T in TGEN]  # anti-fundamental generators
    return lindblad_superop(H, Ls)


def base_color_dissipator_adjoint() -> np.ndarray:
    d = 8
    H = np.zeros((d, d), dtype=complex)
    Ls = FGEN[:]
    return lindblad_superop(H, Ls)


COLOR_SUP_A = base_color_dissipator_fundamental()
COLOR_SUP_B = base_color_dissipator_conjugate()
COLOR_SUP_G = base_color_dissipator_adjoint()


def local_channel_from_superop(base_superop: np.ndarray, d: int, rate: float, dt: float) -> list[np.ndarray]:
    Phi = expm(dt * rate * base_superop)
    J = superop_to_choi(Phi, d)
    return choi_to_kraus(J, d)


def evolve_color_scan_exact(mp_base: ModelParams, kappa_q0_scan: tuple[float, ...], tie_qbar=True, tie_g_casimir=True):
    tau_f = tau_freezeout(mp_base)
    taus = np.arange(mp_base.tau0_fm, tau_f + 1e-12, mp_base.dt_fm)
    results = {}

    dims = (3, 3, 8)

    for k0 in kappa_q0_scan:
        mp = ModelParams(**asdict(mp_base))
        mp.kappa_q0_fm_inv = float(k0)

        if tie_qbar:
            mp.kappa_qbar0_fm_inv = float(k0)
        if tie_g_casimir:
            mp.kappa_g0_fm_inv = float(mp.CA_over_CF * k0)

        rho = initial_color_rho(mp)

        IAB, IAG, IBG, EN, SA, PA = [], [], [], [], [], []

        for tau in taus:
            rhoAB = partial_trace(rho, dims, (0, 1))
            rhoA = partial_trace(rho, dims, (0,))
            IAB.append(mutual_information(rho, dims, (0,), (1,)))
            IAG.append(mutual_information(rho, dims, (0,), (2,)))
            IBG.append(mutual_information(rho, dims, (1,), (2,)))
            EN.append(log_negativity(rhoAB, 3, 3))
            SA.append(von_neumann_entropy(rhoA))
            PA.append(purity(rhoA))

            T = T_of_tau(tau, mp)
            KA = local_channel_from_superop(COLOR_SUP_A, 3, kappa_q(T, mp), mp.dt_fm)
            KB = local_channel_from_superop(COLOR_SUP_B, 3, kappa_qbar(T, mp), mp.dt_fm)
            KG = local_channel_from_superop(COLOR_SUP_G, 8, kappa_g(T, mp), mp.dt_fm)

            rho = apply_local_channel(rho, dims, 0, KA)
            rho = apply_local_channel(rho, dims, 1, KB)
            rho = apply_local_channel(rho, dims, 2, KG)

        results[float(k0)] = {
            "taus": taus,
            "IAB": np.array(IAB),
            "IAG": np.array(IAG),
            "IBG": np.array(IBG),
            "EN": np.array(EN),
            "SA": np.array(SA),
            "PA": np.array(PA),
            "rho_final": rho,
            "params": mp,
        }

    return results


# --------------------------
# exact local energy channels
# --------------------------
def ladder_ops(NE: int):
    a = np.zeros((NE, NE), dtype=complex)
    for n in range(1, NE):
        a[n - 1, n] = np.sqrt(n)
    adag = a.conj().T
    n_op = np.diag(np.arange(NE)).astype(complex)
    return a, adag, n_op


def base_energy_superops(NE: int) -> tuple[np.ndarray, np.ndarray]:
    a, adag, _ = ladder_ops(NE)
    H = np.zeros((NE, NE), dtype=complex)
    Sdown = lindblad_superop(H, [a])
    Sup = lindblad_superop(H, [adag])
    return Sdown, Sup


def initial_energy_rho(NE: int) -> np.ndarray:
    """
    Pure entangled energy initial state:
        (|0,0> + |NE-1,NE-1>) / sqrt(2)
    """
    psi = np.zeros((NE * NE,), dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[(NE - 1) * NE + (NE - 1)] = 1 / np.sqrt(2)
    return density_from_state(psi)


def evolve_energy_exact(mp: ModelParams):
    NE = mp.NE
    tau_f = tau_freezeout(mp)
    taus = np.arange(mp.tau0_fm, tau_f + 1e-12, mp.dt_fm)

    Sdown, Sup = base_energy_superops(NE)
    dims = (NE, NE)

    _, _, n_op = ladder_ops(NE)
    nA = np.kron(n_op, np.eye(NE, dtype=complex))

    rho = initial_energy_rho(NE)

    Emean, PurA = [], []

    for tau in taus:
        rhoA = partial_trace(rho, dims, (0,))
        Emean.append(float(np.real(np.trace(rho @ nA))))
        PurA.append(purity(rhoA))

        T = T_of_tau(tau, mp)
        gd, gu = gamma_down_up(T, mp)

        KA = exact_local_channel_kraus(np.zeros((NE, NE), dtype=complex), [np.sqrt(gd) * ladder_ops(NE)[0], np.sqrt(gu) * ladder_ops(NE)[1]], mp.dt_fm)
        KB = KA  # same rates for A and B in this minimal setup

        rho = apply_local_channel(rho, dims, 0, KA)
        rho = apply_local_channel(rho, dims, 1, KB)

    return taus, np.array(Emean), np.array(PurA)


# --------------------------
# explicit freeze-out color map
# --------------------------
def singlet_projector_AB() -> np.ndarray:
    return density_from_state(singlet_AB())


PI_SING = singlet_projector_AB()


def pair_singlet_reset_kraus() -> list[np.ndarray]:
    """
    Kraus operators for E(rho)=P rho P + Tr[(I-P)rho] P
    on a 9-dim q qbar pair space, with P=|sing><sing|.
    """
    P = PI_SING
    I = np.eye(9, dtype=complex)
    Q = I - P

    vals, vecs = np.linalg.eigh(Q)
    comp = [vecs[:, i] for i, lam in enumerate(vals) if lam > 1e-12]
    psi_s = singlet_AB().reshape((-1, 1))

    kraus = [P.copy()]
    for v in comp:
        v = v.reshape((-1, 1))
        kraus.append(psi_s @ v.conj().T)
    return kraus


PAIR_RESET_KRAUS = pair_singlet_reset_kraus()


def fragmentation_color_state(rhoAB: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rhoAB)
    for K in PAIR_RESET_KRAUS:
        out += K @ rhoAB @ K.conj().T
    out = 0.5 * (out + out.conj().T)
    out /= np.trace(out)
    return out


def recombination_color_state_explicit(rhoAB: np.ndarray) -> np.ndarray:
    """
    Explicit ancilla-based recombination color channel:
      - order of subsystems: A, B, abar_bath, q_bath
      - apply singlet reset on (A, abar_bath)
      - apply singlet reset on (q_bath, B)
      - trace out bath ancillas
    """
    rho_abath = np.eye(3, dtype=complex) / 3.0
    rho_qbath = np.eye(3, dtype=complex) / 3.0

    dims = (3, 3, 3, 3)  # A, B, abar_bath, q_bath
    rho_full = kronN(rhoAB, rho_abath, rho_qbath)

    # A with abar_bath -> targets (0,2)
    rho_full = apply_two_body_channel(rho_full, dims, (0, 2), PAIR_RESET_KRAUS)

    # q_bath with B -> pair order in local channel should be (q, B)
    # Our pair singlet projector is symmetric in the minimal color singlet basis,
    # so the same pair-reset Kraus works.
    rho_full = apply_two_body_channel(rho_full, dims, (3, 1), PAIR_RESET_KRAUS)

    # Trace out bath ancillas (2,3), keep A,B
    rho_post = partial_trace(rho_full, dims, (0, 1))
    rho_post = 0.5 * (rho_post + rho_post.conj().T)
    rho_post /= np.trace(rho_post)
    return rho_post


def post_freezeout_color_state(rhoABG_pre: np.ndarray, xi: float) -> np.ndarray:
    xi = float(np.clip(xi, 0.0, 1.0))
    rhoAB_pre = partial_trace(rhoABG_pre, (3, 3, 8), (0, 1))
    rho_frag = fragmentation_color_state(rhoAB_pre)
    rho_rec = recombination_color_state_explicit(rhoAB_pre)
    rho_post = (1 - xi) * rho_frag + xi * rho_rec
    rho_post = 0.5 * (rho_post + rho_post.conj().T)
    rho_post /= np.trace(rho_post)
    return rho_post


# --------------------------
# minimal explicit flavor / hadron sector
# --------------------------
FLAVORS = {"u": 0, "d": 1, "s": 2}


def initial_flavor_rho_uubar() -> np.ndarray:
    psi = np.zeros((9,), dtype=complex)
    psi[3 * FLAVORS["u"] + FLAVORS["u"]] = 1.0
    return density_from_state(psi)


def flavor_bath_distribution(ps: float) -> np.ndarray:
    ps = float(np.clip(ps, 0.0, 0.49))
    pu = pd = (1.0 - ps) / 2.0
    return np.array([pu, pd, ps], dtype=float)


def is_light(i: int) -> bool:
    return i in (FLAVORS["u"], FLAVORS["d"])


def species_projectors_pair_flavor() -> tuple[np.ndarray, np.ndarray]:
    """
    Minimal two-species flavor projectors on 3 x 3bar:
      - pion-like: light-light pairs
      - kaon-like: one light + one strange
    """
    P_pi = np.zeros((9, 9), dtype=complex)
    P_K = np.zeros((9, 9), dtype=complex)

    for q in range(3):
        for aq in range(3):
            idx = 3 * q + aq
            e = np.zeros((9, 1), dtype=complex)
            e[idx, 0] = 1.0

            if is_light(q) and is_light(aq):
                P_pi += e @ e.conj().T
            elif (is_light(q) and aq == FLAVORS["s"]) or (q == FLAVORS["s"] and is_light(aq)):
                P_K += e @ e.conj().T

    return P_pi, P_K


P_PI_FLAVOR, P_K_FLAVOR = species_projectors_pair_flavor()


def boltzmann_weights_piK(Tf: float) -> np.ndarray:
    m_pi = 139.57
    m_K = 493.68
    w = np.array([np.exp(-m_pi / max(Tf, 1e-12)), np.exp(-m_K / max(Tf, 1e-12))], dtype=float)
    return w / w.sum()


def species_probs_from_pair_flavor_state(rho_pair: np.ndarray, Tf: float) -> np.ndarray:
    """
    Compatibility with flavor + Boltzmann species weighting.
    """
    compat = np.array([
        float(np.real(np.trace(P_PI_FLAVOR @ rho_pair))),
        float(np.real(np.trace(P_K_FLAVOR @ rho_pair))),
    ], dtype=float)

    weights = boltzmann_weights_piK(Tf)
    raw = compat * weights
    if raw.sum() <= 1e-15:
        raw = np.array([1.0, 0.0], dtype=float)
    return raw / raw.sum()


def fragmentation_hadron_register(rhoAB_flavor: np.ndarray, Tf: float) -> tuple[np.ndarray, dict]:
    probs = species_probs_from_pair_flavor_state(rhoAB_flavor, Tf)
    rho_h = np.diag(probs).astype(complex)
    return rho_h, {
        "p_pi": float(probs[0]),
        "p_K": float(probs[1]),
        "K_over_pi": float(probs[1] / max(probs[0], 1e-14)),
    }


def recombination_hadron_register(rhoAB_flavor: np.ndarray, Tf: float, mp: ModelParams) -> tuple[np.ndarray, dict]:
    """
    Minimal explicit flavor-bath recombination:
      - compute flavor marginal of A and B
      - pair A with antiq bath; pair q bath with B
      - compute the hadron species distribution for each branch
      - average A-side and B-side tagged-hadron marginals
    """
    rhoA = partial_trace(rhoAB_flavor, (3, 3), (0,))
    rhoB = partial_trace(rhoAB_flavor, (3, 3), (1,))

    pbath = flavor_bath_distribution(mp.strange_bath_fraction)
    rho_antiq_bath = np.diag(pbath).astype(complex)
    rho_q_bath = np.diag(pbath).astype(complex)

    rho_Aab = np.kron(rhoA, rho_antiq_bath)   # A with antiq bath
    rho_qB = np.kron(rho_q_bath, rhoB)        # q bath with B

    pA = species_probs_from_pair_flavor_state(rho_Aab, Tf)
    pB = species_probs_from_pair_flavor_state(rho_qB, Tf)

    probs = 0.5 * (pA + pB)
    probs = probs / probs.sum()

    rho_h = np.diag(probs).astype(complex)
    return rho_h, {
        "p_pi": float(probs[0]),
        "p_K": float(probs[1]),
        "K_over_pi": float(probs[1] / max(probs[0], 1e-14)),
    }


def hadron_register_from_xi(xi: float, Tf: float, mp: ModelParams) -> tuple[np.ndarray, dict]:
    rhoABf = initial_flavor_rho_uubar()
    rho_h_frag, info_frag = fragmentation_hadron_register(rhoABf, Tf)
    rho_h_rec, info_rec = recombination_hadron_register(rhoABf, Tf, mp)

    xi = float(np.clip(xi, 0.0, 1.0))
    rho_h = (1 - xi) * rho_h_frag + xi * rho_h_rec
    rho_h = 0.5 * (rho_h + rho_h.conj().T)
    rho_h /= np.trace(rho_h)

    probs = np.real(np.diag(rho_h))
    return rho_h, {
        "p_pi": float(probs[0]),
        "p_K": float(probs[1]),
        "K_over_pi": float(probs[1] / max(probs[0], 1e-14)),
        "frag_info": info_frag,
        "rec_info": info_rec,
    }


# --------------------------
# plotting helpers
# --------------------------
def add_tau_freezeout_line(ax, tau_f):
    ax.axvline(tau_f, linestyle="--", linewidth=1.0, alpha=0.9)


def plot_scan_curves(ax, x, ys, labels):
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, (y, lab) in enumerate(zip(ys, labels)):
        ax.plot(
            x, y,
            label=lab,
            marker=markers[i % len(markers)],
            markevery=max(1, len(x) // 8),
            linewidth=1.8,
        )


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

    fig_singlecol(2.45)
    ax = plt.gca()
    plot_scan_curves(
        ax,
        res[kscan[0]]["taus"],
        [res[k]["EN"] for k in kscan],
        [fr"$\kappa_q(T_0)={k:.2f}$" for k in kscan],
    )
    add_tau_freezeout_line(ax, tau_f)
    ax.set_xlabel(r"$\tau\,[\mathrm{fm}/c]$")
    ax.set_ylabel(r"$E_{\mathcal{N}}^{(c)}(A{:}B)$")
    ax.legend(loc="upper right")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_logneg_color_vs_tau_kappaq_scan"))

    # Representative AB / AG / BG
    kref = kscan[len(kscan) // 2]
    fig_singlecol(2.50)
    ax = plt.gca()
    ax.plot(res[kref]["taus"], res[kref]["IAB"], label=r"$I(A{:}B)$", marker="o",
            markevery=max(1, len(res[kref]["taus"]) // 8))
    ax.plot(res[kref]["taus"], res[kref]["IAG"], label=r"$I(A{:}G)$", marker="s",
            markevery=max(1, len(res[kref]["taus"]) // 8))
    ax.plot(res[kref]["taus"], res[kref]["IBG"], label=r"$I(B{:}G)$", marker="^",
            markevery=max(1, len(res[kref]["taus"]) // 8))
    add_tau_freezeout_line(ax, tau_f)
    ax.set_xlabel(r"$\tau\,[\mathrm{fm}/c]$")
    ax.set_ylabel(r"$I(\cdot{:}\cdot)$")
    ax.legend(loc="upper right")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_MI_AB_AG_BG_vs_tau"))

    # 3) Exact energy sector
    tausE, Emean, PurA = evolve_energy_exact(mp)

    fig_singlecol(2.35)
    ax = plt.gca()
    ax.plot(tausE, Emean, marker="o", markevery=max(1, len(tausE) // 8))
    add_tau_freezeout_line(ax, tau_f)
    ax.set_xlabel(r"$\tau\,[\mathrm{fm}/c]$")
    ax.set_ylabel(r"$\langle n\rangle_A$")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_energy_mean_level_vs_tau"))

    fig_singlecol(2.35)
    ax = plt.gca()
    ax.plot(tausE, PurA, marker="s", markevery=max(1, len(tausE) // 8))
    add_tau_freezeout_line(ax, tau_f)
    ax.set_xlabel(r"$\tau\,[\mathrm{fm}/c]$")
    ax.set_ylabel(r"$\mathrm{Tr}(\rho_{A,E}^2)$")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_energy_purity_vs_tau"))

    # 4) Freeze-out xi scan, using explicit ancilla-based color recombination
    xi_scan = np.linspace(0, 1, 21)
    rho_final = res[kref]["rho_final"]
    MIpost, ENpost = [], []

    for xi in xi_scan:
        rhoAB_post = post_freezeout_color_state(rho_final, float(xi))
        MIpost.append(mutual_information(rhoAB_post, (3, 3), (0,), (1,)))
        ENpost.append(log_negativity(rhoAB_post, 3, 3))

    fig_singlecol(2.35)
    ax = plt.gca()
    ax.plot(xi_scan, MIpost, marker="o", markevery=2)
    ax.set_xlabel(r"recombination fraction $\xi$")
    ax.set_ylabel(r"$I_c(A{:}B)$ post FO")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_freezeout_MI_color_vs_xi"))

    fig_singlecol(2.35)
    ax = plt.gca()
    ax.plot(xi_scan, ENpost, marker="s", markevery=2)
    ax.set_xlabel(r"recombination fraction $\xi$")
    ax.set_ylabel(r"$E_{\mathcal{N}}^{(c)}$ post FO")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_freezeout_logneg_color_vs_xi"))

    # 5) K/pi vs Tf from minimal explicit flavor/hadron sector
    Tf_grid = np.linspace(120, 200, 81)
    xi_choices = (0.0, 0.5, 1.0)

    fig_singlecol(2.40)
    ax = plt.gca()
    markers = ["o", "s", "^"]
    for i, xi in enumerate(xi_choices):
        ratios = []
        for Tf in Tf_grid:
            _, info = hadron_register_from_xi(float(xi), float(Tf), mp)
            ratios.append(info["K_over_pi"])
        ax.plot(
            Tf_grid, ratios,
            label=fr"$\xi={xi:.1f}$",
            marker=markers[i],
            markevery=max(1, len(Tf_grid) // 8),
            linewidth=1.8,
        )
    ax.axvline(mp.Tf_MeV, linestyle="--", linewidth=1.0, label=r"$T_f$")
    ax.set_xlabel(r"$T_f\,[\mathrm{MeV}]$")
    ax.set_ylabel(r"$K/\pi$")
    ax.legend(loc="upper left")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_Kpi_vs_Tf_xi_choices"))

    # Save metadata
    with open(os.path.join(mp.outdir, "run_meta.json"), "w") as f:
        json.dump({
            "ModelParams": asdict(mp),
            "kappa_q0_scan": list(kscan),
            "freezeout_time_fm": float(tau_f),
            "notes": [
                "This version uses exact local GKSL channels via matrix exponentials.",
                "Color sector evolved exactly as local CPTP channels on 3 x 3bar x 8.",
                "Energy sector evolved exactly as local CPTP channels on NE x NE.",
                "Freeze-out color recombination implemented with explicit bath color ancillas and singlet-reset channels.",
                "Hadron sector is minimal explicit 2-level {|pi>,|K>} register used for the K/pi diagnostic.",
                "The full 10800-dimensional dense exact evolution is not attempted; this is the exact sectorized implementation relevant to the plotted observables."
            ]
        }, f, indent=2)

    print(f"[OK] Exact-CPTP theory-aligned plots saved in: {mp.outdir}")


if __name__ == "__main__":
    run_all()
