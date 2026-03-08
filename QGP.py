#!/usr/bin/env python3
from __future__ import annotations
import os, json
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ============================================================
# Paper-ready Option-B theory-aligned plotting code
# ============================================================

# ---------- plotting ----------
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

# ---------- linear algebra ----------
def basis(d, i):
    v = np.zeros((d, 1), dtype=complex)
    v[i, 0] = 1.0
    return v

def density_from_state(psi):
    psi = psi.reshape((-1, 1))
    return psi @ psi.conj().T

def kronN(*ops):
    out = np.array([[1.0]], dtype=complex)
    for op in ops:
        out = np.kron(out, op)
    return out

def partial_trace(rho, dims, keep):
    dims = tuple(dims)
    keep = tuple(keep)
    n = len(dims)
    R = rho.reshape(dims + dims)
    trace_out = [k for k in range(n) if k not in keep]
    current_n = n
    for k in sorted(trace_out, reverse=True):
        R = np.trace(R, axis1=k, axis2=k + current_n)
        current_n -= 1
    kept_dims = [dims[k] for k in keep]
    return R.reshape(int(np.prod(kept_dims)), int(np.prod(kept_dims)))

def von_neumann_entropy(rho, tol=1e-12):
    rhoH = 0.5 * (rho + rho.conj().T)
    evals = np.linalg.eigvalsh(rhoH)
    evals = np.real(evals)
    evals = evals[evals > tol]
    if evals.size == 0:
        return 0.0
    return float(-np.sum(evals * np.log2(evals)))

def mutual_information(rho, dims, keepA, keepB):
    keepAB = tuple(sorted(set(keepA + keepB)))
    rhoAB = partial_trace(rho, dims, keepAB)
    rhoA = partial_trace(rho, dims, keepA)
    rhoB = partial_trace(rho, dims, keepB)
    return von_neumann_entropy(rhoA) + von_neumann_entropy(rhoB) - von_neumann_entropy(rhoAB)

def partial_transpose_two_party(rhoAB, dA, dB, sys=1):
    arr = rhoAB.reshape((dA, dB, dA, dB))
    if sys == 1:
        arr = arr.transpose(0, 3, 2, 1)
    else:
        arr = arr.transpose(2, 1, 0, 3)
    return arr.reshape((dA * dB, dA * dB))

def log_negativity(rhoAB, dA, dB):
    rhoPT = partial_transpose_two_party(rhoAB, dA, dB, sys=1)
    svals = np.linalg.svd(rhoPT, compute_uv=False)
    return max(0.0, float(np.log2(np.sum(svals))))

def purity(rho):
    return float(np.real(np.trace(rho @ rho)))

def kraus_step_first_order(rho, Ls, dt):
    d = rho.shape[0]
    I = np.eye(d, dtype=complex)
    Q = np.zeros((d, d), dtype=complex)
    for L in Ls:
        Q += L.conj().T @ L
    K0 = I - 0.5 * dt * Q
    out = K0 @ rho @ K0.conj().T
    for L in Ls:
        out += dt * (L @ rho @ L.conj().T)
    out = 0.5 * (out + out.conj().T)
    tr = np.trace(out)
    if abs(tr) > 0:
        out /= tr
    return out

# ---------- SU(3) ----------
def gell_mann():
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

def structure_constants(lams):
    f = np.zeros((8, 8, 8), dtype=float)
    for a in range(8):
        for b in range(8):
            comm = lams[a] @ lams[b] - lams[b] @ lams[a]
            for c in range(8):
                val = np.trace(comm @ lams[c]) / (4j)
                f[a, b, c] = float(np.real_if_close(val))
    return f

FABC = structure_constants(LAM)

def adjoint_generators(fabc):
    Fs = []
    for a in range(8):
        F = np.zeros((8, 8), dtype=complex)
        for b in range(8):
            for c in range(8):
                F[b, c] = -1j * fabc[a, b, c]
        Fs.append(F)
    return Fs

FGEN = adjoint_generators(FABC)

# ---------- model parameters ----------
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

    # initial color state: mix pair-singlet x maximally-mixed gluon
    # with octet-dressed total singlet
    eta_octet_dressed: float = 0.25

    outdir: str = "outputs_qgp_channel_optionB_theory_aligned_pretty"

# ---------- medium ----------
def T_of_tau(tau, mp):
    return mp.T0_MeV * (mp.tau0_fm / tau) ** (1.0 / 3.0)

def tau_freezeout(mp):
    return mp.tau0_fm * (mp.T0_MeV / mp.Tf_MeV) ** 3

def kappaE(T, mp):
    return mp.kappaE0_fm_inv * (T / mp.T0_MeV) ** mp.expE

def kappa_q(T, mp):
    return mp.kappa_q0_fm_inv * (T / mp.T0_MeV) ** mp.expC

def kappa_qbar(T, mp):
    return mp.kappa_qbar0_fm_inv * (T / mp.T0_MeV) ** mp.expC

def kappa_g(T, mp):
    return mp.kappa_g0_fm_inv * (T / mp.T0_MeV) ** mp.expC

def gamma_down_up(T, mp):
    gd = kappaE(T, mp)
    gu = gd * np.exp(-mp.omega_MeV / max(T, 1e-12))
    return float(gd), float(gu)

# ---------- color states ----------
def singlet_AB():
    psi = np.zeros((9,), dtype=complex)
    psi[0] = 1 / np.sqrt(3)
    psi[4] = 1 / np.sqrt(3)
    psi[8] = 1 / np.sqrt(3)
    return psi

def octet_AB_states():
    states = []
    for Ta in TGEN:
        vec = np.zeros((9,), dtype=complex)
        for i in range(3):
            for j in range(3):
                vec[3 * i + j] = np.sqrt(2.0) * Ta[i, j]
        nrm = np.linalg.norm(vec)
        states.append(vec / nrm)
    return states

OCTET_STATES = octet_AB_states()

def octet_dressed_total_singlet():
    psi = np.zeros((9 * 8,), dtype=complex)
    for a in range(8):
        psi += np.kron(OCTET_STATES[a], basis(8, a).ravel()) / np.sqrt(8.0)
    psi /= np.linalg.norm(psi)
    return psi

def initial_color_rho(mp):
    psi_s = singlet_AB()
    rho_pair_sing = np.kron(density_from_state(psi_s), np.eye(8, dtype=complex) / 8.0)
    psi_oct = octet_dressed_total_singlet()
    rho_oct = density_from_state(psi_oct)
    eta = float(np.clip(mp.eta_octet_dressed, 0.0, 1.0))
    rho = (1 - eta) * rho_pair_sing + eta * rho_oct
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)
    return rho

def build_color_lindblad_ops(mp, T):
    I3 = np.eye(3, dtype=complex)
    I8 = np.eye(8, dtype=complex)
    Ls = []
    for a in range(8):
        TA = kronN(TGEN[a], I3, I8)
        TB = kronN(I3, -TGEN[a].conj(), I8)
        FG = kronN(I3, I3, FGEN[a])
        Ls.append(np.sqrt(kappa_q(T, mp)) * TA)
        Ls.append(np.sqrt(kappa_qbar(T, mp)) * TB)
        Ls.append(np.sqrt(kappa_g(T, mp)) * FG)
    return Ls

def color_observables(rhoABG):
    dims = (3, 3, 8)
    rhoAB = partial_trace(rhoABG, dims, (0, 1))
    rhoA = partial_trace(rhoABG, dims, (0,))
    return {
        "rhoAB": rhoAB,
        "I_AB": mutual_information(rhoABG, dims, (0,), (1,)),
        "I_AG": mutual_information(rhoABG, dims, (0,), (2,)),
        "I_BG": mutual_information(rhoABG, dims, (1,), (2,)),
        "E_AB": log_negativity(rhoAB, 3, 3),
        "S_A": von_neumann_entropy(rhoA),
        "P_A": purity(rhoA),
    }

# ---------- energy sector ----------
def ladder_ops(NE):
    a = np.zeros((NE, NE), dtype=complex)
    for n in range(1, NE):
        a[n - 1, n] = np.sqrt(n)
    adag = a.conj().T
    n_op = np.diag(np.arange(NE)).astype(complex)
    return a, adag, n_op

def initial_energy_rho(NE):
    psi = np.zeros((NE * NE,), dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[(NE - 1) * NE + (NE - 1)] = 1 / np.sqrt(2)
    return density_from_state(psi)

def evolve_energy(mp):
    NE = mp.NE
    tau_f = tau_freezeout(mp)
    taus = np.arange(mp.tau0_fm, tau_f + 1e-12, mp.dt_fm)

    a, adag, n_op = ladder_ops(NE)
    I = np.eye(NE, dtype=complex)
    LmA, LmB = np.kron(a, I), np.kron(I, a)
    LpA, LpB = np.kron(adag, I), np.kron(I, adag)
    nA = np.kron(n_op, I)

    rhoE = initial_energy_rho(NE)
    Emean, PurA = [], []

    for tau in taus:
        rhoA = partial_trace(rhoE, (NE, NE), (0,))
        Emean.append(float(np.real(np.trace(rhoE @ nA))))
        PurA.append(purity(rhoA))

        T = T_of_tau(tau, mp)
        gd, gu = gamma_down_up(T, mp)
        Ls = [np.sqrt(gd) * LmA, np.sqrt(gd) * LmB, np.sqrt(gu) * LpA, np.sqrt(gu) * LpB]
        rhoE = kraus_step_first_order(rhoE, Ls, mp.dt_fm)

    return taus, np.array(Emean), np.array(PurA)

# ---------- freeze-out instrument ----------
def singlet_projector_AB():
    psi = singlet_AB()
    return density_from_state(psi)

PI_SING = singlet_projector_AB()

def singlet_reset_channel(rhoAB):
    I9 = np.eye(9, dtype=complex)
    p_out = np.real(np.trace((I9 - PI_SING) @ rhoAB))
    out = PI_SING @ rhoAB @ PI_SING + p_out * PI_SING
    out = 0.5 * (out + out.conj().T)
    out /= np.trace(out)
    return out

def effective_recombination_AB_state():
    return np.eye(9, dtype=complex) / 9.0

def post_freezeout_color_state(rhoABG_pre, xi):
    rhoAB_pre = partial_trace(rhoABG_pre, (3, 3, 8), (0, 1))
    rho_frag = singlet_reset_channel(rhoAB_pre)
    rho_rec = effective_recombination_AB_state()
    rho_post = (1 - xi) * rho_frag + xi * rho_rec
    rho_post /= np.trace(rho_post)
    return rho_post

def flavor_bath_distribution(ps):
    ps = float(np.clip(ps, 0.0, 0.49))
    pu = pd = (1 - ps) / 2.0
    return np.array([pu, pd, ps], dtype=float)

def hadron_register_from_xi(xi, Tf, mp):
    frag_raw = np.array([np.exp(-139.57 / max(Tf, 1e-12)), 0.0], dtype=float)

    bath = flavor_bath_distribution(mp.strange_bath_fraction)
    light = bath[0] + bath[1]
    strange = bath[2]

    rec_raw = np.array([
        light * np.exp(-139.57 / max(Tf, 1e-12)),
        strange * np.exp(-493.68 / max(Tf, 1e-12))
    ], dtype=float)

    raw = (1 - xi) * frag_raw + xi * rec_raw
    if raw.sum() <= 0:
        probs = np.array([1.0, 0.0], dtype=float)
    else:
        probs = raw / raw.sum()

    rho_h = np.diag(probs).astype(complex)
    return rho_h, {
        "p_pi": float(probs[0]),
        "p_K": float(probs[1]),
        "K_over_pi": float(probs[1] / max(probs[0], 1e-14))
    }

# ---------- runs ----------
def evolve_color_scan(mp_base, kappa_q0_scan, tie_qbar=True, tie_g_casimir=True):
    tau_f = tau_freezeout(mp_base)
    taus = np.arange(mp_base.tau0_fm, tau_f + 1e-12, mp_base.dt_fm)
    results = {}

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
            obs = color_observables(rho)
            IAB.append(obs["I_AB"])
            IAG.append(obs["I_AG"])
            IBG.append(obs["I_BG"])
            EN.append(obs["E_AB"])
            SA.append(obs["S_A"])
            PA.append(obs["P_A"])

            T = T_of_tau(tau, mp)
            rho = kraus_step_first_order(rho, build_color_lindblad_ops(mp, T), mp.dt_fm)

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

# ---------- helper for pretty tau-plots ----------
def add_tau_freezeout_line(ax, tau_f):
    ax.axvline(tau_f, linestyle="--", linewidth=1.0, alpha=0.9)

def plot_scan_curves(ax, x, ys, labels):
    markers = ['o', 's', '^', 'D', 'v', 'P']
    for i, (y, lab) in enumerate(zip(ys, labels)):
        ax.plot(
            x, y,
            label=lab,
            marker=markers[i % len(markers)],
            markevery=max(1, len(x)//8),
            linewidth=1.8
        )

# ---------- plotting / main ----------
def run_all():
    set_style()
    mp = ModelParams()
    ensure_dir(mp.outdir)

    tau_f = tau_freezeout(mp)
    taus = np.arange(mp.tau0_fm, tau_f + 1e-12, mp.dt_fm)
    Ts = np.array([T_of_tau(t, mp) for t in taus])

    # Temperature profile
    fig_singlecol(2.35)
    ax = plt.gca()
    ax.plot(taus, Ts, marker='o', markevery=max(1, len(taus)//8), linewidth=1.8)
    ax.axhline(mp.Tf_MeV, linestyle="--", linewidth=1.0, label=r"$T_f$")
    add_tau_freezeout_line(ax, tau_f)
    ax.set_xlabel(r"$\tau\,[\mathrm{fm}/c]$")
    ax.set_ylabel(r"$T(\tau)\,[\mathrm{MeV}]$")
    ax.legend(loc="upper right")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_T_profile"))

    # Color scan
    kscan = (0.03, 0.06, 0.10, 0.16)
    res = evolve_color_scan(mp, kscan)

    # MI color
    fig_singlecol(2.45)
    ax = plt.gca()
    plot_scan_curves(
        ax,
        res[kscan[0]]["taus"],
        [res[k]["IAB"] for k in kscan],
        [fr"$\kappa_q(T_0)={k:.2f}$" for k in kscan]
    )
    add_tau_freezeout_line(ax, tau_f)
    ax.set_xlabel(r"$\tau\,[\mathrm{fm}/c]$")
    ax.set_ylabel(r"$I_c(A{:}B)$")
    ax.legend(loc="upper right")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_MI_color_vs_tau_kappaq_scan"))

    # Log-negativity color
    fig_singlecol(2.45)
    ax = plt.gca()
    plot_scan_curves(
        ax,
        res[kscan[0]]["taus"],
        [res[k]["EN"] for k in kscan],
        [fr"$\kappa_q(T_0)={k:.2f}$" for k in kscan]
    )
    add_tau_freezeout_line(ax, tau_f)
    ax.set_xlabel(r"$\tau\,[\mathrm{fm}/c]$")
    ax.set_ylabel(r"$E_{\mathcal{N}}^{(c)}(A{:}B)$")
    ax.legend(loc="upper right")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_logneg_color_vs_tau_kappaq_scan"))

    # Representative AB/AG/BG
    kref = kscan[len(kscan) // 2]
    fig_singlecol(2.50)
    ax = plt.gca()
    ax.plot(res[kref]["taus"], res[kref]["IAB"], label=r"$I(A{:}B)$", marker='o',
            markevery=max(1, len(res[kref]["taus"])//8))
    ax.plot(res[kref]["taus"], res[kref]["IAG"], label=r"$I(A{:}G)$", marker='s',
            markevery=max(1, len(res[kref]["taus"])//8))
    ax.plot(res[kref]["taus"], res[kref]["IBG"], label=r"$I(B{:}G)$", marker='^',
            markevery=max(1, len(res[kref]["taus"])//8))
    add_tau_freezeout_line(ax, tau_f)
    ax.set_xlabel(r"$\tau\,[\mathrm{fm}/c]$")
    ax.set_ylabel(r"$I(\cdot{:}\cdot)$")
    ax.legend(loc="upper right")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_MI_AB_AG_BG_vs_tau"))

    # Energy sector
    tausE, Emean, PurA = evolve_energy(mp)

    fig_singlecol(2.35)
    ax = plt.gca()
    ax.plot(tausE, Emean, marker='o', markevery=max(1, len(tausE)//8))
    add_tau_freezeout_line(ax, tau_f)
    ax.set_xlabel(r"$\tau\,[\mathrm{fm}/c]$")
    ax.set_ylabel(r"$\langle n\rangle_A$")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_energy_mean_level_vs_tau"))

    fig_singlecol(2.35)
    ax = plt.gca()
    ax.plot(tausE, PurA, marker='s', markevery=max(1, len(tausE)//8))
    add_tau_freezeout_line(ax, tau_f)
    ax.set_xlabel(r"$\tau\,[\mathrm{fm}/c]$")
    ax.set_ylabel(r"$\mathrm{Tr}(\rho_{A,E}^2)$")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_energy_purity_vs_tau"))

    # Freeze-out xi scan
    xi_scan = np.linspace(0, 1, 21)
    rho_final = res[kref]["rho_final"]
    MIpost, ENpost = [], []

    for xi in xi_scan:
        rhoAB_post = post_freezeout_color_state(rho_final, float(xi))
        MIpost.append(mutual_information(rhoAB_post, (3, 3), (0,), (1,)))
        ENpost.append(log_negativity(rhoAB_post, 3, 3))

    fig_singlecol(2.35)
    ax = plt.gca()
    ax.plot(xi_scan, MIpost, marker='o', markevery=2)
    ax.set_xlabel(r"recombination fraction $\xi$")
    ax.set_ylabel(r"$I_c(A{:}B)$ post FO")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_freezeout_MI_color_vs_xi"))

    fig_singlecol(2.35)
    ax = plt.gca()
    ax.plot(xi_scan, ENpost, marker='s', markevery=2)
    ax.set_xlabel(r"recombination fraction $\xi$")
    ax.set_ylabel(r"$E_{\mathcal{N}}^{(c)}$ post FO")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_freezeout_logneg_color_vs_xi"))

    # K/pi vs Tf
    Tf_grid = np.linspace(120, 200, 81)
    xi_choices = (0.0, 0.5, 1.0)

    fig_singlecol(2.40)
    ax = plt.gca()
    markers = ['o', 's', '^']
    for i, xi in enumerate(xi_choices):
        ratios = []
        for Tf in Tf_grid:
            _, info = hadron_register_from_xi(float(xi), float(Tf), mp)
            ratios.append(info["K_over_pi"])
        ax.plot(
            Tf_grid, ratios,
            label=fr"$\xi={xi:.1f}$",
            marker=markers[i],
            markevery=max(1, len(Tf_grid)//8),
            linewidth=1.8
        )
    ax.axvline(mp.Tf_MeV, linestyle="--", linewidth=1.0, label=r"$T_f$")
    ax.set_xlabel(r"$T_f\,[\mathrm{MeV}]$")
    ax.set_ylabel(r"$K/\pi$")
    ax.legend(loc="upper left")
    finalize_ax(ax)
    savefig(os.path.join(mp.outdir, "fig_Kpi_vs_Tf_xi_choices"))

    with open(os.path.join(mp.outdir, "run_meta.json"), "w") as f:
        json.dump({
            "ModelParams": asdict(mp),
            "kappa_q0_scan": list(kscan),
            "notes": [
                "Explicit ABG color Hilbert space with A in 3, B in conjugate 3, G in adjoint 8.",
                "Color diffusion implemented with SU(3)-covariant GKSL jump operators T_A^a, Tbar_B^a, F_G^a.",
                "Gluon traced out at freeze-out before mesonic hadronization.",
                "Freeze-out recombination branch represented as an effective memory-erasing AB reduced map after bath trace.",
                "Hadron register is explicit but minimal: diagonal 2-level {|pi>,|K>} sector.",
                "This version uses paper-ready plotting settings for publication figures."
            ]
        }, f, indent=2)

    print(f"[OK] Beautiful paper-ready plots saved in: {mp.outdir}")

if __name__ == "__main__":
    run_all()
