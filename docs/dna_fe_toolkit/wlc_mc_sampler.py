"""Preliminary worm-like chain (WLC) sampler: bending only, no twist.

PRELIMINARY / EXPLORATORY. Not part of the analysis pipeline.

Purpose
-------
Test whether an ideal, bending-only worm-like chain reproduces the
phenomenological finite-size persistence-length formula quoted in the
Camunas-Soler et al. (2016) review,

    Lp(L) = Lp_inf / (1 + a * Lp_inf / L),  Lp_inf = 49 nm, a = 2.78
    (single-trap value; Chen et al. 2009; Seol et al. 2007),

with emphasis on the DNA-caliper loop, L = 200 nm (about 588 bp at 0.34 nm/bp),
where the formula gives Lp = 29.1 nm.

Model
-----
Discrete Kratky-Porod chain of N bonds with unit tangents t_i and bond length
b = L/N (b = 1 nm here, i.e. b = Lp/49). Energy in units of kBT:

    E/kBT = -kappa * sum_i t_i . t_{i+1}  -  (F b / kBT) * sum_i t_i . z_hat

(equivalent to kappa * sum(1 - t_i.t_{i+1}) up to a constant). kBT = 4.114 pN nm.
kappa is chosen with the EXACT discrete relation

    <cos theta> = coth(kappa) - 1/kappa = exp(-b/Lp),

so that the discrete chain's tangent correlation is exactly
<t_i.t_j> = exp(-|i-j| b / Lp) at F = 0. (kappa = Lp/b is only the
continuum-limit approximation and is not used.) Ends are free: no beads,
no handles, no attachment geometry, and no stretch modulus.

Methods
-------
1. F = 0, direct sampling. With bending only and no force, successive bond
   angles are independent (tangents form a Markov chain), so chains are
   generated exactly by drawing cos(theta) from p ~ exp(kappa cos theta) and a
   uniform azimuth at each joint. No Monte Carlo correlation or equilibration
   is involved. <R^2> is averaged over independent chains (standard error from
   the chain-to-chain spread). An effective Lp is extracted by inverting the
   continuum Kratky-Porod formula
       <R^2> = 2 Lp L [1 - (Lp/L)(1 - exp(-L/Lp))].
2. F > 0, transfer matrix (exact for the discrete chain, deterministic). The
   bending kernel exp(kappa t.t') is diagonal in Legendre polynomials with
   eigenvalues proportional to i_l(kappa); the force factor exp(phi cos theta)
   is applied through Gauss-Legendre quadrature (eigenbasis of the cos theta
   matrix). <z>(F) = b * sum_i <cos theta_i> is computed with forward/backward
   vectors. Truncation l_max is checked by doubling it.
3. F > 0, Metropolis Monte Carlo (independent check of the transfer matrix at
   selected L). Pivot moves: a random bond k and everything after it are rotated
   by a random angle in [-delta, delta] about a random axis. Many independent
   replicas are run in parallel (numpy-vectorized), started from exact F = 0
   samples; the first half of each run is discarded as equilibration; errors
   are the standard error over replicas of each replica's time average.
4. Apparent Lp: fit the inextensible Marko-Siggia interpolation
       F = (kBT/Lp) [1/(4(1-z/Lc)^2) - 1/4 + z/Lc]
   to <z>(F) at 10 log-spaced forces from 0.1 to 10 pN, minimizing extension
   residuals, with Lc fixed at the true L (Lp free), and in a variant with both
   Lc and Lp free. Force-range sensitivity is reported for sub-ranges.

Sanity checks
-------------
(a) Long chain (L = 2000 nm >> Lp): <z>(F) vs Marko-Siggia and vs the Bouchiat
    et al. (1999) 7th-order corrected interpolation (near-exact infinite WLC).
(b) Short chain (L = 5 and 10 nm << Lp): <R>/L -> 1 (rigid rod) at F = 0, and
    <z>/L vs the rigid-rod (Langevin) result coth(FL/kBT) - kBT/(FL).

Caveats
-------
This is a preliminary, bending-only model with ideal free-end boundary
conditions: no beads, no handles or linkers, no attachment geometry, no
stretch modulus, no twist, no electrostatics/salt dependence. Seol et al.
(2007) measured contour lengths of 0.6-7 um, so the 200 nm case is an
extrapolation below their range. The "apparent Lp" depends on the fitted
model (Marko-Siggia is itself an approximate interpolation, with errors of up
to ~10% in force at intermediate extension) and on the force range.

Run from the repository root:
    python3 docs/dna_fe_toolkit/wlc_mc_sampler.py
Outputs go to docs/dna_fe_toolkit/wlc_mc_output/.
"""
import os
import time

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import ive

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KBT = 4.114          # pN nm
LP_IN = 49.0         # nm, input persistence length
LP_INF = 49.0        # nm, phenomenological formula
A_SINGLE = 2.78      # single-trap a (Chen et al. 2009; Seol et al. 2007)
B = 1.0              # nm, bond length
L_LIST = [50, 100, 200, 300, 500, 1000, 2000]
L_SHORT = [5, 10]
FORCES = np.logspace(-1, 1, 10)       # pN, 0.1-10
OUTDIR = "docs/dna_fe_toolkit/wlc_mc_output"
SEED = 20261111


# ---------------------------------------------------------------- basic maths
def langevin(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    s = np.abs(x) < 1e-4
    out[s] = x[s] / 3.0 - x[s] ** 3 / 45.0
    out[~s] = 1.0 / np.tanh(x[~s]) - 1.0 / x[~s]
    return out


def kappa_exact(b, lp):
    """Solve coth(k) - 1/k = exp(-b/lp) for k."""
    target = np.exp(-b / lp)
    k_guess = 1.0 / (1.0 - target)
    return brentq(lambda k: float(langevin(k)) - target, 1e-6, 10 * k_guess + 10, xtol=1e-14)


def formula_lp(L):
    return LP_INF / (1.0 + A_SINGLE * LP_INF / np.asarray(L, float))


def kp_r2(L, lp):
    x = L / lp
    return 2.0 * lp * L * (1.0 - (1.0 - np.exp(-x)) / x)


def invert_kp(L, r2):
    if r2 >= L * L:
        return np.inf
    return brentq(lambda lp: kp_r2(L, lp) - r2, 1e-6, 1e9, xtol=1e-12)


def discrete_r2_exact(N, b, alpha):
    return b * b * (N * (1 + alpha) / (1 - alpha) - 2 * alpha * (1 - alpha ** N) / (1 - alpha) ** 2)


def ms_force(z, Lc, lp):
    x = z / Lc
    return KBT / lp * (0.25 / (1 - x) ** 2 - 0.25 + x)


BOUCHIAT = [0.0, 0.0, -0.5164228, -2.737418, 16.07497, -38.87607, 39.49944, -14.17718]


def bouchiat_force(z, Lc, lp):
    x = z / Lc
    poly = sum(a * x ** i for i, a in enumerate(BOUCHIAT))
    return KBT / lp * (0.25 / (1 - x) ** 2 - 0.25 + x + poly)


def invert_force_model(model, F, Lc, lp):
    return np.array([brentq(lambda z: model(z, Lc, lp) - f, 0.0, Lc * (1 - 1e-12), xtol=1e-12)
                     for f in np.atleast_1d(F)])


# ----------------------------------------------------- F = 0 direct sampling
def sample_cos(kappa, rng, n):
    u = rng.random(n)
    return 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa


def random_perp_frame(t):
    a = np.zeros_like(t)
    use_x = np.abs(t[:, 0]) < 0.9
    a[use_x, 0] = 1.0
    a[~use_x, 1] = 1.0
    e1 = np.cross(t, a)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(t, e1)
    return e1, e2


def sample_chains_f0(N, kappa, n_chains, rng, return_tangents=False):
    """Exact sampling of free chains at F = 0. Returns end vectors (and tangents)."""
    t = np.zeros((n_chains, 3))
    # isotropic first tangent
    cz = rng.uniform(-1, 1, n_chains)
    ph = rng.uniform(0, 2 * np.pi, n_chains)
    sz = np.sqrt(1 - cz ** 2)
    t[:, 0], t[:, 1], t[:, 2] = sz * np.cos(ph), sz * np.sin(ph), cz
    R = t.copy()
    tans = np.empty((n_chains, N, 3)) if return_tangents else None
    if return_tangents:
        tans[:, 0] = t
    for i in range(1, N):
        c = sample_cos(kappa, rng, n_chains)
        s = np.sqrt(np.clip(1 - c * c, 0, None))
        ph = rng.uniform(0, 2 * np.pi, n_chains)
        e1, e2 = random_perp_frame(t)
        t = c[:, None] * t + s[:, None] * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2)
        t /= np.linalg.norm(t, axis=1)[:, None]
        R += t
        if return_tangents:
            tans[:, i] = t
    return (R * B, tans) if return_tangents else R * B


# ------------------------------------------------------ transfer matrix (F>0)
class TransferMatrix:
    def __init__(self, kappa, lmax):
        l = np.arange(lmax + 1)
        # eigenvalue ratios lambda_l / lambda_0 = I_{l+1/2}(k)/I_{1/2}(k)
        self.lam = ive(l + 0.5, kappa) / ive(0.5, kappa)
        # cos(theta) in normalized Legendre (m=0) basis: tridiagonal
        off = (l[:-1] + 1) / np.sqrt((2 * l[:-1] + 1) * (2 * l[:-1] + 3))
        C = np.diag(off, 1) + np.diag(off, -1)
        self.C = C
        self.c_eig, self.U = np.linalg.eigh(C)   # Gauss-Legendre nodes / vectors
        self.n = lmax + 1

    def mean_cos_sum(self, phi, N):
        """Return sum_i <cos theta_i> for an N-bond free chain, phi = F b / kBT."""
        D = (self.U * np.exp(phi * (self.c_eig - self.c_eig.max()))) @ self.U.T
        K = self.lam
        e0 = np.zeros(self.n); e0[0] = 1.0
        # forward: f_1 = D e0 ; f_{i+1} = D K f_i  (rescaled)
        fw = np.empty((N, self.n))
        v = D @ e0; v /= np.abs(v).max(); fw[0] = v
        for i in range(1, N):
            v = D @ (K * v); v /= np.abs(v).max(); fw[i] = v
        # backward: g_N = e0 ; g_i = K D g_{i+1}  -> <cos_i> ~ g_i^T C f_i / g_i^T f_i
        g = e0.copy()
        tot = 0.0
        for i in range(N - 1, -1, -1):
            f = fw[i]
            tot += (g @ (self.C @ f)) / (g @ f)
            g = K * (D @ g); g /= np.abs(g).max()
        return tot


def tm_extension(L, F, kappa, lmax):
    N = int(round(L / B))
    tm = TransferMatrix(kappa, lmax)
    return np.array([B * tm.mean_cos_sum(f * B / KBT, N) for f in np.atleast_1d(F)])


# ------------------------------------------------------- Metropolis pivot MC
def rotate_rodrigues(v, axis, ang):
    """Rotate vectors v (R, M, 3) about per-replica axis (R,3) by ang (R,)."""
    c = np.cos(ang)[:, None, None]; s = np.sin(ang)[:, None, None]
    k = axis[:, None, :]
    kv = np.sum(k * v, axis=2, keepdims=True)
    return v * c + np.cross(k, v) * s + k * kv * (1 - c)


def mc_pivot(L, F, kappa, n_rep, n_sweeps, rng, delta=0.5):
    N = int(round(L / B))
    phi = F * B / KBT
    _, t = sample_chains_f0(N, kappa, n_rep, rng, return_tangents=True)
    n_moves = n_sweeps * N
    zs = np.empty((n_moves // N, n_rep))
    acc = 0
    idx = np.arange(N)[None, :]
    rr = np.arange(n_rep)
    for m in range(n_moves):
        k = rng.integers(0, N, n_rep)
        ax = rng.normal(size=(n_rep, 3)); ax /= np.linalg.norm(ax, axis=1)[:, None]
        ang = rng.uniform(-delta, delta, n_rep)
        tn = rotate_rodrigues(t, ax, ang)
        mask = idx >= k[:, None]
        # bending change at junction (k-1, k)
        km1 = np.maximum(k - 1, 0)
        has = k > 0
        old_b = np.einsum("ij,ij->i", t[rr, km1], t[rr, k])
        new_b = np.einsum("ij,ij->i", t[rr, km1], tn[rr, k])
        dE = -kappa * (new_b - old_b) * has
        dz = np.sum((tn[:, :, 2] - t[:, :, 2]) * mask, axis=1)
        dE -= phi * dz
        ok = rng.random(n_rep) < np.exp(-np.clip(dE, -50, 50))
        upd = mask & ok[:, None]
        t = np.where(upd[:, :, None], tn, t)
        acc += ok.mean()
        if (m + 1) % N == 0:
            zs[(m + 1) // N - 1] = t[:, :, 2].sum(axis=1) * B
    half = zs.shape[0] // 2
    prod = zs[half:]
    per_rep = prod.mean(axis=0)
    # block averaging within replicas as a cross-check of the replica SE
    nb = 5
    blocks = np.array_split(prod, nb, axis=0)
    bm = np.array([b.mean() for b in blocks])
    return dict(z=per_rep.mean(), se=per_rep.std(ddof=1) / np.sqrt(n_rep),
                block_se=bm.std(ddof=1) / np.sqrt(nb), acc=acc / n_moves,
                eq_first_half=zs[:half].mean(), n_meas=prod.size)


# ------------------------------------------------------------- MS fitting
def fit_ms(F, z, L, free_lc=False):
    if free_lc:
        def res(p):
            lc, lp = p
            if lc <= z.max():
                return np.full_like(z, 1e3)
            return (invert_force_model(ms_force, F, lc, lp) - z) / L
        r = least_squares(res, [L, 40.0], bounds=([z.max() * 1.0001, 0.5], [3 * L, 500]))
    else:
        def res(p):
            return (invert_force_model(ms_force, F, L, p[0]) - z) / L
        r = least_squares(res, [40.0], bounds=([0.5], [500]))
    J = r.jac
    dof = max(len(z) - len(r.x), 1)
    s2 = np.sum(r.fun ** 2) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        err = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        err = np.full(len(r.x), np.nan)
    return r.x, err, np.sqrt(np.mean(r.fun ** 2)) * L


# ===================================================================== main
def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    kappa = kappa_exact(B, LP_IN)
    alpha = np.exp(-B / LP_IN)
    print(f"b = {B} nm, Lp_in = {LP_IN} nm, kappa(exact) = {kappa:.6f} "
          f"(continuum Lp/b = {LP_IN / B:.1f}); <cos> = {float(langevin(kappa)):.10f} "
          f"vs exp(-b/Lp) = {alpha:.10f}")

    # ---- lmax convergence of transfer matrix
    LMAX = 120
    zc1 = tm_extension(200, [10.0], kappa, LMAX)[0]
    zc2 = tm_extension(200, [10.0], kappa, 2 * LMAX)[0]
    print(f"TM lmax check (L=200, F=10 pN): lmax={LMAX}: {zc1:.8f}; lmax={2*LMAX}: {zc2:.8f}")

    # ---- F = 0 sampling
    n_chains = {5: 20000, 10: 20000, 50: 20000, 100: 20000, 200: 20000,
                300: 20000, 500: 10000, 1000: 5000, 2000: 4000}
    f0 = {}
    for L in L_SHORT + L_LIST:
        N = int(round(L / B))
        R = sample_chains_f0(N, kappa, n_chains[L], rng)
        r2 = np.sum(R ** 2, axis=1)
        rn = np.sqrt(r2)
        m, se = r2.mean(), r2.std(ddof=1) / np.sqrt(len(r2))
        lp = invert_kp(L, m)
        lo = invert_kp(L, m - se) if m - se > 0 else np.nan
        hi = invert_kp(L, m + se) if m + se < L * L else np.inf
        f0[L] = dict(N=N, n=len(r2), r2=m, r2_se=se, r2_exact=discrete_r2_exact(N, B, alpha),
                     r2_kp=kp_r2(L, LP_IN), lp=lp, lp_err=0.5 * (hi - lo),
                     lp_exact_discrete=invert_kp(L, discrete_r2_exact(N, B, alpha)),
                     R_over_L=rn.mean() / L, R_over_L_se=rn.std(ddof=1) / np.sqrt(len(rn)) / L)
        print(f"F=0 L={L:5d} N={N:5d} n={len(r2):6d} <R2>={m:12.2f}+-{se:9.2f} "
              f"(exact discrete {f0[L]['r2_exact']:12.2f}, KP {f0[L]['r2_kp']:12.2f}) "
              f"Lp_R2={lp:8.2f}+-{f0[L]['lp_err']:.2f} <R>/L={f0[L]['R_over_L']:.4f}")
    print(f"[t={time.time()-t0:.1f}s]")

    # ---- transfer-matrix FE curves
    fe = {}
    for L in L_SHORT + L_LIST:
        fe[L] = tm_extension(L, FORCES, kappa, LMAX)
    print(f"[TM done t={time.time()-t0:.1f}s]")

    # ---- MC check under force
    mc_rows = []
    mc_cfg = {50: (200, 600), 200: (64, 300)}
    mc_forces = FORCES[[0, 3, 6, 9]]
    for L, (n_rep, n_sweeps) in mc_cfg.items():
        for F in mc_forces:
            ts = time.time()
            r = mc_pivot(L, F, kappa, n_rep, n_sweeps, rng)
            ztm = tm_extension(L, [F], kappa, LMAX)[0]
            dev = (r["z"] - ztm) / r["se"]
            mc_rows.append((L, F, r["z"], r["se"], r["block_se"], ztm, dev, r["acc"], r["eq_first_half"],
                            n_rep, n_sweeps))
            print(f"MC L={L} F={F:6.3f}: <z>={r['z']:9.3f}+-{r['se']:.3f} (block SE {r['block_se']:.3f}) "
                  f"TM={ztm:9.3f} dev={dev:+.2f} SE acc={r['acc']:.2f} "
                  f"eq-half mean={r['eq_first_half']:.3f} [{time.time()-ts:.1f}s]")
    print(f"[MC done t={time.time()-t0:.1f}s]")

    # ---- fits
    results = []
    ranges = {"0.1-10": (0.1, 10), "0.1-1": (0.1, 1.0), "1-10": (1.0, 10), "0.5-5": (0.5, 5.0)}
    for L in L_LIST:
        z = fe[L]
        (lp_fix,), (lp_fix_e,), rms_fix = fit_ms(FORCES, z, L)
        (lc_fr, lp_fr), (lc_fr_e, lp_fr_e), rms_fr = fit_ms(FORCES, z, L, free_lc=True)
        sens = {}
        for name, (a, b2) in ranges.items():
            sel = (FORCES >= a * 0.999) & (FORCES <= b2 * 1.001)
            sens[name] = fit_ms(FORCES[sel], z[sel], L)[0][0]
        row = dict(L_nm=L, N_bonds=f0[L]["N"], bp_at_0p34=L / 0.34,
                   Lp_R2_nm=f0[L]["lp"], Lp_R2_err_nm=f0[L]["lp_err"], n_chains_F0=f0[L]["n"],
                   Lp_R2_exact_discrete_nm=f0[L]["lp_exact_discrete"],
                   Lp_app_MS_Lcfixed_nm=lp_fix, Lp_app_MS_Lcfixed_fit_se_nm=lp_fix_e, rms_resid_Lcfixed_nm=rms_fix,
                   Lc_fit_MS_free_nm=lc_fr, Lc_fit_MS_free_se_nm=lc_fr_e,
                   Lp_app_MS_free_nm=lp_fr, Lp_app_MS_free_se_nm=lp_fr_e, rms_resid_free_nm=rms_fr,
                   Lp_formula_nm=float(formula_lp(L)))
        for name in ranges:
            row[f"Lp_app_Lcfixed_F{name}pN_nm"] = sens[name]
        results.append(row)
        print(f"L={L:5d} Lp_R2={row['Lp_R2_nm']:7.2f}+-{row['Lp_R2_err_nm']:.2f} "
              f"Lp_app(Lc fixed)={lp_fix:7.2f}+-{lp_fix_e:.2f} "
              f"[free: Lc={lc_fr:8.2f}, Lp={lp_fr:7.2f}+-{lp_fr_e:.2f}] formula={row['Lp_formula_nm']:6.2f} "
              f"| ranges " + " ".join(f"{k}:{v:.2f}" for k, v in sens.items()))

    # ---- sanity checks
    L = 2000
    z_ms = invert_force_model(ms_force, FORCES, L, LP_IN)
    z_bo = invert_force_model(bouchiat_force, FORCES, L, LP_IN)
    dev_ms = (fe[L] - z_ms) / z_ms
    dev_bo = (fe[L] - z_bo) / z_bo
    hi = FORCES >= 1.0
    print("Sanity (a) L=2000: F, <z>/L, MS z/L, Bouchiat z/L, rel dev MS, rel dev Bouchiat")
    for f, a1, a2, a3, d1, d2 in zip(FORCES, fe[L] / L, z_ms / L, z_bo / L, dev_ms, dev_bo):
        print(f"   {f:7.3f} {a1:.5f} {a2:.5f} {a3:.5f} {d1:+.4f} {d2:+.4f}")
    san = dict(max_abs_dev_MS=np.abs(dev_ms).max(), max_abs_dev_Bouchiat=np.abs(dev_bo).max(),
               max_abs_dev_Bouchiat_F_ge_1=np.abs(dev_bo[hi]).max(),
               max_abs_dev_MS_F_ge_1=np.abs(dev_ms[hi]).max())
    print("   ", {k: round(float(v), 5) for k, v in san.items()})
    print("Sanity (b) short chains:")
    short_rows = []
    for Ls in L_SHORT:
        for f in [1.0, 10.0]:
            i = int(np.argmin(np.abs(FORCES - f)))
            zt = fe[Ls][i]
            rod = Ls * float(langevin(FORCES[i] * Ls / KBT))
            short_rows.append((Ls, FORCES[i], f0[Ls]["R_over_L"], zt / Ls, rod / Ls))
            print(f"   L={Ls} nm: <R>/L(F=0)={f0[Ls]['R_over_L']:.4f}+-{f0[Ls]['R_over_L_se']:.4f}; "
                  f"F={FORCES[i]:.3f} pN <z>/L={zt/Ls:.4f}, rigid rod={rod/Ls:.4f}")

    # ---- write CSVs
    import csv
    with open(f"{OUTDIR}/wlc_mc_results_by_L.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(f"{OUTDIR}/wlc_fe_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["L_nm", "F_pN", "z_TM_nm", "z_over_L", "z_MS_Lp49_nm", "z_Bouchiat_Lp49_nm",
                    "z_MSfit_Lcfixed_nm"])
        for L in L_SHORT + L_LIST:
            zms = invert_force_model(ms_force, FORCES, L, LP_IN)
            zbo = invert_force_model(bouchiat_force, FORCES, L, LP_IN)
            match = [r for r in results if r["L_nm"] == L]
            zfit = (invert_force_model(ms_force, FORCES, L, match[0]["Lp_app_MS_Lcfixed_nm"])
                    if match else np.full_like(FORCES, np.nan))
            for f, a, b1, c, d in zip(FORCES, fe[L], zms, zbo, zfit):
                w.writerow([L, f"{f:.6g}", f"{a:.6g}", f"{a/L:.6g}", f"{b1:.6g}", f"{c:.6g}", f"{d:.6g}"])
    with open(f"{OUTDIR}/wlc_mc_force_check.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["L_nm", "F_pN", "z_MC_nm", "z_MC_se_nm", "z_MC_block_se_nm", "z_TM_nm",
                    "dev_in_SE", "acceptance", "mean_z_equilibration_half_nm", "n_replicas", "n_sweeps"])
        for r in mc_rows:
            w.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in r])
    with open(f"{OUTDIR}/wlc_sanity_checks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["check", "quantity", "value"])
        for k, v in san.items():
            w.writerow(["long_chain_L2000", k, f"{v:.6g}"])
        for Ls, f, rl, zl, rod in short_rows:
            w.writerow([f"short_chain_L{Ls}", "R_over_L_F0", f"{rl:.6g}"])
            w.writerow([f"short_chain_L{Ls}", f"z_over_L_F{f:.3g}", f"{zl:.6g}"])
            w.writerow([f"short_chain_L{Ls}", f"rigid_rod_z_over_L_F{f:.3g}", f"{rod:.6g}"])

    # ---- figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=300)
    Lg = np.linspace(30, 2100, 500)
    ax1.plot(Lg, formula_lp(Lg), color="#3a6ea5", lw=2,
             label=r"Formula $L_p^\infty/(1+aL_p^\infty/L)$, $a$=2.78")
    ax1.axhline(LP_IN, color="0.5", ls="--", lw=1, label=r"Input $L_p$ = 49 nm")
    Ls_ = np.array([r["L_nm"] for r in results])
    ax1.errorbar(Ls_, [r["Lp_R2_nm"] for r in results], yerr=[r["Lp_R2_err_nm"] for r in results],
                 fmt="o", color="#4f8f4f", ms=5, capsize=2, label=r"$L_p$ from $\langle R^2\rangle$ (F = 0)")
    ax1.errorbar(Ls_, [r["Lp_app_MS_Lcfixed_nm"] for r in results],
                 yerr=[r["Lp_app_MS_Lcfixed_fit_se_nm"] for r in results],
                 fmt="s", color="#c0504d", ms=5, capsize=2, label=r"Apparent $L_p$, MS fit ($L_c$ = L fixed)")
    ax1.plot(Ls_, [r["Lp_app_MS_free_nm"] for r in results], "^", color="#8064a2", ms=5,
             label=r"Apparent $L_p$, MS fit ($L_c$, $L_p$ free)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Contour length $L$ (nm)")
    ax1.set_ylabel("Persistence length $L_p$ (nm)")
    ax1.set_ylim(0, 70)
    ax1.legend(fontsize=7, frameon=False, loc="lower right")
    ax1.set_title("(a) Bending-only WLC, b = 1 nm, 0.1–10 pN fit", fontsize=9)
    L = 200
    ax2.plot(FORCES, fe[L] / L, "o", color="#c0504d", label="Transfer matrix (exact, discrete chain)")
    mcL = [r for r in mc_rows if r[0] == L]
    ax2.errorbar([r[1] for r in mcL], [r[2] / L for r in mcL], yerr=[r[3] / L for r in mcL],
                 fmt="x", color="k", ms=6, capsize=2, label="Metropolis pivot MC")
    Ff = np.logspace(-1, 1, 200)
    ax2.plot(Ff, invert_force_model(ms_force, Ff, L, LP_IN) / L, color="#3a6ea5", lw=1.5,
             label=r"Marko–Siggia, $L_p$ = 49 nm")
    lpa = [r for r in results if r["L_nm"] == L][0]["Lp_app_MS_Lcfixed_nm"]
    ax2.plot(Ff, invert_force_model(ms_force, Ff, L, lpa) / L, color="#c0504d", ls="--", lw=1,
             label=f"MS fit, $L_p$ = {lpa:.1f} nm")
    ax2.plot(Ff, invert_force_model(ms_force, Ff, L, float(formula_lp(L))) / L, color="#4f8f4f", ls=":",
             lw=1.5, label=f"MS, formula $L_p$ = {float(formula_lp(L)):.1f} nm")
    ax2.set_xscale("log")
    ax2.set_xlabel("Force $F$ (pN)")
    ax2.set_ylabel(r"Relative extension $\langle z\rangle/L$")
    ax2.set_title("(b) L = 200 nm (≈588 bp)", fontsize=9)
    ax2.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/wlc_mc_apparent_Lp.png", facecolor="white")
    print(f"Total runtime {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
