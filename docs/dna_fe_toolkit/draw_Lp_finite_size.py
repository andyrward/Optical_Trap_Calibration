"""Plot apparent dsDNA persistence length vs contour length (finite-size formula).

Run from the repository root:
    python3 docs/dna_fe_toolkit/draw_Lp_finite_size.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LP_INF = 49.0   # nm
A = 2.78        # single-trap value (Chen et al. 2009; Seol et al. 2007)

def lp(L):
    return LP_INF / (1.0 + A * LP_INF / L)

L = np.linspace(50, 1500, 1000)
L90 = A * LP_INF / (1 / 0.9 - 1)          # contour length where Lp = 0.9 Lp_inf
Lloop = 200.0

fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
ax.plot(L, lp(L), color="#3a6ea5", lw=2, label=r"$L_p(L)=L_p^\infty/(1+aL_p^\infty/L)$")
ax.axhline(LP_INF, color="0.5", ls="--", lw=1, label=r"$L_p^\infty$ = 49 nm")
ax.axhline(0.9 * LP_INF, color="#c0504d", ls=":", lw=1)
ax.plot([L90], [0.9 * LP_INF], "o", color="#c0504d")
ax.annotate(f"90% of $L_p^\\infty$: $L_p$ = {0.9*LP_INF:.1f} nm\nat $L$ = {L90:.0f} nm",
            xy=(L90, 0.9 * LP_INF), xytext=(800, 33), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#c0504d"), color="#c0504d")
ax.plot([Lloop], [lp(Lloop)], "s", color="#4f8f4f")
ax.annotate(f"DNA-caliper loop\n$L$ = {Lloop:.0f} nm, $L_p$ = {lp(Lloop):.1f} nm",
            xy=(Lloop, lp(Lloop)), xytext=(320, 17), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#4f8f4f"), color="#4f8f4f")
ax.set_xlim(0, 1500); ax.set_ylim(0, 55)
ax.set_xlabel("Contour length $L$ (nm)")
ax.set_ylabel("Apparent persistence length $L_p$ (nm)")
ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("docs/dna_fe_toolkit/DNA_Lp_finite_size.png", facecolor="white")
fig.savefig("docs/dna_fe_toolkit/DNA_Lp_finite_size.svg", facecolor="white")
