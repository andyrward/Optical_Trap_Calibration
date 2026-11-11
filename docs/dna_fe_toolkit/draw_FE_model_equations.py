import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
rows = [
 ("1. Worm-like chain (WLC)", "Marko & Siggia, Macromolecules 28, 8759 (1995)",
  r"$F=\dfrac{k_BT}{p}\left[\dfrac{1}{4\,(1-z/L_c)^{2}}-\dfrac{1}{4}+\dfrac{z}{L_c}\right]$"),
 ("2. Extensible worm-like chain (eWLC)", "Wang, Yin, Landick, Gelles & Block, Biophys. J. 72, 1335 (1997)",
  r"$F=\dfrac{k_BT}{p}\left[\dfrac{1}{4\,(1-z/L_c+F/S)^{2}}-\dfrac{1}{4}+\dfrac{z}{L_c}-\dfrac{F}{S}\right]$"),
 ("3. Freely jointed chain (FJC)", "Smith, Cui & Bustamante, Science 271, 795 (1996)",
  r"$z=L_c\,\mathcal{L}\!\left(\dfrac{F\,b}{k_BT}\right),\qquad \mathcal{L}(u)=\coth u-\dfrac{1}{u}$"),
 ("4. Extensible freely jointed chain (eFJC)", "Smith, Cui & Bustamante, Science 271, 795 (1996)",
  r"$z=L_c\,\mathcal{L}\!\left(\dfrac{F\,b}{k_BT}\right)\left(1+\dfrac{F}{S}\right)$"),
]
fig = plt.figure(figsize=(9, 8.2), facecolor="white")
y = 0.95
for name, cite, eq in rows:
    fig.text(0.05, y, name, fontsize=15, weight="bold", va="top")
    fig.text(0.05, y-0.045, cite, fontsize=11, style="italic", color="#444", va="top")
    fig.text(0.5, y-0.135, eq, fontsize=21, ha="center", va="center")
    y -= 0.225
fig.text(0.05, 0.05, r"$z$: end-to-end extension;  $L_c$: contour length;  $p$: persistence length;  $b$: Kuhn length ($b = 2p$);", fontsize=11, color="#333")
fig.text(0.05, 0.022, r"$S$: stretch modulus;  $k_BT$: thermal energy (4.11 pN$\cdot$nm at 298 K)", fontsize=11, color="#333")
out = "docs/dna_fe_toolkit/DNA_FE_model_equations"
fig.savefig(out + ".png", dpi=300, facecolor="white")
fig.savefig(out + ".svg", facecolor="white")
