"""Paper figures. Okabe-Ito palette (CVD-validated); linestyle/hatch as secondary encoding so
every figure survives greyscale printing. Direct labels satisfy the contrast relief requirement.
Outputs PNGs into the paper directory."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.expanduser("~/projects/graph_ws/samyama-research/papers/paper21-numeric-oracle")

BLUE, VERM, GREEN, ORANGE, SKY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9"
INK, MUTED = "#222222", "#666666"
U = 2.0**-53

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#dddddd", "grid.linewidth": 0.5,
})


def gamma(n):
    nu = n * U
    return nu / (1 - nu)


# ---------------- Figure 1: the two boundaries ----------------
def fig1():
    n = np.logspace(1, 7, 200)
    g = n * U / (1 - n * U)
    kstar_plain = 1 / g
    kstar_comp = np.full_like(n, 1 / (2 * U))          # n-independent to first order
    keps_1pct = 1e-2 / g
    keps_1e9 = 1e-9 / g

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.fill_between(n, kstar_plain, 1e20, color="#eeeeee", hatch="///",
                    edgecolor="#cccccc", linewidth=0, zorder=0)
    ax.text(1.4e1, 6e17, "INDETERMINATE\n(no oracle can decide, plain sum)",
            fontsize=8, color=MUTED, va="top", ha="left")

    series = [
        (kstar_comp, GREEN, "--", r"$\kappa^*$ compensated $=1/2u$"),
        (kstar_plain, BLUE, "-", r"$\kappa^*$ plain $=1/\gamma_n$"),
        (keps_1pct, VERM, "-.", r"$\kappa_\varepsilon$, $\varepsilon=1\%$ (DuckDB)"),
        (keps_1e9, ORANGE, ":", r"$\kappa_\varepsilon$, $\varepsilon=10^{-9}$"),
    ]
    for y, c, ls, lab in series:
        ax.loglog(n, y, color=c, linestyle=ls, linewidth=2, label=lab)
        ax.annotate(lab, xy=(n[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color=c, va="center")

    ax.plot([600572], [1.0], marker="*", markersize=14, color="black", zorder=5)
    ax.annotate("TPC-H aggregates: $\\kappa=1.000$\n(10 orders inside the decidable zone)",
                xy=(600572, 1.0), xycoords="data",
                xytext=(0.04, 0.06), textcoords="axes fraction",
                fontsize=7.5, ha="left", va="center", color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7,
                                shrinkA=0, shrinkB=6))

    ax.set_xlim(10, 1e7); ax.set_ylim(1e-2, 1e18)
    ax.set_xlabel("number of summands $n$")
    ax.set_ylabel(r"condition number $\kappa$")
    ax.set_title("Testability boundary and epsilon crossover", loc="left")
    ax.grid(True, which="major", axis="y", zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              fontsize=7.5, frameon=False)
    fig.savefig(os.path.join(OUT, "fig1_boundaries.png"), dpi=220, bbox_inches="tight")
    print("wrote fig1_boundaries.png")


# ---------------- Figure 2: the classification map ----------------
def fig2():
    d = json.load(open(os.path.join(RES, "map.json")))
    rows = [r for r in d["rows"] if r["n"] == 10000]
    keys = [("sqlite", "kbn_default"), ("duckdb", "plain"), ("duckdb", "fsum_kahan"),
            ("postgres", "plain"), ("mysql", "plain"),
            ("clickhouse", "plain"), ("clickhouse", "sumKahan")]
    labels = ["sqlite\nKBN", "duckdb\nplain", "duckdb\nfsum", "postgres\nplain",
              "mysql\nplain", "clickhouse\nplain", "clickhouse\nsumKahan"]
    kappas = sorted({r["target_kappa"] for r in rows})
    CMAP = {"exact": GREEN, "bounded": SKY, "indeterminate": VERM}
    HATCH = {"exact": "", "bounded": "", "indeterminate": "///"}

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for i, k in enumerate(kappas):
        for j, (eng, var) in enumerate(keys):
            m = next((r for r in rows if r["target_kappa"] == k and r["engine"] == eng
                      and r["variant"] == var), None)
            if not m:
                continue
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.96, facecolor=CMAP[m["verdict"]],
                                       alpha=0.55, hatch=HATCH[m["verdict"]],
                                       edgecolor="white", linewidth=1.5))
            txt = "0" if m["relerr"] == 0 else f"{m['relerr']:.0e}".replace("e-0", "e-")
            ax.text(j + 0.48, i + 0.48, txt, ha="center", va="center", fontsize=6.6, color=INK)

    ax.set_xticks([j + 0.48 for j in range(len(keys))]); ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks([i + 0.48 for i in range(len(kappas))])
    ax.set_yticklabels([f"$10^{{{int(np.log10(k))}}}$" for k in kappas], fontsize=8)
    ax.set_xlim(0, len(keys)); ax.set_ylim(0, len(kappas))
    ax.set_ylabel(r"condition number $\kappa$")
    ax.set_title(r"Oracle verdicts, $n=10{,}000$ (cell = relative error vs exact)", loc="left")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(facecolor=CMAP[v], alpha=0.55, hatch=HATCH[v], label=v)
                       for v in ["exact", "bounded", "indeterminate"]],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, "fig2_map.png"), dpi=220, bbox_inches="tight")
    print("wrote fig2_map.png")


# ---------------- Figure 3: fuzzer undecidability is non-monotone ----------------
def fig3():
    e3 = json.load(open(os.path.join(RES, "e3.json")))
    ns = sorted(int(k) for k in e3)
    p_und = [e3[str(n)]["p_undec"] * 100 for n in ns]
    p_both = [e3[str(n)]["p_both"] * 100 for n in ns]

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.semilogx(ns, p_both, color=MUTED, linestyle=":", linewidth=1.6, marker="s",
                markersize=5, label=r"P(column contains both $\pm$MAX)")
    ax.semilogx(ns, p_und, color=VERM, linestyle="-", linewidth=2.2, marker="o",
                markersize=7, label="P(UNDECIDABLE)")
    ax.axhline(0, color=BLUE, linewidth=2, linestyle="--")
    ax.annotate("real data (TPC-H): 0% — always decidable", xy=(12, 0), xytext=(0, 5),
                textcoords="offset points", fontsize=7.5, color=BLUE)

    peak = int(np.argmax(p_und))
    ax.annotate(f"peak {p_und[peak]:.1f}% at $n$={ns[peak]:,}\n(giants cancel exactly: $k=m$)",
                xy=(ns[peak], p_und[peak]), xytext=(-10, -30), textcoords="offset points",
                fontsize=7.5, ha="right", va="top", color=VERM,
                arrowprops=dict(arrowstyle="-", color=VERM, linewidth=0.7, shrinkB=5))
    ax.annotate("all columns contain $\\pm$MAX,\nyet undecidability *falls*",
                xy=(ns[-1], p_und[-1]), xytext=(-4, 26), textcoords="offset points",
                fontsize=7.5, ha="right", color=INK)

    ax.set_xlabel("rows per column $n$ (SQLancer-style random doubles)")
    ax.set_ylabel("percent of columns")
    ax.set_ylim(-4, 108)
    ax.set_title("Undecidability of fuzzer-generated columns is non-monotone in $n$", loc="left")
    ax.grid(True, axis="y")
    ax.legend(loc="center left", fontsize=7.5, frameon=False)
    fig.savefig(os.path.join(OUT, "fig3_fuzzer.png"), dpi=220, bbox_inches="tight")
    print("wrote fig3_fuzzer.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1(); fig2(); fig3()
