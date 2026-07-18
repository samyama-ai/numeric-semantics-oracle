"""Figures for the re-scoped paper (general C_A*kappa_f^p frame; SUM=p=1, variance=p=2).
Okabe-Ito palette (CVD-validated); linestyle/marker/hatch as secondary encoding so every
figure survives greyscale. Direct labels. Reads results/{variance_map,ve3}.json."""
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
PURPLE = "#CC79A7"
INK, MUTED = "#222222", "#666666"
U = 2.0**-53

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#dddddd", "grid.linewidth": 0.5,
})

# distinct style per engine, greyscale-safe (color + marker + linestyle)
ENG_STYLE = {
    "clickhouse": (VERM,   "o", "-"),
    "duckdb":     (BLUE,   "s", "--"),
    "postgres":   (GREEN,  "^", "-."),
    "mysql":      (ORANGE, "D", ":"),
    "sqlite":     (PURPLE, "v", (0, (3, 1, 1, 1))),
}


def _gamma(n):
    nu = n * U
    return nu / (1 - nu)


# ---------- Figure V1: the general testability boundary, by algorithm exponent p ----------
def figv1():
    n = np.logspace(2, 7, 300)
    g = n * U / (1 - n * U)
    kstar_p1 = 1 / g                       # SUM & Welford variance (p=1)
    kstar_p2 = (1 / g) ** 0.5              # one-pass variance (p=2)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.fill_between(n, kstar_p2, 1e18, color="#fbe9e2", zorder=0)
    ax.fill_between(n, kstar_p1, kstar_p2, color="#eaf3fb", zorder=0)

    ax.loglog(n, kstar_p1, color=BLUE, linestyle="-", linewidth=2.4,
              label=r"$\kappa^*_{f,A}=1/\gamma_n$  (SUM, AVG, Welford var; $p{=}1$)")
    ax.loglog(n, kstar_p2, color=VERM, linestyle="--", linewidth=2.4,
              label=r"$\kappa^*_{f,A}=(1/\gamma_n)^{1/2}$  (one-pass var; $p{=}2$)")

    ax.annotate(r"$p{=}1$: decidable to $\kappa\sim10^{12}$", xy=(n[-1], kstar_p1[-1]),
                xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=BLUE, va="center")
    ax.annotate(r"$p{=}2$: decidable only to $\kappa\sim10^{6}$", xy=(n[-1], kstar_p2[-1]),
                xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=VERM, va="center")
    ax.text(3e2, 4e14, "INDETERMINATE for one-pass variance,\nstill decidable for Welford / SUM",
            fontsize=8, color=MUTED, va="top")

    # the measured gap at n=10^4
    ax.annotate("", xy=(1e4, kstar_p1[np.argmin(abs(n - 1e4))]),
                xytext=(1e4, kstar_p2[np.argmin(abs(n - 1e4))]),
                arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.0))
    ax.text(1.2e4, 3e8, r"$\sim\!10^{6}\times$ narrower", fontsize=7.8, color=INK, rotation=90,
            va="center")

    ax.set_xlim(1e2, 1e7); ax.set_ylim(1, 1e18)
    ax.set_xlabel("column length $n$")
    ax.set_ylabel(r"condition number $\kappa_f$")
    ax.set_title(r"The testability boundary is a property of the algorithm: $\kappa^*_{f,A}=(1/C_A)^{1/p}$",
                 loc="left", fontsize=9)
    ax.grid(True, which="major", axis="y", zorder=0)
    ax.legend(loc="lower left", fontsize=7.6, frameon=False)
    fig.savefig(os.path.join(OUT, "figv1_boundary.png"), dpi=220, bbox_inches="tight")
    print("wrote figv1_boundary.png")


# ---------- Figure V2: measured exponent p per engine (the finding) ----------
def figv2():
    d = json.load(open(os.path.join(RES, "variance_map.json")))
    rows = [r for r in d["rows"] if r["n"] == 10000]
    fits = d["fits"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for eng, (c, mk, ls) in ENG_STYLE.items():
        pts = sorted([(r["kappa_V"], r["relerr"]) for r in rows if r["engine"] == eng])
        xs = [k for k, e in pts if e > 0]
        ys = [e for k, e in pts if e > 0]
        p = fits[eng]["p"]
        lab = f"{eng}  (p={p:.2f})" if p is not None else eng
        ax.loglog(xs, ys, color=c, marker=mk, linestyle=ls, linewidth=1.6,
                  markersize=6, label=lab, markerfacecolor="white", markeredgewidth=1.3)

    # reference slopes p=1 and p=2
    kk = np.array([1e2, 1e9])
    ax.loglog(kk, 1e-16 * kk**1, color=MUTED, linestyle=":", linewidth=1.0)
    ax.loglog(kk, 1e-16 * kk**2, color=MUTED, linestyle=":", linewidth=1.0)
    ax.text(1.1e9, 1e-16 * (1e9)**1, r" slope $p{=}1$", fontsize=7.5, color=MUTED, va="center")
    ax.text(3e6, 1e-16 * (3e6)**2, r"slope $p{=}2$", fontsize=7.5, color=MUTED, va="bottom", ha="right")
    ax.axhline(1.0, color=INK, linewidth=0.8, linestyle="-")
    ax.text(1e9, 1.6, "rel err $=1$", fontsize=7, color=INK, ha="right", va="bottom")

    ax.set_xlabel(r"variance condition number $\kappa_V=\sqrt{\sum x_i^2/\sum(x_i-\bar x)^2}$")
    ax.set_ylabel("relative error vs exact rational variance")
    ax.set_title(r"Measured error growth recovers each engine's algorithm ($n=10{,}000$)", loc="left")
    ax.grid(True, which="major", axis="both", zorder=0)
    ax.legend(loc="upper left", fontsize=7.6, frameon=False)
    fig.savefig(os.path.join(OUT, "figv2_exponent.png"), dpi=220, bbox_inches="tight")
    print("wrote figv2_exponent.png")


# ---------- Figure V3: the variance classification map ----------
def figv3():
    d = json.load(open(os.path.join(RES, "variance_map.json")))
    rows = [r for r in d["rows"] if r["n"] == 10000]
    engs = ["clickhouse", "duckdb", "postgres", "mysql", "sqlite"]
    labels = ["clickhouse\n(one-pass)", "duckdb\n(Welford)", "postgres\n(Welford)",
              "mysql\n(Welford)", "sqlite\n(Welford UDF)"]
    kappas = sorted({r["ratio"] for r in rows})
    CMAP = {"exact": GREEN, "bounded": SKY, "indeterminate": VERM, "ANOMALY": "#000000"}
    HATCH = {"exact": "", "bounded": "", "indeterminate": "///", "ANOMALY": "xxx"}

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for i, k in enumerate(kappas):
        for j, eng in enumerate(engs):
            m = next((r for r in rows if r["ratio"] == k and r["engine"] == eng), None)
            if not m:
                continue
            ax.add_patch(plt.Rectangle((j, i), 0.96, 0.96, facecolor=CMAP[m["verdict"]],
                                       alpha=0.55, hatch=HATCH[m["verdict"]],
                                       edgecolor="white", linewidth=1.5))
            txt = "0" if m["relerr"] == 0 else f"{m['relerr']:.0e}".replace("e-0", "e-").replace("e+0", "e")
            ax.text(j + 0.48, i + 0.48, txt, ha="center", va="center", fontsize=6.4, color=INK)

    kvs = {r["ratio"]: r["kappa_V"] for r in rows}
    ax.set_xticks([j + 0.48 for j in range(len(engs))]); ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks([i + 0.48 for i in range(len(kappas))])
    ax.set_yticklabels([f"$10^{{{int(round(np.log10(kvs[k])))}}}$" for k in kappas], fontsize=8)
    ax.set_xlim(0, len(engs)); ax.set_ylim(0, len(kappas))
    ax.set_ylabel(r"variance condition number $\kappa_V$")
    ax.set_title(r"Variance oracle verdicts, $n=10{,}000$ (cell = rel err vs exact)", loc="left")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(facecolor=CMAP[v], alpha=0.55, hatch=HATCH[v], label=v)
                       for v in ["bounded", "indeterminate"]],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, "figv3_map.png"), dpi=220, bbox_inches="tight")
    print("wrote figv3_map.png")


# ---------- Figure V4: real storage conventions cross the one-pass boundary ----------
def figv4():
    d = json.load(open(os.path.join(RES, "ve3.json")))
    kstar_one = d["kstar_onepass_10k"]
    kstar_wel = d["kstar_welford_10k"]
    conv = d["ve3b_conventions"]
    short = {
        "epoch-nanosecond timestamps, events in a 1-hour window (pandas datetime64[ns])":
            "epoch-ns timestamps,\n1-hour window",
        "unix-second timestamps, events in a 1-minute window":
            "unix-sec timestamps,\n1-min window",
        "Kelvin temperature, milli-kelvin-precision sensor around 300 K":
            "milli-K sensor\naround 300 K",
        "UTM easting (metres), survey points over a ~1 m grid":
            "UTM easting (m),\n~1 m grid",
        "intraday price of a high-value index around 500,000 units, tick 0.5":
            "high-value index\nprice ~5e5",
        "daily close price around 100 over a year (std ~ 15) -- coarse, should stay decidable":
            "daily close ~100\n(coarse)",
    }
    names = [short[c["case"]] for c in conv]
    kvs = [c["kappa_V"] for c in conv]
    ch = [c["engines"].get("clickhouse", {}).get("verdict", "-") for c in conv]
    y = np.arange(len(conv))[::-1]

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.axvspan(kstar_one, 1e12, color="#fbe9e2", zorder=0)
    ax.axvline(kstar_one, color=VERM, linestyle="--", linewidth=1.8)
    ax.axvline(kstar_wel, color=BLUE, linestyle="-", linewidth=1.8)
    ax.text(kstar_one, len(conv) - 0.3, r" $\kappa^*_{one\text{-}pass}$", color=VERM, fontsize=8, va="top")
    ax.text(kstar_wel, len(conv) - 0.3, r" $\kappa^*_{Welford/SUM}$", color=BLUE, fontsize=8, va="top")

    for yi, (nm, kv, v) in zip(y, zip(names, kvs, ch)):
        broke = v in ("indeterminate", "ANOMALY")
        ax.plot(kv, yi, marker="o" if broke else "s", markersize=11,
                color=VERM if broke else GREEN, markeredgecolor=INK, markeredgewidth=0.8, zorder=3)
        ax.text(kv * (0.5 if broke else 1.7), yi, nm, fontsize=7,
                ha="right" if broke else "left", va="center", color=INK)

    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xscale("log"); ax.set_xlim(1, 1e12); ax.set_ylim(-0.7, len(conv) - 0.1)
    ax.set_xlabel(r"variance condition number $\kappa_V$ of the real column")
    ax.set_title("Ordinary storage conventions cross the one-pass boundary (circle = ClickHouse breaks)",
                 loc="left", fontsize=8.6)
    ax.legend(handles=[Patch(color=VERM, label="ClickHouse varPop indeterminate"),
                       Patch(color=GREEN, label="all engines decidable")],
              loc="lower right", fontsize=7.6, frameon=False)
    ax.grid(True, axis="x", which="major", zorder=0)
    fig.savefig(os.path.join(OUT, "figv4_realdata.png"), dpi=220, bbox_inches="tight")
    print("wrote figv4_realdata.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    figv1(); figv2(); figv3(); figv4()


# ---------- Figure V5: p-axis taxonomy across engine classes (Phase 1+2 expansion) ----------
CLASS_COLOR = {"columnar OLAP": VERM, "columnar OLAP (Arrow)": BLUE, "OLTP row-store": GREEN,
               "time-series": ORANGE, "embedded": PURPLE}

def figv5():
    p1 = json.load(open(os.path.join(RES, "variance_phase1.json")))
    p2 = json.load(open(os.path.join(RES, "covar_phase2.json")))
    var_p = {e: f["p"] for e, f in p1["fits"].items()}
    var_cls = {e: f["cls"] for e, f in p1["fits"].items()}
    cov_p = {e: f["p"] for e, f in p2["fits"].items()}
    # order: stable engines by p, ClickHouse (one-pass) at top
    engs = sorted(var_p, key=lambda e: var_p[e])
    y = list(range(len(engs)))

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.axvspan(1.7, 2.5, color="#fbe9e2", zorder=0)
    ax.axvline(1.0, color=MUTED, ls="--", lw=1.2); ax.axvline(2.0, color=MUTED, ls="--", lw=1.2)
    ax.text(1.0, len(engs) - 0.3, " p=1 stable", color=MUTED, fontsize=8, va="top")
    ax.text(2.0, len(engs) - 0.3, " p=2 one-pass", color=VERM, fontsize=8, va="top")
    for yi, e in zip(y, engs):
        c = CLASS_COLOR.get(var_cls[e], INK)
        ax.plot(var_p[e], yi, "o", ms=11, color=c, mec=INK, mew=0.8, zorder=3)
        if e in cov_p:
            ax.plot(cov_p[e], yi, "D", ms=7, color="white", mec=c, mew=1.6, zorder=3)
        lbl = e + ("" if e in cov_p else "  (no covar)")
        ax.text(min(var_p[e], cov_p.get(e, 9)) - 0.05, yi, lbl, ha="right", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([var_cls[e] for e in engs], fontsize=7.5)
    ax.set_xlim(0.4, 2.4); ax.set_ylim(-0.6, len(engs) - 0.2)
    ax.set_xlabel("measured error exponent $p$   (circle = variance, diamond = covariance)")
    ax.set_title("The one-pass choice is a concentrated, engine-level outlier", loc="left", fontsize=9.5)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in CLASS_COLOR.items()],
              loc="lower right", fontsize=7, frameon=False, title="engine class", title_fontsize=7.5)
    ax.grid(True, axis="x", zorder=0)
    fig.savefig(os.path.join(OUT, "figv5_taxonomy.png"), dpi=220, bbox_inches="tight")
    print("wrote figv5_taxonomy.png")


if __name__ == "__main__":
    figv5()
