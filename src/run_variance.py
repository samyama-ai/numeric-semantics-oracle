"""Confirmatory VARIANCE run (paper21 re-scope, p != 1).
Pre-registered in dbms_cloud/daily/33-numeric-semantics-oracle/HYPOTHESIS-v2-variance.md (frozen first).

Effect-agnostic: we report the per-engine measured exponent p and the classification map against the
per-algorithm bound. No magnitude is claimed. Ground truth is the EXACT rational variance.
"""
import sys, json, math
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import engines, oracle
from oracle import EXACT, BOUNDED, INDETERMINATE, ANOMALY

OUT = __file__.rsplit("/", 2)[0] + "/results"


def make_column(n, ratio, seed):
    """Near-constant column: mean/std ~ ratio, so kappa_V = sqrt(1 + ratio^2) ~ ratio. Exactly
    centred spread (z-z.mean()) then offset. Larger ratio => more ill-conditioned variance.
    Values are float64 as stored; kappa_V is recomputed EXACTLY by the oracle afterwards."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    z = z - z.mean()
    return ((ratio * 1.0) + 1.0 * z).tolist()


def fit_p(pairs):
    """log-log slope of rel_err vs kappa_V over the meaningful window -> effective exponent p."""
    lk, le = [], []
    for kap, er in pairs:
        if kap and kap > 10 and 1e-15 < er < 1e-1:
            lk.append(math.log10(kap)); le.append(math.log10(er))
    if len(lk) < 3:
        return None, len(lk)
    A = np.vstack([lk, np.ones(len(lk))]).T
    slope = float(np.linalg.lstsq(A, np.array(le), rcond=None)[0][0])
    return slope, len(lk)


def assign_algo(p):
    """Recover which algorithm the engine uses from its fitted exponent (VE1)."""
    if p is None:
        return "var_welford"
    return "var_onepass" if p >= 1.5 else "var_welford"


def run():
    NS = [10_000, 100_000]
    RATIOS = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]     # -> kappa_V ~ 1 .. 1e9
    rows = []

    # NC1v: exact rational variance must match numpy on a well-conditioned column
    chk = make_column(5000, 1.0, 1)
    nc1 = abs(float(oracle.exact_var(chk)) - float(np.var(chk))) <= 1e-12 * float(np.var(chk))

    for n in NS:
        for r in (RATIOS if n == 10_000 else [1, 1e3, 1e6, 1e9]):
            xs = make_column(n, float(r), seed=(hash((n, r)) & 0xffff) + 1)
            exV = oracle.exact_var(xs)
            kapV = oracle.kappa_var(xs)
            if exV is None or kapV is None:
                continue
            for cls in engines.ALL:
                e = cls()
                try:
                    e.load(xs)
                    ok, _msg = e.roundtrip_check(xs)
                    if not ok:
                        print(f"  !! transport {e.name} n={n}: {_msg}", file=sys.stderr); continue
                    for variant, val in e.variances().items():
                        if val is None:
                            continue
                        err = oracle.rel_error(val, exV)
                        rows.append(dict(n=n, ratio=float(r), kappa_V=float(kapV), engine=e.name,
                                         variant=variant,
                                         relerr=float(err) if err is not None else 0.0))
                except Exception as ex_:
                    print(f"  !! {cls.__name__} n={n} r={r}: {type(ex_).__name__}: {ex_}", file=sys.stderr)
                finally:
                    try: e.close()
                    except Exception: pass
            print(f"  done n={n:>7} ratio={r:<8g} kappa_V={float(kapV):.3e}", file=sys.stderr)

    # ---- VE1: fit exponent p per engine (over the n=10,000 sweep) ----
    engine_names = sorted({r["engine"] for r in rows})
    fits = {}
    for eng in engine_names:
        pairs = [(r["kappa_V"], r["relerr"]) for r in rows if r["engine"] == eng and r["n"] == 10_000]
        p, npts = fit_p(pairs)
        algo = assign_algo(p)
        fits[eng] = dict(p=p, n_fit=npts, assigned_algo=algo, label=oracle.ALGO[algo][2])

    # ---- classify every cell against the engine's ASSIGNED-algorithm bound; NC2v hard gate ----
    anomalies = []
    verdict_counts = {EXACT: 0, BOUNDED: 0, INDETERMINATE: 0, ANOMALY: 0}
    for r in rows:
        algo = fits[r["engine"]]["assigned_algo"]
        B = oracle.bound_general(r["n"], r["kappa_V"], algo)
        err = r["relerr"]
        if err == 0:
            v = EXACT
        elif B is None or B >= 1:
            v = INDETERMINATE
        else:
            v = BOUNDED if err <= float(B) else ANOMALY
        r["verdict"] = v
        r["bound"] = float(B) if B is not None else float("inf")
        r["assigned_algo"] = algo
        verdict_counts[v] += 1
        if v == ANOMALY:
            anomalies.append(r)

    # ---- NC3v: a Welford-assigned engine must beat a one-pass bound where one-pass is indeterminate
    nc3_ok = True
    for r in rows:
        if fits[r["engine"]]["assigned_algo"] == "var_welford" and r["n"] == 10_000:
            B_one = oracle.bound_general(r["n"], r["kappa_V"], "var_onepass")
            if B_one is not None and B_one >= 1 and r["relerr"] >= 1:
                nc3_ok = False        # stable engine also blew up where one-pass is indeterminate

    # ---- boundaries ----
    boundaries = {n: {a: oracle.kappa_star_general(n, a)
                      for a in ("sum_plain", "var_welford", "var_onepass")} for n in NS}

    out = dict(rows=rows, fits=fits, anomalies=anomalies, boundaries=boundaries,
               nc={"NC1v_exact_gt": nc1, "NC2v_anomalies": len(anomalies), "NC3v_stable_collapse": nc3_ok})
    json.dump(out, open(OUT + "/variance_map.json", "w"), indent=1)

    # ================= report =================
    print("\n=== VE1: measured exponent p per engine (near 1 => Welford/two-pass; near 2 => one-pass) ===")
    for eng, f in fits.items():
        ps = f"{f['p']:.3f}" if f["p"] is not None else "n/a"
        print(f"  {eng:<11} p = {ps:<7} [{f['n_fit']} pts]  -> {f['label']}")

    print("\n=== testability boundary kappa*_{f,A} = (1/C_A)^(1/p) ===")
    for n, b in boundaries.items():
        print(f"  n={n:>7}: SUM/Welford(p=1)={b['var_welford']:.3e}   "
              f"one-pass variance(p=2)={b['var_onepass']:.3e}   "
              f"(ratio {b['var_welford']/b['var_onepass']:.2e}x narrower)")

    print("\n=== variance classification map (n=10,000): rel_err vs exact rational variance ===")
    engs = [(e, fits[e]["assigned_algo"]) for e in engine_names]
    hdr = f"{'kappa_V':>10} | " + " | ".join(f"{e[:10]:<20}" for e, _ in engs)
    print(hdr)
    for r0 in RATIOS:
        sel = [r for r in rows if r["n"] == 10_000 and r["ratio"] == r0]
        if not sel: continue
        line = f"{sel[0]['kappa_V']:10.2e} | "
        cells = []
        for eng, _ in engs:
            m = next((r for r in sel if r["engine"] == eng), None)
            cells.append(f"{m['verdict'][:11]:<11}{m['relerr']:.1e}" if m else " " * 20)
        print(line + " | ".join(f"{c:<20}" for c in cells))

    print(f"\n=== verdict totals over {len(rows)} (query,engine) cells ===")
    for k in (EXACT, BOUNDED, INDETERMINATE, ANOMALY):
        print(f"  {k:<14} {verdict_counts[k]}")
    print(f"\nNC1v ground-truth exactness : {'PASS' if nc1 else 'FAIL'}")
    print(f"NC2v bound soundness (HARD) : {len(anomalies)} anomalies "
          f"{'-> MUST TRIAGE' if anomalies else '-> PASS'}")
    print(f"NC3v stable-variant collapse: {'PASS' if nc3_ok else 'FAIL'}")
    for a in anomalies[:12]:
        print(f"   ANOMALY {a['engine']}/{a['variant']} n={a['n']} kappa_V={a['kappa_V']:.2e} "
              f"relerr={a['relerr']:.3e} > bound={a['bound']:.3e} (algo {a['assigned_algo']})")


if __name__ == "__main__":
    run()
