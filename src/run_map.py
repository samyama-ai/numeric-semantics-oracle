"""Confirmatory run: the classification map across (kappa, n, engine, summation variant).
Pre-registered in dbms_cloud/daily/33-numeric-semantics-oracle/HYPOTHESIS.md (frozen first).

Effect-agnostic: we report the map and locate kappa* / kappa_eps. No magnitude is claimed.
"""
import sys, json, random, math
from fractions import Fraction
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import engines, oracle
from oracle import EXACT, BOUNDED, INDETERMINATE, ANOMALY

COMPENSATED = {"kbn_default", "fsum_kahan", "sumKahan"}


def make_workload(n, target_kappa, seed):
    """Exact kappa control. Take positives summing to P, append one negative v so that
    total = t: then kappa = (2P - t)/t  =>  t = 2P/(kappa+1)."""
    rnd = random.Random(seed)
    xs = [rnd.uniform(0.5, 1.5) for _ in range(n - 1)]
    P = sum(map(Fraction, xs))
    if target_kappa <= 1:
        return xs + [float(rnd.uniform(0.5, 1.5))]
    t = 2 * P / (Fraction(target_kappa) + 1)
    v = float(t - P)                       # negative; rounds, so recompute kappa exactly after
    return xs + [v]


def run():
    NS = [10_000, 100_000]
    KAPPAS = [1, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12]
    EPSILONS = {"1e-9": 1e-9, "duckdb_1pct": 1e-2}
    rows, anomalies = [], []

    for n in NS:
        for tk in (KAPPAS if n == 10_000 else [1, 1e6, 1e12]):
            xs = make_workload(n, tk, seed=hash((n, tk)) & 0xffff)
            ex = oracle.exact_sum(xs)
            kap = oracle.kappa(xs)
            if kap is None:
                continue
            for cls in engines.ALL:
                e = cls()
                try:
                    e.load(xs)
                    for variant, val in e.sums().items():
                        comp = variant in COMPENSATED
                        verdict, err, B = oracle.classify(val, ex, n, kap, comp)
                        rows.append(dict(n=n, target_kappa=tk, kappa=float(kap), engine=e.name,
                                         variant=variant, compensated=comp,
                                         relerr=float(err) if err is not None else 0.0,
                                         bound=float(B) if B is not None else float("inf"),
                                         verdict=verdict))
                        if verdict == ANOMALY:
                            anomalies.append(rows[-1] | {"observed": repr(val), "exact": float(ex)})
                except Exception as ex_:
                    print(f"  !! {cls.__name__} n={n} k={tk}: {type(ex_).__name__}: {ex_}", file=sys.stderr)
                finally:
                    try: e.close()
                    except Exception: pass
            print(f"  done n={n:>7} target_kappa={tk:<8g} actual_kappa={float(kap):.3e}", file=sys.stderr)

    # ---- A1 / A2 boundaries (analytic, pre-registered) ----
    boundaries = {}
    for n in NS:
        b = {"kappa_star_plain": float(oracle.kappa_star(n)),
             "kappa_star_compensated": float(oracle.kappa_star(n, True))}
        for name, eps in EPSILONS.items():
            b[f"kappa_eps_plain[{name}]"] = float(oracle.kappa_eps(n, eps))
        boundaries[n] = b

    out = dict(rows=rows, anomalies=anomalies, boundaries=boundaries)
    json.dump(out, open(__file__.rsplit("/", 2)[0] + "/results/map.json", "w"), indent=1)

    # ---- report ----
    print("\n=== A1: testability boundary kappa* = 1/gamma_n  (beyond it, NO oracle can decide) ===")
    for n, b in boundaries.items():
        print(f"  n={n:>7}: kappa*(plain)={b['kappa_star_plain']:.3e}   "
              f"kappa*(compensated)={b['kappa_star_compensated']:.3e}")
    print("\n=== A2: epsilon crossover kappa_eps = eps/gamma_n (below: too LAX/FN; above: too STRICT/FP) ===")
    for n, b in boundaries.items():
        for k, v in b.items():
            if k.startswith("kappa_eps"):
                print(f"  n={n:>7}  {k:<28} = {v:.3e}")

    print("\n=== classification map (n=10,000) ===")
    print(f"{'kappa':>10} | " + " | ".join(f"{r:<22}" for r in
          ["sqlite/kbn", "duckdb/plain", "duckdb/fsum", "postgres/plain", "mysql/plain", "clickhouse/plain", "clickhouse/sumKahan"]))
    keys = [("sqlite","kbn_default"),("duckdb","plain"),("duckdb","fsum_kahan"),("postgres","plain"),
            ("mysql","plain"),("clickhouse","plain"),("clickhouse","sumKahan")]
    for tk in KAPPAS:
        sel = [r for r in rows if r["n"] == 10_000 and r["target_kappa"] == tk]
        if not sel: continue
        line = f"{sel[0]['kappa']:10.2e} | "
        cells = []
        for eng, var in keys:
            m = next((r for r in sel if r["engine"] == eng and r["variant"] == var), None)
            cells.append(f"{m['verdict']:<11} {m['relerr']:.1e}" if m else " " * 22)
        print(line + " | ".join(f"{c:<22}" for c in cells))

    from collections import Counter
    c = Counter(r["verdict"] for r in rows)
    print(f"\n=== verdict totals over {len(rows)} (query,engine,variant) cells ===")
    for k in (EXACT, BOUNDED, INDETERMINATE, ANOMALY):
        print(f"  {k:<14} {c[k]}")
    print(f"\nNC2 (bound soundness, HARD GATE): {len(anomalies)} anomalies "
          f"{'-> MUST TRIAGE' if anomalies else '-> PASS (no engine exceeded its bound)'}")
    for a in anomalies[:10]:
        print(f"   ANOMALY {a['engine']}/{a['variant']} n={a['n']} kappa={a['kappa']:.2e} "
              f"relerr={a['relerr']:.3e} > bound={a['bound']:.3e}")


if __name__ == "__main__":
    run()
