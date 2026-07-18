"""Phase-1 confirmatory fit: the p-axis taxonomy across the EXPANDED engine set
(5 original + QuestDB/MonetDB/DataFusion), by engine class. Effect-agnostic, same
discipline as run_variance.py: exact rational ground truth, fit each engine's exponent p.
"""
import sys, json, math
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import engines, oracle
from engines_phase1 import NEW
from run_variance import make_column, fit_p, assign_algo

OUT = __file__.rsplit("/", 2)[0] + "/results"

CLASS = {
    "postgres": "OLTP row-store", "mysql": "OLTP row-store", "sqlite": "embedded",
    "duckdb": "columnar OLAP (Arrow)", "clickhouse": "columnar OLAP",
    "datafusion": "columnar OLAP (Arrow)", "monetdb": "columnar OLAP", "questdb": "time-series",
}


def run():
    ALL = engines.ALL + NEW
    RATIOS = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    n = 10_000
    rows = []
    for r in RATIOS:
        xs = make_column(n, float(r), seed=(hash((n, r)) & 0xffff) + 1)
        exV = oracle.exact_var(xs); kapV = oracle.kappa_var(xs)
        if exV is None or kapV is None:
            continue
        for cls in ALL:
            e = cls()
            try:
                e.load(xs)
                ok, msg = e.roundtrip_check(xs)
                if not ok:
                    print(f"  !! transport {e.name} r={r}: {msg}", file=sys.stderr); continue
                for variant, val in e.variances().items():
                    if val is None: continue
                    err = oracle.rel_error(val, exV)
                    rows.append(dict(ratio=float(r), kappa_V=float(kapV), engine=e.name,
                                     relerr=float(err) if err is not None else 0.0))
            except Exception as ex_:
                print(f"  !! {cls.__name__} r={r}: {type(ex_).__name__}: {str(ex_)[:80]}", file=sys.stderr)
            finally:
                try: e.close()
                except Exception: pass
        print(f"  done r={r:<8g} kappa_V={float(kapV):.3e}", file=sys.stderr)

    names = sorted({r["engine"] for r in rows})
    fits = {}
    for eng in names:
        pairs = [(r["kappa_V"], r["relerr"]) for r in rows if r["engine"] == eng]
        p, npts = fit_p(pairs)
        fits[eng] = dict(p=p, n_fit=npts, algo=assign_algo(p), cls=CLASS.get(eng, "?"))
    json.dump({"rows": rows, "fits": fits}, open(OUT + "/variance_phase1.json", "w"), indent=1)

    print("\n=== p-axis taxonomy across 8 engines (fitted exponent; near 1 = stable, near 2 = one-pass) ===")
    print(f"{'engine':<12} {'class':<24} {'p':>7}  algorithm")
    for eng in sorted(names, key=lambda e: -(fits[e]['p'] or 0)):
        f = fits[eng]; ps = f"{f['p']:.2f}" if f['p'] is not None else "n/a"
        print(f"{eng:<12} {f['cls']:<24} {ps:>7}  {oracle.ALGO[f['algo']][2]}")
    onepass = [e for e in names if fits[e]['algo'] == 'var_onepass']
    print(f"\none-pass (p~2) engines: {onepass or 'NONE but ClickHouse-class'}  "
          f"| stable (p~1): {[e for e in names if e not in onepass]}")


if __name__ == "__main__":
    run()
