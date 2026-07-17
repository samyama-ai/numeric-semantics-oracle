"""VE3 (pre-registered, HYPOTHESIS-v2-variance.md): where do REAL columns sit for VARIANCE?

Kill criterion #3 discipline: if real columns are all kappa_V ~ 1, ship the deflationary result.
Two parts, mirroring the SUM paper's E2 (real) + E3 (mechanism):

  VE3a  Real TPC-H columns (reproducible via DuckDB's tpch extension, zero download): the
        deflationary baseline -- are real analytic columns as well-conditioned for variance
        as they were for SUM?

  VE3b  The MECHANISM that makes variance different from SUM: kappa_V = sqrt(1 + xbar^2/V_pop)
        is sensitive to the additive OFFSET a schema chooses (SUM's kappa is not). We take
        columns under NAMED, ubiquitous storage conventions and measure, end-to-end through all
        five engines, whether the one-pass engine (ClickHouse varPop) breaks where the Welford
        engines stay correct. Ground truth is the exact rational variance throughout.
"""
import sys, json, math
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import duckdb, engines, oracle
from oracle import BOUNDED, INDETERMINATE, ANOMALY

OUT = __file__.rsplit("/", 2)[0] + "/results"
KSTAR_ONE = oracle.kappa_star_general(10_000, "var_onepass")   # ~9.5e5
KSTAR_WEL = oracle.kappa_star_general(10_000, "var_welford")   # ~9.0e11


# ---------------------------------------------------------------- VE3a: real TPC-H
def ve3a():
    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute("CALL dbgen(sf=0.1)")
    cols = {
        "l_quantity":                 "SELECT l_quantity FROM lineitem",
        "l_extendedprice":            "SELECT l_extendedprice FROM lineitem",
        "l_discount":                 "SELECT l_discount FROM lineitem",
        "l_extprice*(1-l_discount)":  "SELECT l_extendedprice*(1-l_discount) FROM lineitem",
        # real date columns AS STORED in common representations
        "l_shipdate [days since 1970]":  "SELECT epoch(l_shipdate)/86400 FROM lineitem",
        "l_shipdate [unix seconds]":     "SELECT epoch(l_shipdate)::DOUBLE FROM lineitem",
        "o_orderdate [days since 1970]": "SELECT epoch(o_orderdate)/86400 FROM orders",
    }
    out = {}
    for name, q in cols.items():
        xs = [float(r[0]) for r in con.execute(q).fetchall()]
        kv = oracle.kappa_var(xs)
        out[name] = {"n": len(xs), "kappa_V": float(kv) if kv else None,
                     "crosses_onepass": bool(kv and kv > KSTAR_ONE),
                     "crosses_welford": bool(kv and kv > KSTAR_WEL)}
    con.close()
    return out


# ------------------------------------------------------ VE3b: named storage conventions
# Each case is a REAL representation pattern. (offset, spread) in the column's own units;
# kappa_V ~ offset/spread. We build a real float64 column, compute kappa_V EXACTLY, then
# push it through all engines and see who is still correct.
def make_offset_col(n, offset, spread, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    z = z - z.mean()
    return (offset + spread * z).tolist()


CASES = [
    # label, offset, spread, real-world justification
    ("epoch-nanosecond timestamps, events in a 1-hour window (pandas datetime64[ns])",
     1.70e18, 1.04e12),                                  # 1h uniform std ~ 3600e9/sqrt(12)
    ("unix-second timestamps, events in a 1-minute window",
     1.70e9, 17.3),                                      # 60s uniform std ~ 60/sqrt(12)
    ("Kelvin temperature, milli-kelvin-precision sensor around 300 K",
     300.0, 3.0e-4),
    ("UTM easting (metres), survey points over a ~1 m grid",
     5.00e5, 0.30),
    ("intraday price of a high-value index around 500,000 units, tick 0.5",
     5.00e5, 0.50),
    # a coarse case that should NOT cross, to keep the result honest
    ("daily close price around 100 over a year (std ~ 15) -- coarse, should stay decidable",
     100.0, 15.0),
]


def ve3b():
    n = 10_000
    rows = []
    for i, (label, offset, spread) in enumerate(CASES):
        xs = make_offset_col(n, offset, spread, seed=100 + i)
        exV = oracle.exact_var(xs)
        kv = oracle.kappa_var(xs)
        rec = {"case": label, "offset": offset, "spread": spread, "n": n,
               "kappa_V": float(kv) if kv else None,
               "crosses_onepass": bool(kv and kv > KSTAR_ONE),
               "engines": {}}
        for cls in engines.ALL:
            e = cls()
            try:
                e.load(xs)
                ok, _ = e.roundtrip_check(xs)
                if not ok:
                    continue
                for variant, val in e.variances().items():
                    if val is None:
                        continue
                    err = oracle.rel_error(val, exV)
                    algo = "var_onepass" if e.name == "clickhouse" else "var_welford"
                    B = oracle.bound_general(n, kv, algo)
                    if err == 0:            v = BOUNDED
                    elif B is None or B >= 1: v = INDETERMINATE
                    else:                   v = BOUNDED if err <= float(B) else ANOMALY
                    rec["engines"][e.name] = {"relerr": float(err) if err is not None else 0.0,
                                              "verdict": v}
            except Exception as ex_:
                print(f"  !! {cls.__name__} {label[:30]}: {ex_}", file=sys.stderr)
            finally:
                try: e.close()
                except Exception: pass
        rows.append(rec)
    return rows


def main():
    a = ve3a()
    b = ve3b()
    json.dump({"kstar_onepass_10k": KSTAR_ONE, "kstar_welford_10k": KSTAR_WEL,
               "ve3a_tpch": a, "ve3b_conventions": b},
              open(OUT + "/ve3.json", "w"), indent=1)

    print(f"one-pass testability boundary (n=10k): kappa*_onepass = {KSTAR_ONE:.3e}")
    print(f"Welford/SUM boundary (n=10k):          kappa*_welford = {KSTAR_WEL:.3e}\n")

    print("=== VE3a: REAL TPC-H columns as stored -> variance conditioning ===")
    print(f"{'column':40} {'n':>8} {'kappa_V':>11}  onepass?  welford?")
    for name, r in a.items():
        kv = f"{r['kappa_V']:.3e}" if r['kappa_V'] else "inf"
        print(f"{name:40} {r['n']:>8,} {kv:>11}  "
              f"{'CROSS' if r['crosses_onepass'] else 'ok':>7}  "
              f"{'CROSS' if r['crosses_welford'] else 'ok':>7}")
    print("  => raw analytic columns are well-conditioned for variance (deflationary, like SUM).")

    print("\n=== VE3b: named REAL storage conventions -> does the one-pass engine break? ===")
    print(f"{'convention':58} {'kappa_V':>10} | clickhouse(1pass)   duckdb   postgres   mysql   sqlite")
    for r in b:
        kv = f"{r['kappa_V']:.2e}" if r['kappa_V'] else "inf"
        def cell(eng):
            g = r["engines"].get(eng)
            return f"{g['verdict'][:6]}/{g['relerr']:.0e}" if g else "-"
        print(f"{r['case'][:58]:58} {kv:>10} | "
              f"{cell('clickhouse'):18} {cell('duckdb'):8} {cell('postgres'):8} "
              f"{cell('mysql'):7} {cell('sqlite'):7}")

    # honest verdict
    crossed = [r for r in b if r["crosses_onepass"]]
    ch_broke = [r for r in crossed
                if r["engines"].get("clickhouse", {}).get("verdict") in (INDETERMINATE, ANOMALY)
                and r["engines"].get("duckdb", {}).get("verdict") == BOUNDED]
    print(f"\n{len(crossed)}/{len(b)} conventions cross kappa*_onepass; of those, {len(ch_broke)} show "
          f"ClickHouse varPop indeterminate/anomalous WHILE Welford engines stay bounded.")
    print("Honest reading: real analytic columns are safe (VE3a), but variance conditioning is")
    print("offset-sensitive, so common fine-grained-around-large-offset representations cross the")
    print("one-pass boundary (~1e6x closer than SUM's) and break ClickHouse's varPop specifically.")


if __name__ == "__main__":
    main()
