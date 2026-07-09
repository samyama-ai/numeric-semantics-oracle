"""E2 (pre-registered): where do REAL aggregate queries sit relative to kappa* and kappa_eps?

Kill criterion #3: if real workloads all sit at kappa ~ 1, the indeterminate regime is a
curiosity, not a practical hazard -> ship the deflationary result, do NOT inflate it.

Data: TPC-H via DuckDB's official tpch extension (public, standard, reproducible, no download).
We take the actual SUM expressions from TPC-H Q1, Q6 and Q9. Q9's `amount` is naturally SIGNED
(revenue minus supply cost) -- the realistic source of cancellation.
"""
import sys
from fractions import Fraction
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import duckdb, oracle

SF = 0.1
con = duckdb.connect()
con.execute("INSTALL tpch; LOAD tpch;")
con.execute(f"CALL dbgen(sf={SF})")
print(f"TPC-H sf={SF}: lineitem rows = {con.execute('SELECT count(*) FROM lineitem').fetchone()[0]:,}\n")

EXPRS = {
    "Q1  sum(l_quantity)":                      "SELECT l_quantity FROM lineitem",
    "Q1  sum(l_extendedprice)":                 "SELECT l_extendedprice FROM lineitem",
    "Q1  sum(l_extprice*(1-l_discount))":       "SELECT l_extendedprice*(1-l_discount) FROM lineitem",
    "Q6  sum(l_extprice*l_discount)":           "SELECT l_extendedprice*l_discount FROM lineitem",
    "Q9  sum(amount)  [SIGNED profit]":         """SELECT l_extendedprice*(1-l_discount)
                                                    - ps_supplycost*l_quantity
                                                  FROM lineitem JOIN partsupp
                                                    ON ps_partkey=l_partkey AND ps_suppkey=l_suppkey""",
}

EPS = {"1e-9": 1e-9, "duckdb_1pct": 1e-2}
print(f"{'expression':40} {'n':>9} {'kappa':>11} {'B=g_n*k':>11} {'verdict':>14} | eps soundness")
print("-" * 122)
for name, q in EXPRS.items():
    xs = [r[0] for r in con.execute(q).fetchall()]
    xs = [float(v) for v in xs]
    n = len(xs)
    kap = oracle.kappa(xs)
    if kap is None:
        print(f"{name:40} {n:>9,}  sum is exactly zero -> kappa infinite"); continue
    B = oracle.bound_rel(n, kap)
    kstar = oracle.kappa_star(n)
    decidable = kap < kstar
    notes = []
    for en, ev in EPS.items():
        ke = oracle.kappa_eps(n, ev)
        notes.append(f"{en}: {'too LAX (FN risk)' if kap < ke else 'too STRICT (FP risk)'} "
                     f"by {float(ev/B) if kap<ke else float(B/Fraction(ev)):.3g}x")
    print(f"{name:40} {n:>9,} {float(kap):11.3e} {float(B):11.3e} "
          f"{('decidable' if decidable else 'INDETERMINATE'):>14} | " + " ; ".join(notes))

print(f"\nkappa*(plain) at these n: {float(oracle.kappa_star(600572)):.3e} (n=600k)")
print("\nAll-positive columns have kappa = 1 EXACTLY (no cancellation). Signed expressions are")
print("where cancellation -- and the whole testability question -- actually arises.")
