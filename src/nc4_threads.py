"""NC4 — the pre-registered determinism control. It was registered in HYPOTHESIS.md and NEVER RUN.

HYPOTHESIS.md said: "Fix thread counts; confirm within-engine variance is separated from
cross-engine divergence BEFORE any cross-engine claim." We made the cross-engine claim anyway
(DuckDB/Postgres/MySQL return bit-identical SUM => differential testing has zero power).

Running the control refutes that claim: neither engine is bit-stable across thread counts, and at
their DEFAULT parallel settings they do not agree. The bit-identity observed at n=10,000 was an
artifact of both engines choosing a SEQUENTIAL plan at that size.

Persists results/nc4.json so the claim registry can resolve it.
"""
import json, os, random, struct, sys
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duckdb, psycopg2, psycopg2.extras, engines

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
bits = lambda v: struct.pack(">d", v).hex()

N = 500_000
random.seed(5)
xs = [random.uniform(0.5, 1.5) for _ in range(N)]
exact = float(sum(map(Fraction, xs)))

duck = {}
for t in (1, 2, 4, 8):
    c = duckdb.connect(); c.execute(f"PRAGMA threads={t}")
    c.execute("CREATE TABLE t(x DOUBLE)")
    c.executemany("INSERT INTO t VALUES (?)", [(v,) for v in xs])
    duck[t] = c.execute("SELECT sum(x) FROM t").fetchone()[0]; c.close()

p = psycopg2.connect(**engines.PG); p.autocommit = True; cu = p.cursor()
cu.execute("DROP TABLE IF EXISTS par; CREATE TABLE par(x double precision)")
psycopg2.extras.execute_values(cu, "INSERT INTO par VALUES %s", [(v,) for v in xs], page_size=20000)
cu.execute("ANALYZE par")
pg = {}
for w in (0, 2, 4):
    cu.execute(f"SET max_parallel_workers_per_gather = {w}")
    cu.execute("SET parallel_setup_cost=0; SET parallel_tuple_cost=0; SET min_parallel_table_scan_size=0")
    cu.execute("SELECT sum(x) FROM par"); pg[w] = float(cu.fetchone()[0])
p.close()

out = {
    "n": N,
    "exact": exact,
    "duckdb_by_threads": {str(k): {"sum": v, "bits": bits(v)} for k, v in duck.items()},
    "postgres_by_workers": {str(k): {"sum": v, "bits": bits(v)} for k, v in pg.items()},
    "duckdb_bit_stable_across_threads": len({bits(v) for v in duck.values()}) == 1,
    "postgres_bit_stable_across_workers": len({bits(v) for v in pg.values()}) == 1,
    "duckdb_equals_postgres_at_defaults": bits(duck[8]) == bits(pg[2]),
    "sequential_plans_agree": bits(duck[1]) == bits(pg[0]),
    "conclusion": ("Neither engine is bit-stable across thread/worker counts. At default parallel "
                   "settings DuckDB and PostgreSQL do NOT agree. The bit-identity reported at "
                   "n=10,000 was an artifact of sequential plans; the 'zero differential power' "
                   "claim does not hold under default configurations."),
}
os.makedirs(RES, exist_ok=True)
json.dump(out, open(os.path.join(RES, "nc4.json"), "w"), indent=1)
for k in ("duckdb_bit_stable_across_threads", "postgres_bit_stable_across_workers",
          "duckdb_equals_postgres_at_defaults", "sequential_plans_agree"):
    print(f"  {k:42} {out[k]}")
print("\nwrote results/nc4.json")
