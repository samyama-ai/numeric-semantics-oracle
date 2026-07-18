"""Phase 3: randomized NC2-anomaly bug hunt.
An ANOMALY = an engine's relative error EXCEEDS the bound of the algorithm it uses, AT A DECIDABLE
condition number (bound < 1). That is a genuine candidate bug (or a modelling error to triage) -- as
opposed to the indeterminate blow-ups past kappa* we already characterised, which are within-bound and
by-design. Ground truth is the exact rational value. Diverse generators; SUM / AVG / population variance.

ClickHouse's *default* varPop is documented-unstable (one-pass) -> judged against the p=2 bound, so its
large-offset blow-ups are NOT anomalies. We ALSO test varPopStable, which must satisfy the p=1 bound; if it
does not, that IS a real bug. Likewise any stable engine exceeding its bound is a real bug.
"""
import sys, json, math, itertools
import numpy as np
import psycopg2, psycopg2.extras, pymysql, clickhouse_connect, duckdb, sqlite3
from datafusion import SessionContext
import pyarrow as pa
from fractions import Fraction
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import oracle
from engines import _WelfordVarPop
OUT = __file__.rsplit("/", 2)[0] + "/results"
U = oracle.U


# ---------------- generators (diverse; aim to stress every path) ----------------
def gen(kind, n, seed):
    r = np.random.default_rng(seed)
    if kind == "uniform_pos":
        return r.uniform(0.5, 1.5, n).tolist()                       # kappa=1, best-conditioned
    if kind == "signed_kappa":                                       # controlled cancellation
        xs = r.uniform(0.5, 1.5, n - 1); P = float(sum(map(Fraction, xs)))
        tk = 10 ** r.integers(1, 7)
        return xs.tolist() + [float(2 * Fraction(P) / (Fraction(int(tk)) + 1) - Fraction(P))]
    if kind == "mixed_magnitude":
        big = r.uniform(1e12, 1e15, n // 2); small = r.uniform(1e-6, 1e-3, n - n // 2)
        v = np.concatenate([big, small]); r.shuffle(v); return v.tolist()
    if kind == "subnormals":
        return (r.uniform(1, 4, n) * 5e-324 * r.integers(1, 1e6, n)).tolist()
    if kind == "near_overflow":
        return r.uniform(1e300, 1.5e300, n).tolist()
    if kind == "integers":
        return r.integers(-10**6, 10**6, n).astype(float).tolist()
    if kind == "near_constant":                                      # variance stress, moderate kappa_V
        off = 10 ** r.integers(2, 6); z = r.standard_normal(n); z -= z.mean()
        return (off + z).tolist()
    raise ValueError(kind)

KINDS = ["uniform_pos", "signed_kappa", "mixed_magnitude", "subnormals", "near_overflow",
         "integers", "near_constant"]


# ---------------- engine measurement (sum / avg / var_pop [+ stable]) ----------------
def m_pg(port, xs):
    c = psycopg2.connect(host="127.0.0.1", port=port, user="postgres", password="oracle", dbname="numeric"); c.autocommit = True
    cur = c.cursor(); cur.execute("DROP TABLE IF EXISTS t;CREATE TABLE t(x float8)")
    psycopg2.extras.execute_values(cur, "INSERT INTO t VALUES %s", [(v,) for v in xs], page_size=10000)
    cur.execute("SELECT sum(x),avg(x),var_pop(x) FROM t"); s, a, v = cur.fetchone(); c.close()
    return {"sum": s, "avg": a, "var": v}
def m_mysql(xs):
    c = pymysql.connect(host="127.0.0.1", port=53306, user="root", password="oracle", database="numeric", autocommit=True)
    cur = c.cursor(); cur.execute("DROP TABLE IF EXISTS t"); cur.execute("CREATE TABLE t(x DOUBLE)")
    cur.executemany("INSERT INTO t VALUES(%s)", [(repr(v),) for v in xs])
    cur.execute("SELECT sum(x),avg(x),var_pop(x) FROM t"); s, a, v = cur.fetchone(); c.close()
    return {"sum": float(s), "avg": float(a), "var": float(v)}
def m_ch(xs):
    ch = clickhouse_connect.get_client(host="127.0.0.1", port=58123, username="default", password="oracle")
    ch.command("DROP TABLE IF EXISTS t"); ch.command("CREATE TABLE t(x Float64) ENGINE=Memory"); ch.insert("t", [[v] for v in xs], column_names=["x"])
    row = ch.query("SELECT sum(x),avg(x),varPop(x),varPopStable(x) FROM t").result_rows[0]
    return {"sum": row[0], "avg": row[1], "var": row[2], "var_stable": row[3]}
def m_duck(xs):
    d = duckdb.connect(); d.execute("CREATE TABLE t(x DOUBLE)"); d.executemany("INSERT INTO t VALUES(?)", [(v,) for v in xs])
    s, a, v = d.execute("SELECT sum(x),avg(x),var_pop(x) FROM t").fetchone()
    return {"sum": s, "avg": a, "var": v}
def m_sqlite(xs):
    c = sqlite3.connect(":memory:"); c.create_aggregate("wvp", 1, _WelfordVarPop)
    c.execute("CREATE TABLE t(x REAL)"); c.executemany("INSERT INTO t VALUES(?)", [(v,) for v in xs])
    s, a, v = c.execute("SELECT sum(x),avg(x),wvp(x) FROM t").fetchone(); c.close()
    return {"sum": s, "avg": a, "var": v}
def m_df(xs):
    ctx = SessionContext(); t = pa.table({"x": pa.array(xs, pa.float64())}); ctx.register_record_batches("t", [t.to_batches()])
    row = ctx.sql("SELECT sum(x),avg(x),var_pop(x) FROM t").collect()[0]
    return {"sum": row.column(0)[0].as_py(), "avg": row.column(1)[0].as_py(), "var": row.column(2)[0].as_py()}

# variance algorithm each engine's default is measured to use (Phase 1/2); stable variants are Welford
VAR_ALGO = {"postgres": "var_welford", "mysql": "var_welford", "duckdb": "var_welford",
            "sqlite": "var_welford", "datafusion": "var_welford", "clickhouse": "var_onepass"}


def anomalies_for(engine, res, xs, n, exS, kap, exV, kapV):
    """Return list of anomaly dicts for this engine's measured values."""
    out = []
    def check(kind, observed, exact, bound, algo):
        if observed is None or bound is None or bound >= 1:      # indeterminate or missing -> not an anomaly
            return
        if isinstance(observed, float) and (math.isnan(observed) or math.isinf(observed)):
            # NaN/inf where the bound is < 1 (decidable) is itself an anomaly
            out.append(dict(engine=engine, agg=kind, algo=algo, observed=str(observed),
                            relerr="nan/inf", bound=float(bound), n=n)); return
        err = oracle.rel_error(observed, exact)
        if err is not None and err > bound:
            out.append(dict(engine=engine, agg=kind, algo=algo, observed=repr(observed),
                            relerr=float(err), bound=float(bound), n=n))
    # SUM (plain recursive bound; sqlite is compensated)
    comp = engine == "sqlite"
    check("sum", res.get("sum"), exS, oracle.bound_rel(n, kap, comp), "sum_comp" if comp else "sum_plain")
    # AVG = SUM/n  (bound gamma_n*kappa + u)
    if exS != 0:
        b = oracle.bound_rel(n, kap, comp)
        check("avg", res.get("avg"), exS / n, (b + U) if b is not None else None, "avg")
    # VARIANCE (default) against the engine's measured algorithm
    check("var", res.get("var"), exV, oracle.bound_general(n, kapV, VAR_ALGO[engine]), VAR_ALGO[engine])
    # ClickHouse varPopStable MUST satisfy the Welford bound; if not -> real bug
    if "var_stable" in res:
        check("var_stable", res["var_stable"], exV, oracle.bound_general(n, kapV, "var_welford"), "var_welford(Stable)")
    return out


_MIN_NORMAL = 2.2250738585072014e-308
def in_domain(xs):
    """The forward-error bound assumes normalized floats and no overflow. Reject columns that
    contain subnormals, or whose exact sum|x| / sum x^2 would overflow float64 -- these are OUT OF
    SCOPE for the oracle (a stated limitation), not a place to accuse an engine of a bug."""
    if any(v != 0 and abs(v) < _MIN_NORMAL for v in xs):
        return False, "subnormal"
    thresh = Fraction(10) ** 308                     # ~ float64 max; compare as exact rationals (no float overflow)
    s1 = sum(abs(Fraction(v)) for v in xs); s2 = sum(Fraction(v) * Fraction(v) for v in xs)
    if s1 > thresh or s2 > thresh:
        return False, "overflow"
    return True, "ok"


def run():
    configs = []
    for kind in KINDS:
        for n in (1000, 10000, 50000):
            for seed in range(4):
                configs.append((kind, n, seed * 131 + n))
    print(f"hunting over {len(configs)} configs x 6 engines ...", file=sys.stderr)
    anomalies = []; tested = 0; skipped = 0; out_of_domain = {"subnormal": 0, "overflow": 0}
    for kind, n, seed in configs:
        xs = gen(kind, n, seed)
        if not all(math.isfinite(v) for v in xs):
            skipped += 1; continue
        ok, why = in_domain(xs)
        if not ok:
            out_of_domain[why] += 1; continue            # out of the bound's validity domain -> not an anomaly
        exS = oracle.exact_sum(xs); kap = oracle.kappa(xs)
        exV = oracle.exact_var(xs); kapV = oracle.kappa_var(xs)
        if kap is None or exV is None or kapV is None:
            skipped += 1; continue
        engines = {"postgres": lambda: m_pg(55432, xs), "mysql": lambda: m_mysql(xs),
                   "clickhouse": lambda: m_ch(xs), "duckdb": lambda: m_duck(xs),
                   "sqlite": lambda: m_sqlite(xs), "datafusion": lambda: m_df(xs)}
        for eng, fn in engines.items():
            try:
                res = fn(); tested += 1
                anomalies += anomalies_for(eng, res, xs, n, exS, kap, exV, kapV)
            except Exception as e:
                print(f"  !! {eng} {kind} n={n}: {type(e).__name__}: {str(e)[:70]}", file=sys.stderr)
        print(f"  {kind:<16} n={n:<6} kappa={float(kap):.2e} kappaV={float(kapV):.2e}", file=sys.stderr)

    json.dump({"n_configs": len(configs), "n_engine_tests": tested, "skipped": skipped,
               "out_of_domain": out_of_domain, "anomalies": anomalies},
              open(OUT + "/bughunt_phase3.json", "w"), indent=1)
    print(f"\n=== NC2-anomaly hunt: {tested} in-domain engine-tests ===")
    print(f"out-of-domain configs excluded (bound invalid): {out_of_domain}")
    print(f"IN-DOMAIN ANOMALIES FOUND: {len(anomalies)}")
    for a in anomalies[:40]:
        print(f"  {a['engine']}/{a['agg']} ({a['algo']}) n={a['n']} relerr={a['relerr']} > bound={a['bound']:.2e} obs={a['observed'][:30]}")
    if not anomalies:
        print("  -> zero anomalies: no engine exceeded its algorithm's bound at a decidable condition number.")
        print("  -> honest negative: consistent with the oracle being SOUND (bounds hold); no bug surfaced.")


if __name__ == "__main__":
    run()
