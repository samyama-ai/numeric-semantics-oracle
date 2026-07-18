"""Phase 2: the covariance/correlation/regression family (covar_pop, corr, regr_slope, stddev).
Same discipline as variance: exact rational ground truth, sweep the conditioning by additive offset,
fit each engine's exponent p. Plus a qualitative-failure table at high conditioning (the money shot:
one-pass engines return stddev=0, corr=NaN, wrong-sign regr_slope on ordinary large-offset columns).

Engines with the SQL covariance family: PostgreSQL, DuckDB, ClickHouse, DataFusion, QuestDB.
MySQL has no covar_pop/corr/regr; SQLite has none natively. Effect-agnostic.
"""
import sys, json, math
import numpy as np
import psycopg2, psycopg2.extras, clickhouse_connect, duckdb
from datafusion import SessionContext
import pyarrow as pa
from fractions import Fraction
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from run_variance import fit_p, assign_algo
OUT = __file__.rsplit("/", 2)[0] + "/results"


def make_pair(n, offset, seed):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n); u -= u.mean()
    w = rng.standard_normal(n); w -= w.mean()
    v = 0.6 * u + w; v -= v.mean()          # covariance well-defined and nonzero
    return (offset + u).tolist(), (offset + v).tolist()


def exact_covar(xs, ys):
    fx = [Fraction(a) for a in xs]; fy = [Fraction(b) for b in ys]; n = len(fx)
    mx = sum(fx) / n; my = sum(fy) / n
    return sum((a - mx) * (b - my) for a, b in zip(fx, fy)) / n, fx, fy, mx, my


def kappa_cov(xs, ys):
    """|sum x_i y_i| / |sum (x_i-xbar)(y_i-ybar)| -- raw cross-moment over centred; large when
    the columns are near-constant (offsets dominate). Monotone in the additive offset."""
    fx = [Fraction(a) for a in xs]; fy = [Fraction(b) for b in ys]; n = len(fx)
    mx = sum(fx) / n; my = sum(fy) / n
    raw = abs(sum(a * b for a, b in zip(fx, fy)))
    cen = abs(sum((a - mx) * (b - my) for a, b in zip(fx, fy)))
    return None if cen == 0 else math.sqrt(float(raw / cen))


def relerr(v, ex):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return v
    return float(abs(Fraction(v) - Fraction(ex)) / abs(Fraction(ex)))


# --- engine adapters (covariance family) ---
def pg():
    c = psycopg2.connect(host="127.0.0.1", port=55432, user="postgres", password="oracle", dbname="numeric"); c.autocommit = True
    return c
def load_pg(c, xs, ys):
    cur = c.cursor(); cur.execute("DROP TABLE IF EXISTS t;CREATE TABLE t(x float8,y float8)")
    psycopg2.extras.execute_values(cur, "INSERT INTO t VALUES %s", list(zip(xs, ys)), page_size=10000); return cur
def qdb():
    c = psycopg2.connect(host="127.0.0.1", port=58812, user="admin", password="quest", dbname="qdb"); c.autocommit = True
    return c

ENGINES = ["postgres", "duckdb", "clickhouse", "datafusion", "questdb"]

def measure(engine, xs, ys, agg):
    """Return the engine's value for one aggregate ('covar','corr','slope','std')."""
    try:
        if engine in ("postgres", "questdb"):
            c = pg() if engine == "postgres" else qdb()
            cur = c.cursor(); cur.execute("DROP TABLE IF EXISTS t"); cur.execute("CREATE TABLE t(x float8,y float8)" if engine=="postgres" else "CREATE TABLE t(x DOUBLE,y DOUBLE)")
            psycopg2.extras.execute_values(cur, "INSERT INTO t VALUES %s", list(zip(xs, ys)), page_size=5000)
            if engine == "questdb":
                import time
                for _ in range(50):
                    cur.execute("SELECT count(*) FROM t");
                    if cur.fetchone()[0] == len(xs): break
                    time.sleep(0.2)
            q = {"covar":"SELECT covar_pop(y,x) FROM t","corr":"SELECT corr(y,x) FROM t",
                 "slope":"SELECT regr_slope(y,x) FROM t","std":"SELECT stddev_pop(x) FROM t"}[agg]
            cur.execute(q); r = cur.fetchone()[0]; c.close(); return r
        if engine == "duckdb":
            d = duckdb.connect(); d.execute("CREATE TABLE t(x DOUBLE,y DOUBLE)"); d.executemany("INSERT INTO t VALUES(?,?)", list(zip(xs, ys)))
            q = {"covar":"SELECT covar_pop(y,x) FROM t","corr":"SELECT corr(y,x) FROM t",
                 "slope":"SELECT regr_slope(y,x) FROM t","std":"SELECT stddev_pop(x) FROM t"}[agg]
            return d.execute(q).fetchone()[0]
        if engine == "clickhouse":
            ch = clickhouse_connect.get_client(host="127.0.0.1", port=58123, username="default", password="oracle")
            ch.command("DROP TABLE IF EXISTS t"); ch.command("CREATE TABLE t(x Float64,y Float64) ENGINE=Memory"); ch.insert("t", list(zip(xs, ys)), column_names=["x","y"])
            q = {"covar":"SELECT covarPop(x,y) FROM t","corr":"SELECT corr(x,y) FROM t",
                 "slope":"SELECT simpleLinearRegression(x,y).1 FROM t","std":"SELECT stddevPop(x) FROM t"}[agg]
            return ch.query(q).result_rows[0][0]
        if engine == "datafusion":
            ctx = SessionContext(); t = pa.table({"x":pa.array(xs,pa.float64()),"y":pa.array(ys,pa.float64())}); ctx.register_record_batches("t",[t.to_batches()])
            q = {"covar":"SELECT covar_pop(y,x) FROM t","corr":"SELECT corr(y,x) FROM t",
                 "slope":"SELECT regr_slope(y,x) FROM t","std":"SELECT stddev_pop(x) FROM t"}[agg]
            return ctx.sql(q).collect()[0].column(0)[0].as_py()
    except Exception as e:
        return "ERR:" + str(e)[:40]


def run():
    n = 4000
    OFFS = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    fitrows = []
    for off in OFFS:
        xs, ys = make_pair(n, float(off), seed=(hash(off) & 0xffff) + 7)
        exCov, *_ = exact_covar(xs, ys); kc = kappa_cov(xs, ys)
        for eng in ENGINES:
            v = measure(eng, xs, ys, "covar")
            r = relerr(v, exCov)
            if isinstance(r, float) and not math.isnan(r):
                fitrows.append(dict(engine=eng, kappa_cov=kc, relerr=r))
        print(f"  covar sweep off={off:<8g} kappa_cov={kc:.3e}", file=sys.stderr)

    fits = {}
    for eng in ENGINES:
        pairs = [(r["kappa_cov"], r["relerr"]) for r in fitrows if r["engine"] == eng]
        p, npts = fit_p(pairs); fits[eng] = dict(p=p, n_fit=npts, algo=assign_algo(p))

    # qualitative failure table at high conditioning (offset 1e8)
    xs, ys = make_pair(n, 1e8, seed=7)
    exCov, fx, fy, mx, my = exact_covar(xs, ys)
    Sxy = sum((a-mx)*(b-my) for a,b in zip(fx,fy)); Sxx = sum((a-mx)**2 for a in fx); Syy = sum((b-my)**2 for b in fy)
    ex = {"covar": float(exCov), "corr": float(Sxy)/(float(Sxx)*float(Syy))**0.5,
          "slope": float(Sxy/Sxx), "std": math.sqrt(float(Sxx/n))}
    qual = {}
    for eng in ENGINES:
        qual[eng] = {}
        for agg in ("covar","corr","slope","std"):
            v = measure(eng, xs, ys, agg)
            qual[eng][agg] = {"value": (v if isinstance(v,(int,float)) else str(v)), "relerr": relerr(v, ex[{"covar":"covar","corr":"corr","slope":"slope","std":"std"}[agg]])}

    json.dump({"fits": fits, "exact_at_1e8": ex, "qualitative": qual}, open(OUT+"/covar_phase2.json","w"), indent=1)

    print("\n=== covariance one-pass exponent (fitted p; near 2 = one-pass, near 1 = stable) ===")
    for eng in ENGINES:
        f = fits[eng]; ps = f"{f['p']:.2f}" if f['p'] is not None else "n/a"
        print(f"  {eng:<11} p={ps:<6} -> {'ONE-PASS' if f['algo']=='var_onepass' else 'stable'}")
    print("\n=== qualitative failure at offset 1e8 (kappa_cov ~ 1e8) ===")
    print(f"{'engine':<11}{'covar relerr':>14}{'corr':>10}{'regr_slope':>14}{'stddev':>10}")
    for eng in ENGINES:
        q = qual[eng]
        def cell(a, kind):
            r = q[a]["relerr"]; val = q[a]["value"]
            if isinstance(r, float) and math.isnan(r): return "NaN"
            if isinstance(r, float):
                if kind=="slope" and isinstance(val,(int,float)) and val*ex["slope"]<0: return f"{r:.0e}(SIGN)"
                if kind=="std" and isinstance(val,(int,float)) and val==0: return "0(=const)"
                return f"{r:.0e}"
            return str(val)[:9]
        print(f"{eng:<11}{cell('covar','covar'):>14}{cell('corr','corr'):>10}{cell('slope','slope'):>14}{cell('std','std'):>10}")


if __name__ == "__main__":
    run()
