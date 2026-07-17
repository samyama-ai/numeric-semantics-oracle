"""Engine adapters. Every adapter must load float64 values BIT-EXACTLY and expose
plain + (where available) compensated summation. Bit-exact transport is verified by
`roundtrip_check` — if a value is mangled in transit, all error measurements are void.
"""
import sqlite3, struct
import duckdb, psycopg2, psycopg2.extras, pymysql, clickhouse_connect

PG = dict(host="127.0.0.1", port=55432, user="postgres", password="oracle", dbname="numeric")
MY = dict(host="127.0.0.1", port=53306, user="root", password="oracle", database="numeric")
CH = dict(host="127.0.0.1", port=58123, username="default", password="oracle")


class Engine:
    name = "?"
    def load(self, xs): raise NotImplementedError
    def sums(self):
        """-> dict: variant name -> float result of SUM(x)"""
        raise NotImplementedError
    def variances(self):
        """-> dict: variant name -> float result of population VARIANCE(x). Which algorithm
        the engine uses (one-pass vs Welford/two-pass) is RECOVERED by fitting the measured
        exponent p, not assumed. Vendor docs are corroboration, not the finding."""
        raise NotImplementedError
    def readback(self):
        """-> list[float] of stored values, in any order (for bit-exactness check)"""
        raise NotImplementedError
    def close(self): pass

    @staticmethod
    def _canon(v):
        """Canonicalize signed zero. SQLite and PostgreSQL normalize -0.0 -> +0.0 on storage;
        DuckDB and MySQL preserve it. This never changes a SUM, so it is a semantic divergence to
        REPORT, not a transport failure. Tracked separately by `signed_zero_preserved`."""
        return 0.0 if v == 0.0 else v

    def roundtrip_check(self, xs):
        key = lambda v: struct.pack(">d", self._canon(v))
        got = sorted(map(self._canon, self.readback()), key=key)
        want = sorted(map(self._canon, xs), key=key)
        if len(got) != len(want):
            return False, f"count {len(got)} != {len(want)}"
        for a, b in zip(got, want):
            if struct.pack(">d", a) != struct.pack(">d", b):
                return False, f"bit mismatch {a!r} != {b!r}"
        return True, "bit-exact (mod signed zero)"

    def signed_zero_preserved(self):
        """Load only -0.0 and report whether the engine gives it back with the sign bit intact."""
        return struct.pack(">d", self.readback()[0])[0] & 0x80 != 0


class _WelfordVarPop:
    """SQLite aggregate UDF: population variance by Welford's online algorithm. SQLite ships
    no native variance, so this is our KNOWN-STABLE (p=1) reference point -- documented, not a
    measurement of SQLite's own choice (it has none)."""
    def __init__(self): self.k = 0; self.mean = 0.0; self.m2 = 0.0
    def step(self, x):
        if x is None: return
        self.k += 1; d = x - self.mean; self.mean += d / self.k; self.m2 += d * (x - self.mean)
    def finalize(self): return None if self.k == 0 else self.m2 / self.k


class SQLite(Engine):
    name = "sqlite"
    def load(self, xs):
        self.c = sqlite3.connect(":memory:")
        self.c.create_aggregate("welford_var_pop", 1, _WelfordVarPop)
        self.c.execute("CREATE TABLE t(x REAL)")
        self.c.executemany("INSERT INTO t VALUES (?)", ((v,) for v in xs))
    def sums(self):
        # SQLite sum()/avg() use Kahan-Babuska-Neumaier since 3.43.0 (vendor-documented background)
        return {"kbn_default": self.c.execute("SELECT sum(x) FROM t").fetchone()[0]}
    def variances(self):
        # no native VARIANCE in SQLite; our Welford UDF is the stable reference (expect p~1)
        return {"welford_udf": self.c.execute("SELECT welford_var_pop(x) FROM t").fetchone()[0]}
    def readback(self):
        return [r[0] for r in self.c.execute("SELECT x FROM t")]
    def close(self): self.c.close()


class DuckDB(Engine):
    name = "duckdb"
    def __init__(self, threads=None): self.threads = threads
    def load(self, xs):
        self.c = duckdb.connect()
        if self.threads: self.c.execute(f"PRAGMA threads={self.threads}")
        self.c.execute("CREATE TABLE t(x DOUBLE)")
        self.c.executemany("INSERT INTO t VALUES (?)", [(v,) for v in xs])
    def sums(self):
        return {
            "plain": self.c.execute("SELECT sum(x) FROM t").fetchone()[0],
            "fsum_kahan": self.c.execute("SELECT fsum(x) FROM t").fetchone()[0],
        }
    def variances(self):
        return {"var_pop": self.c.execute("SELECT var_pop(x) FROM t").fetchone()[0]}
    def readback(self):
        return [r[0] for r in self.c.execute("SELECT x FROM t").fetchall()]
    def close(self): self.c.close()


class Postgres(Engine):
    name = "postgres"
    def load(self, xs):
        self.c = psycopg2.connect(**PG); self.c.autocommit = True
        cur = self.c.cursor()
        cur.execute("DROP TABLE IF EXISTS t; CREATE TABLE t(x double precision)")
        # psycopg2 adapts Python float via repr() -> shortest round-trip literal -> bit-exact
        psycopg2.extras.execute_values(cur, "INSERT INTO t VALUES %s",
                                       [(v,) for v in xs], page_size=10000)
    def sums(self):
        cur = self.c.cursor(); cur.execute("SELECT sum(x) FROM t")
        return {"plain": cur.fetchone()[0]}
    def variances(self):
        cur = self.c.cursor(); cur.execute("SELECT var_pop(x) FROM t")
        return {"var_pop": cur.fetchone()[0]}
    def readback(self):
        cur = self.c.cursor(); cur.execute("SELECT x FROM t"); return [r[0] for r in cur]
    def close(self): self.c.close()


class MySQL(Engine):
    name = "mysql"
    def load(self, xs):
        self.c = pymysql.connect(**MY, autocommit=True)
        cur = self.c.cursor()
        cur.execute("DROP TABLE IF EXISTS t"); cur.execute("CREATE TABLE t(x DOUBLE)")
        for i in range(0, len(xs), 10000):
            chunk = xs[i:i+10000]
            cur.executemany("INSERT INTO t VALUES (%s)", [(repr(v),) for v in chunk])
    def sums(self):
        cur = self.c.cursor(); cur.execute("SELECT sum(x) FROM t")
        return {"plain": float(cur.fetchone()[0])}
    def variances(self):
        cur = self.c.cursor(); cur.execute("SELECT var_pop(x) FROM t")
        v = cur.fetchone()[0]
        return {"var_pop": None if v is None else float(v)}
    def readback(self):
        cur = self.c.cursor(); cur.execute("SELECT x FROM t"); return [r[0] for r in cur]
    def close(self): self.c.close()


class ClickHouse(Engine):
    name = "clickhouse"
    def load(self, xs):
        self.c = clickhouse_connect.get_client(**CH)
        self.c.command("DROP TABLE IF EXISTS t")
        self.c.command("CREATE TABLE t(x Float64) ENGINE=Memory")
        self.c.insert("t", [[v] for v in xs], column_names=["x"])
    def sums(self):
        return {
            "plain": self.c.query("SELECT sum(x) FROM t").result_rows[0][0],
            "sumKahan": self.c.query("SELECT sumKahan(x) FROM t").result_rows[0][0],
        }
    def variances(self):
        return {"var_pop": self.c.query("SELECT varPop(x) FROM t").result_rows[0][0]}
    def readback(self):
        return [r[0] for r in self.c.query("SELECT x FROM t").result_rows]
    def close(self): pass


ALL = [SQLite, DuckDB, Postgres, MySQL, ClickHouse]
