"""Phase-1 engine expansion: broaden the p-axis taxonomy by engine class.
  - QuestDB    : TIME-SERIES (pg-wire) -- the class whose core columns are near-constant timestamps
  - MonetDB    : classic COLUMNAR OLAP (pymonetdb)
  - DataFusion : Arrow COLUMNAR OLAP, in-process (datafusion pip), like DuckDB/SQLite
Each exposes population variance; the algorithm is RECOVERED by fitting the exponent p, not assumed.
Reuses the base Engine (bit-exact transport check) from engines.py.
"""
import struct, time
import psycopg2, psycopg2.extras, pymonetdb
import pyarrow as pa
from datafusion import SessionContext
from engines import Engine


class QuestDB(Engine):
    name = "questdb"
    _cfg = dict(host="127.0.0.1", port=58812, user="admin", password="quest", dbname="qdb")
    def load(self, xs):
        self.c = psycopg2.connect(**self._cfg); self.c.autocommit = True
        cur = self.c.cursor()
        cur.execute("DROP TABLE IF EXISTS t"); cur.execute("CREATE TABLE t(x DOUBLE)")
        psycopg2.extras.execute_values(cur, "INSERT INTO t VALUES %s",
                                       [(v,) for v in xs], page_size=10000)
        # QuestDB commits pg-wire inserts asynchronously; wait until all rows are visible
        for _ in range(50):
            cur.execute("SELECT count(*) FROM t")
            if cur.fetchone()[0] == len(xs): break
            time.sleep(0.2)
    def variances(self):
        cur = self.c.cursor(); cur.execute("SELECT var_pop(x) FROM t")
        v = cur.fetchone()[0]
        return {"var_pop": None if v is None else float(v)}
    def readback(self):
        cur = self.c.cursor(); cur.execute("SELECT x FROM t"); return [r[0] for r in cur]
    def close(self):
        try: self.c.close()
        except Exception: pass


class MonetDB(Engine):
    name = "monetdb"
    _cfg = dict(username="monetdb", password="monetdb", hostname="127.0.0.1", port=50000, database="monetdb")
    def load(self, xs):
        self.c = pymonetdb.connect(**self._cfg); self.c.set_autocommit(True)
        cur = self.c.cursor()
        cur.execute("DROP TABLE IF EXISTS t"); cur.execute("CREATE TABLE t(x DOUBLE)")
        cur.executemany("INSERT INTO t VALUES (%s)", [(float(v),) for v in xs])
    def variances(self):
        cur = self.c.cursor(); cur.execute("SELECT var_pop(x) FROM t")
        v = cur.fetchone()[0]
        return {"var_pop": None if v is None else float(v)}
    def readback(self):
        cur = self.c.cursor(); cur.execute("SELECT x FROM t"); return [r[0] for r in cur]
    def close(self):
        try: self.c.close()
        except Exception: pass


class DataFusion(Engine):
    name = "datafusion"
    def load(self, xs):
        self.ctx = SessionContext()
        self._xs = list(xs)
        t = pa.table({"x": pa.array(xs, type=pa.float64())})
        self.ctx.register_record_batches("t", [t.to_batches()])
    def variances(self):
        v = self.ctx.sql("SELECT var_pop(x) FROM t").collect()[0].column(0)[0].as_py()
        return {"var_pop": None if v is None else float(v)}
    def readback(self):
        return [r.as_py() for b in self.ctx.sql("SELECT x FROM t").collect() for r in b.column(0)]
    def close(self): pass


NEW = [QuestDB, MonetDB, DataFusion]
