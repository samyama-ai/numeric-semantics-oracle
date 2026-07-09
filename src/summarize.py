"""Principle 0: generate the paper's numbers from the artifact so a stale number cannot exist.

Reads results/map.json and emits results/summary.json with NAMED keys for every quantity quoted in
prose. The paper's claims.yaml points at these keys; tests/claim_evidence_check.py resolves them.

Nothing here recomputes anything — it only names what the confirmatory run already measured.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
m = json.load(open(os.path.join(RES, "map.json")))


def relerr(n, kappa, engine, variant):
    for r in m["rows"]:
        if (r["n"] == n and abs(r["target_kappa"] - kappa) / kappa < 1e-6
                and r["engine"] == engine and r["variant"] == variant):
            return r["relerr"]
    raise KeyError(f"no row for n={n} kappa={kappa} {engine}/{variant}")


summary = {
    "n_cells": len(m["rows"]),
    "n_anomalies": len(m["anomalies"]),
    "kappa_star_plain_10k": m["boundaries"]["10000"]["kappa_star_plain"],
    "kappa_star_plain_100k": m["boundaries"]["100000"]["kappa_star_plain"],
    "kappa_star_compensated": m["boundaries"]["100000"]["kappa_star_compensated"],
    "kappa_eps_1e-9_10k": m["boundaries"]["10000"]["kappa_eps_plain[1e-9]"],
    "kappa_eps_1e-9_100k": m["boundaries"]["100000"]["kappa_eps_plain[1e-9]"],
    "kappa_eps_1pct_10k": m["boundaries"]["10000"]["kappa_eps_plain[duckdb_1pct]"],
    # the plain-summation error at n=10k, quoted in the bit-identity sentence
    "relerr_plain_10k_kappa1": relerr(10000, 1, "duckdb", "plain"),
    "relerr_plain_10k_kappa1e4": relerr(10000, 1e4, "duckdb", "plain"),
    "relerr_plain_10k_kappa1e8": relerr(10000, 1e8, "duckdb", "plain"),
    "relerr_plain_10k_kappa1e12": relerr(10000, 1e12, "duckdb", "plain"),
}

out = os.path.join(RES, "summary.json")
json.dump(summary, open(out, "w"), indent=1)
print(f"wrote {out}")
for k, v in summary.items():
    print(f"  {k:32} {v}")
