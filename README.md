# A Condition-Aware Numeric Oracle for Differential Testing of SQL Aggregates

Reproduction artifact for the paper **"Bounded, Indeterminate, or a Bug: A Condition-Aware Oracle for
Differential Testing of SQL Aggregates"** (Mandarapu & Kunkunuru, 2026).

Differential database testing runs a query on two engines and calls a difference a bug. For
floating-point aggregates that is unsound: summation is not associative, so engines may legitimately
disagree. Practice patches this with a fixed relative epsilon; the leading oracles avoid floating-point
inputs entirely. This repository builds the oracle that practice lacks.

**Ground truth is the exact rational sum of the stored `float64` values** (a float *is* an exact
rational), so the yardstick is arithmetic — not another engine.

## One command

```bash
./run.sh          # brings up the engines, runs everything, writes results/
docker compose down
```

Everything runs on one laptop with free software. No cloud, no GPU, no paid resources — deliberately,
so that a reviewer can re-run it.

## What it measures

Each cross-engine discrepancy in `SUM`/`AVG` is classified against Higham's forward error bound:

| verdict | meaning |
|---|---|
| `exact` | matches the exact rational sum |
| `bounded` | explained by floating-point summation (error ≤ `γₙ·Σ|xᵢ|`) |
| `indeterminate` | the bound admits **any** discrepancy — no oracle can decide here |
| `ANOMALY` | exceeds what floating-point summation can produce → candidate bug, must be triaged |

With `κ = Σ|xᵢ| / |Σxᵢ|` the relative bound is exactly `B = γₙ·κ`, which yields two boundaries
(one-line consequences of Higham, Thm 4.1 — we claim no depth for them):

- **`κ* = 1/γₙ ≈ 1/(n·u)`** — the *testability boundary*. Beyond it, nothing can be concluded.
  It **shrinks as `n` grows**: aggregating more rows narrows the regime where testing can decide.
- **`κ_ε = ε/γₙ`** — the *epsilon crossover*. Below it a fixed `ε` is too lax (false negatives);
  above it, too strict (false positives). **A single `ε` is sound at exactly one condition number.**

## Findings

Engines: PostgreSQL 17, MySQL 8.4, ClickHouse 25.3 (Docker) + DuckDB 1.5.4, SQLite 3.45.1 (in-process).

- **Soundness gate passes: 0 anomalies in 70 cells.** No engine ever exceeded its bound.
- **DuckDB, PostgreSQL and MySQL return bit-identical plain sums** at every `κ`. Differential testing
  across those three therefore has **zero power** to detect a summation bug.
- **`κ*(compensated) = 1/2u` is independent of `n`.** Compensated aggregation removes the data-size
  penalty on testability — but see the correction below.
- **Real queries (TPC-H) sit at `κ ≈ 1`**, ten orders inside the decidable zone. There the sound bound
  is `6.7e-11`, so a `1e-9` epsilon is **15× too lax** and DuckDB's shipped `1%` rule is **~1.5e8× too
  lax**. On real data the hazard is **false negatives**, not the false alarms the field guards against.
- **Fuzzers live in the indeterminate regime.** Replicating SQLancer's value distribution, `κ` explodes
  only when `+MAX` and `−MAX` cancel *exactly*: with `k` and `m` copies, `κ = (k+m)/|k−m|`, undecidable
  iff `k = m ≥ 1`. Since `k,m ~ Poisson(n/600)`, undecidability is **non-monotone in `n`** — peaking at
  **19.6% at n=1000** and falling to 9.0% at n=10,000.
- **Correction, from our own data:** compensated aggregation does **not** rescue exactly-cancelling
  giants (`κ ≈ 1e306 ≫ κ*_c = 4.5e15`). Nothing helps there except not generating them.

**Honest negative: we found no engine bug.** Zero anomalies is evidence that the oracle is sound, not
that the engines are correct.

## Layout

```
src/oracle.py            exact ground truth, kappa, Higham bounds, the classifier
src/engines.py           five engine adapters; verifies bit-exact transport
src/run_map.py           the classification map + kappa* / kappa_eps
src/e2_realdata.py       condition numbers of real TPC-H aggregates
src/e3_fuzzer_kappa.py   kappa distribution of SQLancer-style random columns
src/figures.py           the paper's three figures
docker-compose.yml       PostgreSQL / MySQL / ClickHouse
```

## Prior work we build on and do not claim

The forward error bound and the condition number are standard (Higham, *Accuracy and Stability of
Numerical Algorithms*). Compensated summation is Kahan (1965) and Neumaier (1974). Reproducible
aggregation *within* an engine is Müller et al., ICDE 2018. Condition-number oracles for numerical
*programs* are Kulkarni & Panchekha, ARITH 2025 — which explicitly does not address databases. Which
summation algorithm each engine uses is documented by its own vendor. Our contribution is the oracle's
instantiation for SQL aggregates, the measurement, and the two-regime result.

## License

Code: MIT (see `LICENSE`). The accompanying paper is CC-BY-4.0.
