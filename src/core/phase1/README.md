# `core/phase1/` — the deployed part extractor (StQP + AdaBoost)

Given a target word's positive contexts `A` and a corpus, discover **weak m-of-n parts** over the signature
bits that together discriminate the word from confusable negatives, and combine them into a classifier head.
This is the deployed pipeline. Full design: [`docs/phase1_spec.md`](../../../docs/phase1_spec.md); measured
results: [`docs/phase1_report.md`](../../../docs/phase1_report.md).

## Motivation

The blunt union of a word's positive signatures (`⋁A`) has perfect recall but no precision. The move is to
**break it into pieces** — each an m-of-n bit gate that is individually *weak* (recall > 0.5 on confusable
negatives) but cumulatively covering — and combine them with an α-weighted vote. Parts are found as **dominant
sets** of a signed bit-affinity graph (attractive PMI co-occurrence minus repulsive, plus a unary log-odds
tilt) via Pelillo replicator dynamics, then boosted by AdaBoost sample reweighting. False-positive control
lives in the θ-tuned head, not in the extractor.

## Pipeline (one `fitClass` call)

`buildNegSet` (confusable negatives + frozen stats) → per round: `buildAffinity` (u, M) → `replicate`
(dominant set) → grow to weak-recall → `mfit` (m-of-n gate) → AdaBoost reweight → `makeHead` (α-sum) +
validation θ-tune.

## Contents

| file | § | role |
|---|---|---|
| `negSet.js` | §2 | confusable negative set + frozen `p⁻`/joint statistics |
| `affinity.js` | §1 | `w`-weighted unary log-odds `u` + PMI-difference edge affinity `M`, rebuilt each round |
| `replicator.js` | §4–5 | replicator dynamics (`exp` default / `dc`) on `π=u+Mx−ρx`; support-stability early-stop |
| `mfit.js` | §6 | m-of-n gate fit (precision-max at a weighted-recall floor) |
| `boost.js` | §7 | AdaBoost reweighting loop producing the part ensemble `G` |
| `head.js` | §7.2 | α-weighted sum head `S=Σα_k·1[\|Q_k∧x\|≥m_k] > θ` |
| `fit.js` | §8 | **deployed entry** `fitClass` — split → boost → val early-stop → θ-tune |
| `parallelFit.js` + `fitWorker.js` | — | **parallel per-word extraction** over a worker pool — featurize once, share the read-only negPool zero-copy via `SharedArrayBuffer`, fan `fitClass` out across cores (deterministic ⇒ identical to sequential) |

Unit tests: `__tests__/core/phase1/*.test.js`. Evaluation driver: `benchmark/phase1/phase1.js`.
Parallel verification + speedup: `benchmark/phase1/parallelBench.js`.
