# `core/discovery/` — synergy-clique part discovery (the second formulation)

A **second, independent** formulation for discovering discriminative parts (the cross-check for `phase1/`).
Where `phase1/` builds parts as dominant sets of a *co-occurrence* affinity, `discovery/` builds them as
cliques of a **synergy** graph — bits are joined when together they discriminate *better than the sum of their
parts* — then scores contexts by a max-pool over the resulting gate set.

## Motivation

Agreement between two formulations that share nothing but the input signatures is strong evidence a discovered
part is real, not an artifact of one optimizer. This branch also explores the questions `phase1/` defers:
building an explicit **negative channel** per part, and whether the part count `K` scales with data (Heaps).

## Pipeline

`hardNegatives` (coarse gate + hard negatives) → `growClique`/`synergyGain`/`pruneClique` (assemble & prune a
discriminative clique) → `thresholdFit` + `peelNegative` (per-gate threshold + negative channel) → a fit
strategy (`fitParallel` default / `discoverSeq` / `fitSA`) → `maxReadout` + `acceptance`/`metrics` (score & evaluate).

## Contents

| file | § | role |
|---|---|---|
| `growClique.js` | §5.2 | GROW — greedily assemble a discriminative clique `Q` on the **synergy** graph |
| `synergyGain.js` | §5 | the m-of-n information-gain objective + its synergy increment |
| `pruneClique.js` | — | PRUNE — deterministic per-part bit-drop while holding `Q` fixed |
| `hardNegatives.js` | §3 | the coarse gate `g0`, its threshold, and the hard negatives `Ā_h` |
| `peelNegative.js` | §5.3 | PEEL_NEG — build the negative channel `p⁻_k` for a clique |
| `thresholdFit.js` | — | fit the per-gate decision threshold `t_k = argmax_t GAIN(score>t)` |
| `discoverSeq.js` | §6a | sequential, λ-driven discovery (K emergent) — fallback / cross-check |
| `fitParallel.js` | §6b | parallel, fixed-K discovery (the default) |
| `fitSA.js` | §6d | simulated annealing over the bit→part grouping |
| `baselines.js` | §6c | naive baselines the discovery methods must beat |
| `maxReadout.js` | §7 | final scorer — MAX POOL over the gate set on margins |
| `acceptance.js` / `metrics.js` | §9 | evaluation through the max-pool + primary metrics |
| `heaps.js` | §9.2 | Heaps analysis — does `K` scale with the number of contexts? |

See memory note `discovery-real-corpus-findings` for the empirical results.
