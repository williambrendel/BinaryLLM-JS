# `core/` — the BinaryLLM library

All reusable logic lives here. Everything else (`benchmark/`, `scripts/`) only orchestrates and measures
these modules; `__tests__/core/` mirrors this tree with unit tests.

## The idea

A word's meaning-in-context is represented as a **sparse binary signature** — a set of bit indices, not a
dense embedding. Bits come from decomposing surrounding words into sub-word **parts** and placing them in
positional bands. Prediction and discrimination are then set operations and small linear/game-theoretic
optimizations over these bit-sets — no floating-point embedding matrix, no backprop.

## Subsystems

| folder | role |
|---|---|
| [`math/`](./math) | numeric primitives: **sparse-set** (sorted-array) bit operations + adaptive-threshold cuts |
| [`parts/`](./parts) | the sub-word **part** vocabulary: extraction (peel), decomposition, tokenization; sub-stages for `encoding/`, `generation/`, `normalization/`, `signature/` |
| [`signatures/`](./signatures) | encode a word / context into its **3F bit signature** (the shared input to everything below) |
| [`gate/`](./gate) | a signed-threshold **discriminator gate** for one target, evaluated without materializing the mask |
| [`phase1/`](./phase1) | **deployed** discriminative-part extractor — inhomogeneous StQP (Pelillo replicator) + AdaBoost + θ-tuned α-sum head |
| [`discovery/`](./discovery) | a **second** part-discovery formulation — synergy-clique GROW + sequential/parallel/SA fits + max-pool readout (cross-check for phase1) |
| [`predict/`](./predict) | next-token **prediction** via a CART tree of grown parts + smoothed leaf readout |

## Reading order

`math/sparse` (the primitives) → `parts` + `signatures` (build the bit-sets) → `phase1` (the deployed
extractor; see its README and `docs/phase1_spec.md`). `discovery/` and `predict/` are parallel research
formulations over the same signatures.
