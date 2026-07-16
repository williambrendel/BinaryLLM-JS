# `core/math/sparse/` — sparse-set (bit-vector) operations

Set operations over **sparse sets**: sorted, strictly-increasing arrays of non-negative integers — the sparse
representation of a bit vector (index `i` present ⇔ bit `i` set).

## Motivation

Every signature is a sparse bit-set of a few hundred indices out of a ~100k-bit universe (≈0.1–2% density).
Materializing dense bit vectors would be 100× the memory and work; instead each operation is a **linear merge
of two sorted arrays** — O(|A|+|B|), no allocation of the full universe. The `*Count` variants return just the
cardinality without building the result array, which is the hot path for recall/precision/FP evaluation
(`|Q ∧ y| ≥ m`).

## Contents

| file | operation |
|---|---|
| `and.js` / `andCount.js` | intersection `A ∩ B` / its size |
| `or.js` / `orCount.js` | union `A ∪ B` / its size |
| `xor.js` / `xorCount.js` | symmetric difference `A △ B` / its size |
| `andNot.js` / `andNotCount.js` | difference `A ∖ B` / its size |
| `jaccard.js` | `\|A ∩ B\| / \|A ∪ B\|` |

All are pure, allocation-light, and assume both inputs are sorted & unique. Unit tests mirror each file in
`__tests__/core/math/sparse/`.
