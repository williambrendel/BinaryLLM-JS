# `core/math/` — numeric primitives

Low-level, dependency-free numeric building blocks used throughout the pipeline.

## Motivation

Signatures are **sparse binary vectors** represented as sorted, strictly-increasing integer arrays. Comparing
and combining them (recall, precision, co-occurrence, Jaccard) reduces to **sorted-set operations** — which
are O(|A|+|B|) merges, never O(n) over the full bit universe. Keeping these as tiny, allocation-light,
well-tested primitives is what makes the whole representation cheap.

## Contents

| file | role |
|---|---|
| [`sparse/`](./sparse) | the sorted-array set operations (`and`/`or`/`xor`/`andNot` + their `*Count` variants + `jaccard`) |
| `adaptiveThreshold.js` | adaptive-cut primitives for sorted-**descending** sequences (elbow → scaled-median cascade) — used to pick "where the signal stops" without a fixed cutoff |
