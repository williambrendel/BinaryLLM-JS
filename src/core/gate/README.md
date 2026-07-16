# `core/gate/` — the discriminator gate

A single-target **discriminator gate**: given a context signature `x` (a sorted sparse bit-set), score how
strongly it indicates one target word.

## Motivation

Downstream stages (extraction, prediction) need to ask "does this context belong to word `w`?" cheaply and
consistently. The gate is that primitive: a **signed-threshold** score under the ADAPTED mask, evaluated
directly on the sparse bit-set **without ever materializing the mask** (which would be dense over the full
bit universe). It's the shared, mask-free scoring contract the rest of the pipeline builds on.

## Contents

| file | role |
|---|---|
| `adaptedGate.js` | signed-threshold discriminator gate under the ADAPTED mask, evaluated mask-free on a sparse `x` |
