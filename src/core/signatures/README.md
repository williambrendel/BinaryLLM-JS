# `core/signatures/` — context → bit signature

Encodes a word or a context into its **3F signature**: a list of positional bands, each a sparse sorted
bit-set of part IDs. This is the shared input representation for `phase1/`, `discovery/`, and `predict/`.

## Motivation

The "3F canon" `[L | L1∪L2 | C]` places the parts of surrounding words into three **distance bands** —
adjacent (F2), near (F1), far (F0) — each spanning the full `F`-dim part space, so a full signature lives in
`3F` bits. Banding lets the model weight *how close* a context word is, and the sparse-set form keeps it cheap.
The word encoder produces the "current" (C) band — the bag of part IDs from decomposing the word itself.

## Contents

| file | role |
|---|---|
| `wordEncoder.js` | encode one word → its sorted-unique set of part IDs (the current-word `C` band) |
| `encoder.js` | encode the Word tokens of a token stream → full 3F signatures (list of bands) |
| `streamSignatures.js` | stream a corpus into bounded signature-record chunks (never materialized whole) |
| `signatureDataset.js` | build/serialize a signature dataset for downstream training/evaluation |
