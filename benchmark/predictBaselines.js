"use strict";

// ============================================================================
// benchmark/predictBaselines.js
//
// The FIRST rung of the learning harness: n-gram baselines over a SIG2 signature
// dataset, scored on held-out log-loss (the spec's decision metric) with the
// calibration gate. This is the headroom ceiling every later tree/ablation row is
// read against — a model is only interesting if it beats these on log-loss.
//
//   unigram  P(label)                 — the calibration floor (gate bar)
//   bigram   P(label | ctx1)          — one-word context
//   trigram  P(label | ctx2, ctx1)    — two-word context
//
// Heads (the pooling/label choice is upstream config):
//   GPT  (next-word): label = nextWord, ctx = current / previous+current word
//   BERT (cloze)    : label = curWord,  ctx = previous / previous+next word (both sides)
//
// NOTE: with a single corpus this uses a within-corpus positional held-out split
// (last 20%), NOT the spec's cross-corpus split — a first number, not the verdict.
// Scope is filtered to sentences (scale 0) so a token is not duplicated across
// scales in the split.
//
// Usage: node benchmark/predictBaselines.js [dataset.bin]   (default: alice_sigs.bin)
// ============================================================================

import fs from "node:fs";
import { readDataset, NO_LABEL } from "../src/core/signatures/signatureDataset.js";
import { buildUnigram, buildContextGram, evaluate, calibrationGate } from "../src/core/predict/index.js";

const path = process.argv[2] || "data/signatures/alice_sigs.bin";
const ds = readDataset(new Uint8Array(fs.readFileSync(path)));
const V = ds.vocab.length;

// Sentence scope only (avoid per-token duplication across scales in the split).
const recs = ds.records.filter((r) => r.scale === 0);
// Attach previous word, reset at scope boundaries (a record whose predecessor had
// no next word starts a new scope, so its previous word is undefined).
for (let i = 0; i < recs.length; i++) {
  recs[i].prevWord = i > 0 && recs[i - 1].nextWord !== NO_LABEL ? recs[i - 1].curWord : NO_LABEL;
}

const cut = Math.floor(recs.length * 0.8);
const train = recs.slice(0, cut), held = recs.slice(cut);

const heads = [
  { name: "GPT  (next-word)", labelOf: (r) => r.nextWord, bi: (r) => r.curWord, tri: (r) => r.prevWord * 100003 + r.curWord },
  { name: "BERT (cloze word)", labelOf: (r) => r.curWord, bi: (r) => r.prevWord, tri: (r) => r.prevWord * 100003 + r.nextWord },
];

const w = process.stdout;
w.write(`# Baseline table — ${path.split("/").pop()}  (V=${V}, train=${train.length}, held=${held.length}, scope=sentence)\n`);
w.write(`# held-out = last 20% by position (within-corpus; not cross-corpus)\n`);
for (const h of heads) {
  const uni = buildUnigram(train, h.labelOf);
  const bi = buildContextGram(train, h.labelOf, h.bi);
  const tri = buildContextGram(train, h.labelOf, h.tri);
  const eu = evaluate(uni, held, { V, labelOf: h.labelOf });
  const eb = evaluate(bi, held, { V, labelOf: h.labelOf });
  const et = evaluate(tri, held, { V, labelOf: h.labelOf });
  w.write(`\n## ${h.name}   (n=${eu.n})\n`);
  w.write(`| model | ll (nats) | acc % | vs unigram |\n|---|---|---|---|\n`);
  w.write(`| unigram | ${eu.ll.toFixed(3)} | ${(eu.acc * 100).toFixed(1)} | — (bar) |\n`);
  w.write(`| bigram  | ${eb.ll.toFixed(3)} | ${(eb.acc * 100).toFixed(1)} | ${calibrationGate(eb.ll, eu.ll) ? "PASS " : "FAIL "}(${(eb.ll - eu.ll).toFixed(3)}) |\n`);
  w.write(`| trigram | ${et.ll.toFixed(3)} | ${(et.acc * 100).toFixed(1)} | ${calibrationGate(et.ll, eu.ll) ? "PASS " : "FAIL "}(${(et.ll - eu.ll).toFixed(3)}) |\n`);
}
