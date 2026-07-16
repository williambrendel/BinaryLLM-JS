"use strict";

// ============================================================================
// benchmark/premiseCheck.js
//
// The spec's premise check (§5): one capped, best-first single-bit CART tree over
// the signature bits, swept over the regularizer s_min, read out by the raw
// smoothed leaf distribution, scored on held-out LOG-LOSS against the unigram
// floor and the bigram ceiling. The verdict (leaf beats unigram or not) decides
// whether the pool (§6+) is worth building.
//
//   GPT  (next-word): features = L ∪ C (causal), label = nextWord
//   BERT (cloze)    : features = L ∪ R (bidirectional), label = curWord
//
// Same within-corpus positional split caveat as predictBaselines.js.
//
// Usage: node benchmark/premiseCheck.js [dataset.bin] [K]
// ============================================================================

import fs from "node:fs";
import { readDataset, NO_LABEL } from "../src/core/signatures/signatureDataset.js";
import { buildUnigram, buildContextGram, buildCartTree, evaluate, evaluatePath, calibrationGate } from "../src/core/predict/index.js";

const unionSorted = (a, b) => {
  const out = []; let i = 0, j = 0;
  while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } }
  while (i < a.length) out.push(a[i++]);
  while (j < b.length) out.push(b[j++]);
  return out;
};

const path = process.argv[2] || "data/signatures/alice_sigs.bin";
const K = Number(process.argv[3] || 8192);
const mode = process.env.MODE === "generative" ? "generative" : "discriminative"; // axis 4
const D = Number(process.env.D || 6); // part width cap
const ds = readDataset(new Uint8Array(fs.readFileSync(path)));
const V = ds.vocab.length;
const recs = ds.records.filter((r) => r.scale === 0);
for (let i = 0; i < recs.length; i++) recs[i].prevWord = i > 0 && recs[i - 1].nextWord !== NO_LABEL ? recs[i - 1].curWord : NO_LABEL;
const cut = Math.floor(recs.length * 0.8);
const train = recs.slice(0, cut), held = recs.slice(cut);

const heads = [
  { name: "GPT  (next-word, [L∪C])", labelOf: (r) => r.nextWord, featuresOf: (r) => unionSorted(r.L, r.C), biKey: (r) => r.curWord },
  { name: "BERT (cloze, [L,R])", labelOf: (r) => r.curWord, featuresOf: (r) => unionSorted(r.L, r.R), biKey: (r) => r.prevWord },
];
const SMIN = [1, 10, 20, 30, 50, 100];

const w = process.stdout;
w.write(`# Premise check — ${path.split("/").pop()} (V=${V}, train=${train.length}, held=${held.length}, sentence, K=${K}, mode=${mode}, D=${D})\n`);
for (const h of heads) {
  const uni = buildUnigram(train, h.labelOf), bi = buildContextGram(train, h.labelOf, h.biKey);
  const eu = evaluate(uni, held, { V, labelOf: h.labelOf }), eb = evaluate(bi, held, { V, labelOf: h.labelOf });
  w.write(`\n## ${h.name}   (n=${eu.n})\n`);
  w.write(`bars:  unigram ll=${eu.ll.toFixed(3)}   bigram ll=${eb.ll.toFixed(3)} (ceiling)\n`);
  w.write(`| s_min | leaves | ll_leaf | ll_path | acc_leaf % | acc_path % | leaf gate |\n|---|---|---|---|---|---|---|\n`);
  let bestLeaf = Infinity, bestPath = Infinity;
  for (const sMin of SMIN) {
    const tree = buildCartTree(train, { featuresOf: h.featuresOf, labelOf: h.labelOf, sMin, mode, D, K });
    const e = evaluate(tree, held, { V, labelOf: h.labelOf });
    const p = evaluatePath(tree, held, { V, labelOf: h.labelOf });
    bestLeaf = Math.min(bestLeaf, e.ll); bestPath = Math.min(bestPath, p.ll);
    w.write(`| ${sMin} | ${tree.leaves} | ${e.ll.toFixed(3)} | ${p.ll.toFixed(3)} | ${(e.acc * 100).toFixed(1)} | ${(p.acc * 100).toFixed(1)} | ${calibrationGate(e.ll, eu.ll) ? "PASS" : "FAIL"} |\n`);
  }
  // §5.5 discriminator: leaf log-loss decides overfit-vs-not; the PATH decoder
  // decides leaf-weak-vs-signal-poor.
  const leafBeats = bestLeaf < eu.ll - 1e-9, pathBeats = bestPath < eu.ll - 1e-9;
  const verdict = leafBeats
    ? "leaf BEATS unigram → signal in the leaf (H_overfit) → build the pool (§6)"
    : pathBeats
      ? "leaf ~ unigram but PATH DECODER beats it → H_leaf-weak → keep the path decoder as the readout"
      : "even the path decoder ~ unigram → H_signal-poor → reconsider the representation (radius, pooling, encoder)";
  w.write(`verdict: best ll_leaf=${bestLeaf.toFixed(3)}, best ll_path=${bestPath.toFixed(3)} vs unigram ${eu.ll.toFixed(3)}\n         → ${verdict}\n`);
}
