"use strict";

// ============================================================================
// benchmark/purityProbe.js
//
// Why are there no pure leaves — tree capacity, or no split possible? Sweep s_min
// and report, per tree: terminal reasons, % pure leaves, and the KEY number —
// distinct labels per leaf (size / #distinct). If a leaf of 30 samples holds ~30
// distinct words, no split can purify it: it's a bag of near-unique words (Zipf),
// not a capacity limit. Lowering s_min only buys purity by memorizing singletons.
//
// Usage: node benchmark/purityProbe.js [dataset.bin]   (CHUNK, DMAX, ALPHA via env)
// ============================================================================

import fs from "node:fs";
import { readDataset } from "../src/core/signatures/signatureDataset.js";
import { buildCartTree } from "../src/core/predict/index.js";

const unionSorted = (a, b) => {
  const out = []; let i = 0, j = 0;
  while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } }
  while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out;
};

const path = process.argv[2] || "data/signatures/wiki_valid_sigs.bin";
const CHUNK = Number(process.env.CHUNK || 20000);
const Dmax = Number(process.env.DMAX || 25);
const alpha = Number(process.env.ALPHA ?? 1);
const gainType = process.env.GAIN || "hb";
const M = Number(process.env.MCAND || 48);

const ds = readDataset(new Uint8Array(fs.readFileSync(path)));
const chunk = ds.records.filter((r) => r.scale === 0).slice(0, CHUNK);
const featuresOf = (r) => unionSorted(r.L, r.C), labelOf = (r) => r.nextWord;

// walk leaves → sizes and distinct-label counts
const leafStats = (node, acc) => {
  if (node.part === undefined) {
    const size = node.hist.total, distinct = node.hist.counts.size;
    let top = 0, H = 0; for (const c of node.hist.counts.values()) { if (c > top) top = c; const p = c / size; H -= p * Math.log(p); }
    acc.push({ size, distinct, top, H });
    return;
  }
  leafStats(node.present, acc); leafStats(node.absent, acc);
};

// Root label entropy (nats) of the chunk — the ceiling total IG can reduce.
const rootCounts = new Map(); let rootN = 0;
for (const r of chunk) { const l = labelOf(r); if (l === undefined || l === 0xffffffff) continue; rootCounts.set(l, (rootCounts.get(l) || 0) + 1); rootN++; }
let Hroot = 0; for (const c of rootCounts.values()) { const p = c / rootN; Hroot -= p * Math.log(p); }

process.stdout.write(`# purityProbe — ${path.split("/").pop()} | chunk=${chunk.length} Dmax=${Dmax} α=${alpha} gain=${gainType} | Hroot=${Hroot.toFixed(3)}\n`);
process.stdout.write(`sMin  leaves depth   pure%   smin  nosplit  meanLeaf  size/distinct  modalPur%  leafEntropy  totalIG\n`);
for (const sMin of [1, 5, 20, 50]) {
  const tree = buildCartTree(chunk, { featuresOf, labelOf, sMin, alpha, D: 3, Dmax, gainType, M });
  const depth = (function d(n) { return n.part === undefined ? 0 : 1 + Math.max(d(n.present), d(n.absent)); })(tree.root);
  const L = []; leafStats(tree.root, L);
  const pure = L.filter((l) => l.distinct === 1).length;
  const mean = (f) => L.reduce((s, l) => s + f(l), 0) / L.length;
  const tr = tree.terminalReasons;
  const totSize = L.reduce((s, l) => s + l.size, 0);
  const wLeafH = L.reduce((s, l) => s + l.size * l.H, 0) / totSize; // weighted mean leaf entropy (nats)
  process.stdout.write(
    `${String(sMin).padStart(4)}  ${String(tree.leaves).padStart(6)} ${String(depth).padStart(4)}  ${(100 * pure / L.length).toFixed(1).padStart(6)}  ${String(tr.smin).padStart(5)} ${String(tr.nosplit).padStart(7)}  ${mean((l) => l.size).toFixed(1).padStart(8)}  ${(mean((l) => l.size) / mean((l) => l.distinct)).toFixed(2).padStart(13)}  ${(100 * mean((l) => l.top / l.size)).toFixed(1).padStart(9)}  ${wLeafH.toFixed(3).padStart(11)}  ${(Hroot - wLeafH).toFixed(3).padStart(7)}\n`,
  );
}
