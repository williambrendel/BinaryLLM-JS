"use strict";

// ============================================================================
// benchmark/mechanismProbe.js
//
// Why is r5's per-split IG higher yet its tree worse? Per internal split, measure
// TWO things from the stored histograms:
//   IG          — average entropy reduction (what the grower maximizes)
//   modalGain   — does a CHILD become purer than the parent? = max(child modal) −
//                 parent modal. A peel isolates a class → modalGain > 0. A bisect
//                 splits every class across both sides → modalGain ≈ 0.
// If r5 has higher IG but ~zero modalGain, IG is buying "average separation" that
// never isolates a class, so purity never compounds. That's the local≠global.
//
// Usage: node benchmark/mechanismProbe.js [maxRecords]   (SMIN, DMAX via env)
// ============================================================================

import fs from "node:fs";
import { readDataset } from "../src/core/signatures/signatureDataset.js";
import { buildCartTree } from "../src/core/predict/index.js";

const unionSorted = (a, b) => { const out = []; let i = 0, j = 0; while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } } while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out; };
const featOf = (r) => unionSorted(r.L, r.C), labOf = (r) => r.nextWord;
const N = Number(process.argv[2] || 20000);
const SMIN = Number(process.env.SMIN || 20), DMAX = Number(process.env.DMAX || 25);
const load = (f) => readDataset(new Uint8Array(fs.readFileSync(`data/signatures/${f}.bin`))).records.filter((r) => r.scale === 0).slice(0, N);

const H = (h) => { let e = 0; for (const c of h.counts.values()) { const p = c / h.total; e -= p * Math.log(p); } return e; };
const modal = (h) => { let m = 0; for (const c of h.counts.values()) if (c > m) m = c; return m / h.total; };

const analyze = (records, name) => {
  const tree = buildCartTree(records, { featuresOf: featOf, labelOf: labOf, sMin: SMIN, D: 3, Dmax: DMAX });
  const splits = [];
  const walk = (n) => {
    if (n.part === undefined) return;
    const w = n.present.hist.total / n.hist.total, w2 = n.absent.hist.total / n.hist.total;
    const ig = H(n.hist) - w * H(n.present.hist) - w2 * H(n.absent.hist);
    const modalGain = Math.max(modal(n.present.hist), modal(n.absent.hist)) - modal(n.hist);
    const bal = 2 * Math.min(w, w2); // 1 = 50/50, →0 = lopsided peel
    splits.push({ ig, modalGain, bal, size: n.hist.total });
    walk(n.present); walk(n.absent);
  };
  walk(tree.root);
  const mean = (f) => splits.reduce((s, x) => s + f(x), 0) / splits.length;
  const wsum = splits.reduce((s, x) => s + x.size, 0);
  const wmean = (f) => splits.reduce((s, x) => s + x.size * f(x), 0) / wsum;
  // fraction of splits that actually purify a child by ≥ 5 points
  const purifying = splits.filter((x) => x.modalGain > 0.05).length / splits.length;
  process.stdout.write(`${name.padEnd(6)} leaves=${tree.leaves}  meanIG=${mean((x) => x.ig).toFixed(4)}  meanModalGain=${mean((x) => x.modalGain).toFixed(4)}  wMeanModalGain=${wmean((x) => x.modalGain).toFixed(4)}  meanBal=${(100 * mean((x) => x.bal)).toFixed(0)}%  splits-that-purify=${(100 * purifying).toFixed(0)}%\n`);
};

process.stdout.write(`# mechanismProbe — records=${N} sMin=${SMIN} | per-split IG vs child-purity-gain\n`);
analyze(load("wiki_valid_r1"), "r1");
analyze(load("wiki_valid_r5"), "r5");
