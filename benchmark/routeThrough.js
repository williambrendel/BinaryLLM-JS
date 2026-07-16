"use strict";

// ============================================================================
// benchmark/routeThrough.js
//
// ONE tree, built once (on r1). Then push the SAME records — but encoded at
// different radii — through that fixed partition, and measure the purity of where
// they land. This separates the tree STRUCTURE from the input ENCODING: the splits
// (parts + k) never change; only the signatures fed to them do. Answers "does
// running r5 through THE tree change the purity" without rebuilding the tree.
//
// Usage: node benchmark/routeThrough.js   (SMIN, CHUNK, ALPHA via env)
// ============================================================================

import fs from "node:fs";
import { readDataset, NO_LABEL } from "../src/core/signatures/signatureDataset.js";
import { buildCartTree } from "../src/core/predict/index.js";

const unionSorted = (a, b) => { const out = []; let i = 0, j = 0; while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } } while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out; };
const featOf = (r) => unionSorted(r.L, r.C);
const labOf = (r) => r.nextWord;

const CHUNK = Number(process.env.CHUNK || 20000);
const SMIN = Number(process.env.SMIN || 20);
const ALPHA = Number(process.env.ALPHA ?? 1);
const load = (f) => readDataset(new Uint8Array(fs.readFileSync(`data/signatures/${f}.bin`))).records.filter((r) => r.scale === 0).slice(0, CHUNK);

const enc = { r1: load("wiki_valid_r1"), r5: load("wiki_valid_r5"), full: load("wiki_valid_sigs") };
// Build THE tree once, on r1.
const tree = buildCartTree(enc.r1, { featuresOf: featOf, labelOf: labOf, sMin: SMIN, alpha: ALPHA, D: 3, Dmax: 25 });
const root = tree.root;

// Walk to the leaf NODE (identity) for a bit-set, using the fixed splits.
const routeLeaf = (bits) => { let n = root; while (n.part !== undefined) { let inter = 0; for (const b of n.part) if (bits.has(b)) inter++; n = inter >= n.k ? n.present : n.absent; } return n; };

// Purity of routing one encoding's records through the fixed tree.
const purity = (recs) => {
  const leaves = new Map(); // leaf node -> Map(label -> count)
  for (const r of recs) { const l = labOf(r); if (l === NO_LABEL) continue; const leaf = routeLeaf(new Set(featOf(r))); let m = leaves.get(leaf); if (!m) leaves.set(leaf, (m = new Map())); m.set(l, (m.get(l) || 0) + 1); }
  let szSum = 0, distSum = 0, topSum = 0, n = 0, occupied = 0;
  for (const m of leaves.values()) { let sz = 0, top = 0; for (const c of m.values()) { sz += c; if (c > top) top = c; } szSum += sz; distSum += m.size; topSum += top / sz; n++; occupied += sz; }
  return { leaves: n, meanLeaf: szSum / n, meanDistinct: distSum / n, sizeOverDistinct: szSum / distSum, modalPurity: 100 * topSum / n };
};

const w = process.stdout;
w.write(`# routeThrough — ONE tree built on r1 (sMin=${SMIN} α=${ALPHA}, ${tree.leaves} leaves), route other encodings through it | chunk=${CHUNK}\n`);
w.write(`encoding  leaves-hit  meanLeaf  meanDistinct  size/distinct  modalPurity\n`);
for (const [name, recs] of Object.entries(enc)) {
  const s = purity(recs);
  w.write(`${name.padEnd(8)}  ${String(s.leaves).padStart(9)}  ${s.meanLeaf.toFixed(1).padStart(8)}  ${s.meanDistinct.toFixed(1).padStart(12)}  ${s.sizeOverDistinct.toFixed(2).padStart(13)}  ${s.modalPurity.toFixed(1).padStart(10)}%\n`);
}
