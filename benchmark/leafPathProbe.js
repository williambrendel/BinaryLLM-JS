"use strict";

// ============================================================================
// benchmark/leafPathProbe.js
//
// Per leaf, the root→leaf path is a mix of "contains" conditions (leaf is on the
// FIRE side of a split) and "lacks" conditions (leaf on the ABSENT side). Does the
// contains-fraction of a leaf's path correlate with its purity / entropy?
// Hypothesis: leaves defined by shared parts (high contains-frac) are purer than
// leaves defined by absences (catch-alls).
//
// Usage: node benchmark/leafPathProbe.js [dataset.bin]   (CHUNK via env)
// ============================================================================

import fs from "node:fs";
import { readDataset } from "../src/core/signatures/signatureDataset.js";
import { buildCartTree } from "../src/core/predict/index.js";

const unionSorted = (a, b) => { const out = []; let i = 0, j = 0; while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } } while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out; };
const featOf = (r) => unionSorted(r.L, r.C), labOf = (r) => r.nextWord;
const path = process.argv[2] || "data/signatures/wiki_valid_r5.bin";
const CHUNK = Number(process.env.CHUNK || 20000);
const ds = readDataset(new Uint8Array(fs.readFileSync(path)));
const chunk = ds.records.filter((r) => r.scale === 0).slice(0, CHUNK);
const tree = buildCartTree(chunk, { featuresOf: featOf, labelOf: labOf, sMin: 20, D: 3, Dmax: 25 });

const leaves = [];
(function walk(n, nC, nL) {
  if (n.part === undefined) {
    const size = n.hist.total; let top = 0, H = 0; for (const c of n.hist.counts.values()) { if (c > top) top = c; const p = c / size; H -= p * Math.log(p); }
    leaves.push({ size, modal: top / size, H, nC, nL, frac: nC + nL ? nC / (nC + nL) : 0 });
    return;
  }
  walk(n.present, nC + 1, nL); walk(n.absent, nC, nL + 1);
})(tree.root, 0, 0);

const pearson = (xs, ys) => { const n = xs.length; const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n; let sxy = 0, sx = 0, sy = 0; for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; sxy += dx * dy; sx += dx * dx; sy += dy * dy; } return sxy / Math.sqrt(sx * sy); };
const frac = leaves.map((l) => l.frac);
const w = process.stdout;
w.write(`# leafPathProbe — ${path.split("/").pop()} | ${leaves.length} leaves | contains-frac of path vs purity/entropy\n`);
w.write(`  Pearson r:  frac↔modalPurity = ${pearson(frac, leaves.map((l) => l.modal)).toFixed(3)}   frac↔leafEntropy = ${pearson(frac, leaves.map((l) => l.H)).toFixed(3)}   frac↔size = ${pearson(frac, leaves.map((l) => l.size)).toFixed(3)}\n\n`);
w.write(`  contains-frac   #leaves   meanSize   meanModal%   meanEntropy\n`);
const bins = [[0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.01]];
for (const [lo, hi] of bins) {
  const L = leaves.filter((l) => l.frac >= lo && l.frac < hi);
  if (!L.length) { w.write(`  [${lo}-${hi})        0\n`); continue; }
  const mean = (f) => L.reduce((s, l) => s + f(l), 0) / L.length;
  w.write(`  [${lo}-${hi})   ${String(L.length).padStart(7)}   ${mean((l) => l.size).toFixed(1).padStart(8)}   ${(100 * mean((l) => l.modal)).toFixed(1).padStart(9)}   ${mean((l) => l.H).toFixed(3).padStart(11)}\n`);
}
