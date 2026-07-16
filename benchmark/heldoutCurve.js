"use strict";

// ============================================================================
// benchmark/heldoutCurve.js
//
// In-sample global IG AND held-out cross-entropy vs #splits, best-first, for r1
// and r5. If IG "reflects purity" but r5 overfits, r5's in-sample IG keeps rising
// while its HELD-OUT cross-entropy turns UP past the crossover — the overfitting
// signature. r1 should keep its held-out flat/low. Emits CSV.
// ============================================================================

import fs from "node:fs";
import { readDataset, NO_LABEL } from "../src/core/signatures/signatureDataset.js";
import { growPart } from "../src/core/predict/index.js";

const unionSorted = (a, b) => { const out = []; let i = 0, j = 0; while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } } while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out; };
const featOf = (r) => unionSorted(r.L, r.C), labOf = (r) => r.nextWord;
const N = Number(process.argv[2] || 15000);
const MAXSPLITS = Number(process.env.MAXSPLITS || 180), SMIN = 20, EVAL = 4, A = 0.1, B = 0.1;
const load = (f) => readDataset(new Uint8Array(fs.readFileSync(`data/signatures/${f}.bin`))).records.filter((r) => r.scale === 0).slice(0, N);

const R1 = load("wiki_valid_r1"), R5 = load("wiki_valid_r5");
const labels = [], fs1 = [], fs5 = [];
for (let i = 0; i < Math.min(R1.length, R5.length); i++) { if (R1[i].nextWord === NO_LABEL) continue; labels.push(R1[i].nextWord); fs1.push(new Set(featOf(R1[i]))); fs5.push(new Set(featOf(R5[i]))); }
const M = labels.length, cut = Math.floor(M * 0.8);
const trainIdx = [...Array(cut).keys()], testIdx = []; for (let i = cut; i < M; i++) testIdx.push(i);
const Hset = (idx) => { const c = new Map(); for (const i of idx) c.set(labels[i], (c.get(labels[i]) || 0) + 1); let h = 0; for (const v of c.values()) { const p = v / idx.length; h -= p * Math.log(p); } return h; };

const run = (fset) => {
  const uni = new Map(); for (const i of trainIdx) uni.set(labels[i], (uni.get(labels[i]) || 0) + 1);
  const V = uni.size, TN = cut;
  const Puni = (w) => ((uni.get(w) || 0) + A) / (TN + A * V);
  const faAll = fset.map((s) => [...s]); // global-indexed, as growPart expects
  const propose = (idx) => growPart(idx, faAll, fset, labels, { sMin: SMIN });
  const mkLeaf = (idx) => { const counts = new Map(); for (const i of idx) counts.set(labels[i], (counts.get(labels[i]) || 0) + 1); return { idx, counts, total: idx.length, split: idx.length >= 2 * SMIN ? propose(idx) : null, H: Hset(idx), leaf: true }; };
  const root = mkLeaf(trainIdx);
  const frontier = [root];
  const Hroot = root.H; let wLeafH = Hroot;
  const routeLeaf = (i) => { let n = root; while (!n.leaf) { let x = 0; for (const b of n.bits) if (fset[i].has(b)) x++; n = x >= n.k ? n.present : n.absent; } return n; };
  const testCE = () => { let ce = 0; for (const i of testIdx) { const L = routeLeaf(i); const w = labels[i]; const p = (1 - B) * ((L.counts.get(w) || 0) + A) / (L.total + A * V) + B * Puni(w); ce -= Math.log(p); } return ce / testIdx.length; };
  const out = [[0, 0, testCE()]];
  for (let step = 0; step < MAXSPLITS; step++) {
    let bi = -1, bp = 0; for (let i = 0; i < frontier.length; i++) { const f = frontier[i]; if (f.split) { const pr = f.idx.length * f.split.gain; if (pr > bp) { bp = pr; bi = i; } } }
    if (bi < 0) break;
    const node = frontier[bi]; frontier.splice(bi, 1); const s = node.split;
    const fire = mkLeaf(s.fire), not = mkLeaf(s.notFire);
    wLeafH += (fire.total / cut) * fire.H + (not.total / cut) * not.H - (node.total / cut) * node.H;
    node.leaf = false; node.bits = s.bits; node.k = s.k; node.present = fire; node.absent = not; node.split = null; node.idx = null; node.counts = null;
    frontier.push(fire, not);
    if ((step + 1) % EVAL === 0) out.push([step + 1, Hroot - wLeafH, testCE()]);
  }
  return out;
};

process.stderr.write(`M=${M} train=${cut} test=${testIdx.length}; running...\n`);
const c1 = run(fs1), c5 = run(fs5);
const map1 = new Map(c1.map((r) => [r[0], r])), map5 = new Map(c5.map((r) => [r[0], r]));
const xs = [...new Set([...map1.keys(), ...map5.keys()])].sort((a, b) => a - b);
process.stdout.write("split,r1_IG,r1_CE,r5_IG,r5_CE\n");
for (const x of xs) { const a = map1.get(x), b = map5.get(x); process.stdout.write(`${x},${a ? a[1].toFixed(4) : ""},${a ? a[2].toFixed(4) : ""},${b ? b[1].toFixed(4) : ""},${b ? b[2].toFixed(4) : ""}\n`); }
