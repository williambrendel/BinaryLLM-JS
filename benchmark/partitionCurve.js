"use strict";

// ============================================================================
// benchmark/partitionCurve.js
//
// Global IG (= H(root) − weighted mean leaf entropy) as a function of #splits,
// grown BEST-FIRST (each step splits the leaf giving the largest global-IG gain),
// so at every x the four curves have the same number of leaves — a fair race.
//   r1            — native tree on r1
//   r5            — native tree on r5
//   r5 r1-retune  — r5 tree restricted to r1's parts, threshold RE-OPTIMIZED on r5
//   r5 r1-fixed   — r5 tree restricted to r1's parts AND their r1 thresholds (no tune)
// Emits CSV: split,r1,r5,r5_retune,r5_fixed
// ============================================================================

import fs from "node:fs";
import { readDataset, NO_LABEL } from "../src/core/signatures/signatureDataset.js";
import { buildCartTree, growPart } from "../src/core/predict/index.js";

const unionSorted = (a, b) => { const out = []; let i = 0, j = 0; while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } } while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out; };
const featOf = (r) => unionSorted(r.L, r.C), labOf = (r) => r.nextWord;
const N = Number(process.argv[2] || 12000);
const MAXSPLITS = Number(process.env.MAXSPLITS || 200), SMIN = 20;
const load = (f) => readDataset(new Uint8Array(fs.readFileSync(`data/signatures/${f}.bin`))).records.filter((r) => r.scale === 0).slice(0, N);

// Aligned records with a valid label (r1[i] ↔ r5[i], same corpus position).
const R1 = load("wiki_valid_r1"), R5 = load("wiki_valid_r5");
const labels = [], fs1 = [], fs5 = [], fa1 = [], fa5 = [];
for (let i = 0; i < Math.min(R1.length, R5.length); i++) { if (R1[i].nextWord === NO_LABEL) continue; labels.push(R1[i].nextWord); const a1 = featOf(R1[i]), a5 = featOf(R5[i]); fa1.push(a1); fa5.push(a5); fs1.push(new Set(a1)); fs5.push(new Set(a5)); }
const M = labels.length;

const Hof = (idx) => { const c = new Map(); for (const i of idx) c.set(labels[i], (c.get(labels[i]) || 0) + 1); let h = 0; for (const v of c.values()) { const p = v / idx.length; h -= p * Math.log(p); } return h; };

// r1 parts pool (from a full r1 tree).
const t1 = buildCartTree(R1, { featuresOf: featOf, labelOf: labOf, sMin: SMIN, D: 3, Dmax: 25 });
const pool = []; (function walk(n) { if (n.part === undefined) return; pool.push({ bits: n.part, k: n.k }); walk(n.present); walk(n.absent); })(t1.root);

const splitByPart = (idx, fset, bits, k) => { const fire = [], not = []; for (const i of idx) { let x = 0; for (const b of bits) if (fset[i].has(b)) x++; (x >= k ? fire : not).push(i); } if (fire.length < SMIN || not.length < SMIN) return null; const Hf = Hof(fire), Hn = Hof(not); const ig = Hof(idx) - (fire.length / idx.length) * Hf - (not.length / idx.length) * Hn; return { fire, notFire: not, ig, Hfire: Hf, Hnot: Hn }; };

const proposeNative = (fa, fset) => (idx) => { const s = growPart(idx, fa, fset, labels, { sMin: SMIN }); if (!s) return null; const Hf = Hof(s.fire), Hn = Hof(s.notFire); return { fire: s.fire, notFire: s.notFire, ig: Hof(idx) - (s.fire.length / idx.length) * Hf - (s.notFire.length / idx.length) * Hn, Hfire: Hf, Hnot: Hn }; };
const proposePool = (retune) => (idx) => { let best = null; for (const p of pool) { if (retune) { for (let k = 1; k <= p.bits.length; k++) { const s = splitByPart(idx, fs5, p.bits, k); if (s && (!best || s.ig > best.ig)) best = s; } } else { const s = splitByPart(idx, fs5, p.bits, p.k); if (s && (!best || s.ig > best.ig)) best = s; } } return best; };

const curve = (propose) => {
  const Hroot = Hof([...Array(M).keys()]);
  const mk = (idx, H) => ({ idx, H, split: idx.length >= 2 * SMIN ? propose(idx) : null });
  const frontier = [mk([...Array(M).keys()], Hroot)];
  let wLeafH = Hroot; const out = [0];
  for (let step = 0; step < MAXSPLITS; step++) {
    let bi = -1, bp = 0; for (let i = 0; i < frontier.length; i++) { const f = frontier[i]; if (f.split) { const pr = f.idx.length * f.split.ig; if (pr > bp) { bp = pr; bi = i; } } }
    if (bi < 0) break;
    const leaf = frontier[bi]; frontier.splice(bi, 1); const s = leaf.split;
    wLeafH += (s.fire.length / M) * s.Hfire + (s.notFire.length / M) * s.Hnot - (leaf.idx.length / M) * leaf.H;
    out.push(Hroot - wLeafH);
    frontier.push(mk(s.fire, s.Hfire), mk(s.notFire, s.Hnot));
  }
  return out;
};

process.stderr.write(`records=${M}, pool=${pool.length} parts; computing curves...\n`);
const c1 = curve(proposeNative(fa1, fs1)), c5 = curve(proposeNative(fa5, fs5)), cr = curve(proposePool(true)), cf = curve(proposePool(false));
const L = Math.max(c1.length, c5.length, cr.length, cf.length);
process.stdout.write("split,r1,r5,r5_retune,r5_fixed\n");
for (let i = 0; i < L; i++) process.stdout.write(`${i},${(c1[i] ?? "").toString().slice(0, 6)},${(c5[i] ?? "").toString().slice(0, 6)},${(cr[i] ?? "").toString().slice(0, 6)},${(cf[i] ?? "").toString().slice(0, 6)}\n`);
