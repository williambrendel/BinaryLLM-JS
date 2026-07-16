"use strict";

// ============================================================================
// benchmark/bugHunt.js
//
// Claim: r5 ⊇ r1 bit-wise, so the r1 tree's parts are valid coverage queries on
// r5 data — if the r5 grower doesn't find them, that's a growing/selection bug.
// Test at the root: build both trees, take the r1 root part, and evaluate ITS gain
// on r5 data vs the part the r5 grower actually chose. Also check whether the r1
// part even survives the r5 candidate shortlist (growPart shortlists by FREQUENCY).
// ============================================================================

import fs from "node:fs";
import { readDataset, NO_LABEL } from "../src/core/signatures/signatureDataset.js";
import { buildCartTree } from "../src/core/predict/index.js";

const unionSorted = (a, b) => { const out = []; let i = 0, j = 0; while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } } while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out; };
const featOf = (r) => unionSorted(r.L, r.C);
const labOf = (r) => r.nextWord;
const Hb = (p) => (p <= 0 || p >= 1 ? 0 : -p * Math.log(p) - (1 - p) * Math.log(1 - p));

const CHUNK = Number(process.env.CHUNK || 20000);
const SMIN = Number(process.env.SMIN || 20);
const ALPHA = Number(process.env.ALPHA ?? 1);
const M = Number(process.env.M || 48);
const load = (f) => readDataset(new Uint8Array(fs.readFileSync(`data/signatures/${f}.bin`))).records.filter((r) => r.scale === 0).slice(0, CHUNK);
const r1 = load("wiki_valid_r1"), r5 = load("wiki_valid_r5");

// Best coverage gain of a fixed bit-set over records under an encoding (α-weighted
// per-class binary entropy — the SAME objective growPart optimizes).
const gainOfPart = (recs, bits) => {
  const bset = new Set(bits);
  const cls = new Map(); const labPos = []; const ints = [];
  for (const r of recs) { const l = labOf(r); if (l === NO_LABEL) continue; let c = cls.get(l); if (c === undefined) { c = cls.size; cls.set(l, c); } labPos.push(c); let inter = 0; for (const b of featOf(r)) if (bset.has(b)) inter++; ints.push(inter); }
  const C = cls.size, P = labPos.length, s = bits.length;
  const n = new Float64Array(C); for (const c of labPos) n[c]++;
  let Z = 0; const w = new Float64Array(C); for (let c = 0; c < C; c++) { const wc = ALPHA === 0 ? 1 : n[c] ** -ALPHA; w[c] = wc; Z += wc; } for (let c = 0; c < C; c++) w[c] /= Z;
  const H = new Int32Array(C * (s + 1)); for (let i = 0; i < P; i++) H[labPos[i] * (s + 1) + ints[i]]++;
  const a = new Int32Array(C); let best = { g: 0, k: 0 }, mLeft = 0;
  for (let k = s; k >= 1; k--) { let add = 0; for (let c = 0; c < C; c++) { const h = H[c * (s + 1) + k]; a[c] += h; add += h; } mLeft += add; if (mLeft >= SMIN && P - mLeft >= SMIN) { let PL = 0, sp = 0; for (let c = 0; c < C; c++) { const f = a[c] / n[c]; PL += w[c] * f; sp += w[c] * Hb(f); } const g = Hb(PL) - sp; if (g > best.g) best = { g, k }; } }
  return best;
};

// Frequency shortlist (top-M) among a record set under an encoding — what growPart
// would actually consider as candidate seed bits.
const shortlistRank = (recs, bits) => {
  const freq = new Map(); for (const r of recs) for (const b of featOf(r)) freq.set(b, (freq.get(b) || 0) + 1);
  const ranked = [...freq.entries()].sort((x, y) => y[1] - x[1]).map((e) => e[0]);
  const pos = new Map(ranked.map((b, i) => [b, i]));
  return bits.map((b) => ({ b, rank: pos.get(b) ?? -1, freq: freq.get(b) ?? 0, inTopM: (pos.get(b) ?? 1e9) < M }));
};

const t1 = buildCartTree(r1, { featuresOf: featOf, labelOf: labOf, sMin: SMIN, alpha: ALPHA, D: 3, Dmax: 25 });
const t5 = buildCartTree(r5, { featuresOf: featOf, labelOf: labOf, sMin: SMIN, alpha: ALPHA, D: 3, Dmax: 25 });
const P1 = t1.root.part, P5 = t5.root.part;

const w = process.stdout;
w.write(`# bugHunt — root-split adjudication | chunk=${CHUNK} sMin=${SMIN} α=${ALPHA} M=${M}\n\n`);
w.write(`r1 root part = {${P1.join(",")}}  k=${t1.root.k}  gain(on r1)=${t1.root.gain.toFixed(4)}\n`);
w.write(`r5 root part = {${P5.join(",")}}  k=${t5.root.k}  gain(on r5)=${t5.root.gain.toFixed(4)}\n\n`);

const g5_P1 = gainOfPart(r5, P1), g5_P5 = gainOfPart(r5, P5), g1_P1 = gainOfPart(r1, P1);
w.write(`Sanity: r1 part gain recomputed on r1 = ${g1_P1.g.toFixed(4)} (should ≈ tree's ${t1.root.gain.toFixed(4)})\n\n`);
w.write(`KEY — evaluate the r1 root part ON r5 DATA:\n`);
w.write(`  gain of r1-part on r5 = ${g5_P1.g.toFixed(4)} (k=${g5_P1.k})\n`);
w.write(`  gain of r5-part on r5 = ${g5_P5.g.toFixed(4)} (k=${g5_P5.k})  ← what the r5 grower chose\n`);
w.write(`  → r1-part is ${g5_P1.g > g5_P5.g ? "BETTER" : "worse"} on r5 data than the r5 grower's own pick\n\n`);
w.write(`Did the r1-part's bits survive the r5 frequency shortlist (top-M=${M})?\n`);
for (const s of shortlistRank(r5, P1)) w.write(`  bit ${s.b}: r5-freq=${s.freq} rank=${s.rank} ${s.inTopM ? "IN top-M" : "**MISSED** (beyond top-M)"}\n`);
