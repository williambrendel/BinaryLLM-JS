"use strict";

// benchmark/phase1/greedyBench.js — CV the greedy bit-sort part construction (greedy.js, [CANDIDATE]) against the
// deployed replicator (fitClass) on the canon: (held-rec, confFP, K, intra-Jaccard, build-time). Validation
// checklist for whether the co-occurrence-free greedy can replace the replicator (spec: buildPartGreedy §checklist).
//   node --max-old-space-size=8192 benchmark/phase1/greedyBench.js <dict> <corpus>

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../../src/core/parts/tokenize.js";
import { encodeWord } from "../../src/core/signatures/wordEncoder.js";
import { segmentText } from "../../src/utilities/textSegmentation/segmentText.js";
import fitClass from "../../src/core/phase1/fit.js";
import makeHead from "../../src/core/phase1/head.js";
import { fitClassGreedy } from "../../src/core/phase1/greedy.js";
import { jaccard } from "../../src/core/math/sparse/jaccard.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGETS = (process.env.TARGETS || "time,world,state,music,century,river,government,bank,physics,philosophy,hydrogen").split(",");
const R = Number(process.env.R || 5), D = R + 1, RHO = Number(process.env.RHO || 80);
const NEGPOOL = Number(process.env.NEGPOOL || 60000), DELTA = Number(process.env.DELTA || 0.05), DBAND = Number(process.env.DBAND || 0.05);

const [dictArg, corpusArg] = process.argv.slice(2);
const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F;
const pc = new Map(); const partsOf = (wd) => { let a = pc.get(wd); if (!a) { a = [...new Set(encodeWord(dict, wd))]; pc.set(wd, a); } return a; };
const sents = [];
for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
  const ws = tokenizeStream(line.slice(seg.start, seg.end)).filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
  if (ws.length >= 2) sents.push(ws);
}
const band = (dd) => (dd === 1 ? 2 : dd <= 3 ? 1 : 0);
const stmp = new Int32Array(DIM).fill(-1); let sp = 0;
const feat = (s, t) => { sp++; const y = []; for (let dd = 1; dd <= D; dd++) { const q = t - dd; if (q < 0) break; const off = band(dd) * F; for (const a of partsOf(s[q])) { const b = a + off; if (stmp[b] !== sp) { stmp[b] = sp; y.push(b); } } } return y.sort((a, b) => a - b); };

const tset = new Set(TARGETS), classA = new Map(TARGETS.map((t) => [t, []])), negPool = [], bitCount = new Map();
let Ntot = 0, ni = 0; const addC = (y) => { Ntot++; for (const b of y) bitCount.set(b, (bitCount.get(b) || 0) + 1); return y; };
for (const s of sents) for (let t = 1; t < s.length; t++) { if (tset.has(s[t])) classA.get(s[t]).push(addC(feat(s, t))); else if (ni++ % Math.max(1, Math.floor((sents.length * 8) / NEGPOOL)) === 0) negPool.push(addC(feat(s, t))); }
const Mglob = new Set(); for (const [b, c] of bitCount) if (c / Ntot > DELTA) Mglob.add(b);

const rate = (pred, pool) => (pool.length ? pool.filter(pred).length / pool.length : 0);
const intraJac = (Qs) => { let s = 0, n = 0; for (let i = 0; i < Qs.length; i++) for (let j = i + 1; j < Qs.length; j++) { s += jaccard(Qs[i], Qs[j]); n++; } return n ? s / n : 0; };
const ms = (ns) => Number(ns) / 1e6;
const sortAsc = (a) => [...new Set(a)].sort((x, y) => x - y);
// cross-method overlap between greedy `g` and replicator `r` on held-out `te`: coverage Jaccard, containment
// (fraction of greedy-covered positives the replicator also covers), and bit Jaccard of ⋃-bits.
const crossOverlap = (rM, gM, te) => {
  const rC = te.map((y) => rM.fires(y)), gC = te.map((y) => gM.fires(y));
  let inter = 0, uni = 0, gOnly = 0; for (let i = 0; i < te.length; i++) { const r = rC[i], g = gC[i]; if (r && g) inter++; if (r || g) uni++; if (g && !r) gOnly++; }
  return [uni ? inter / uni : 0, inter + gOnly ? inter / (inter + gOnly) : 0, jaccard(sortAsc(rM.Qs.flat()), sortAsc(gM.Qs.flat()))];
};
// per fold: full replicator, replicator's FIRST core only (repK1), greedy few-core (R*0.6), greedy single core (gU1).
const buildModels = (fa) => {
  let t0 = process.hrtime.bigint(); const fit = fitClass(fa, negPool, Mglob, { rho: RHO, delta: DBAND }); const repMs = ms(process.hrtime.bigint() - t0);
  const g0 = fit.G[0];
  const rep = { fires: fit.head.fires, neg: fit.neg, Qs: fit.G.map((g) => g.Qbits), k: fit.G.length, ms: repMs };
  const repK1 = { fires: g0 ? makeHead([g0]).fires : () => false, neg: fit.neg, Qs: g0 ? [g0.Qbits] : [], k: g0 ? 1 : 0, ms: repMs };  // 1st AdaBoost core only (~free, derived)
  t0 = process.hrtime.bigint(); const g6 = fitClassGreedy(fa, negPool, Mglob, { delta: DBAND, peel: true, candidates: "union", Rstar: 0.6 }); const g6ms = ms(process.hrtime.bigint() - t0);
  const gU06 = { fires: g6.fires, neg: g6.neg, Qs: g6.parts.map((p) => p.Qs), k: g6.parts.length, ms: g6ms };
  t0 = process.hrtime.bigint(); const g1 = fitClassGreedy(fa, negPool, Mglob, { delta: DBAND, peel: true, candidates: "union", Rstar: 0.6, maxRounds: 1 }); const g1ms = ms(process.hrtime.bigint() - t0);
  const gU1 = { fires: g1.fires, neg: g1.neg, Qs: g1.parts.map((p) => p.Qs), k: g1.parts.length, ms: g1ms };
  return { rep, repK1, gU06, gU1 };
};
const KEYS = ["rep", "repK1", "gU06", "gU1"];
const w = process.stdout;
w.write(`# greedy(union,p⁻-t) vs replicator — full + single-core | canon=${TARGETS.length} | 4-fold | rec/FP/K/|Q|/jac/ms\n`);
w.write(`  word          |A|    rep full                rep K=1 (1st core)      greedy R*0.6            greedy K=1              | overlap vs rep (cov/bit)\n`);
const agg = Object.fromEntries(KEYS.map((k) => [k, { r: 0, f: 0, k: 0, q: 0, j: 0, ms: 0 }])); agg.n = 0;
const ovAgg = { full: [0, 0, 0], single: [0, 0, 0] };                 // full = gU06↔rep, single = gU1↔repK1
const rows = [];
for (const word of TARGETS) {
  const Aw = classA.get(word); if (Aw.length < 8) continue;
  const acc = Object.fromEntries(KEYS.map((k) => [k, [0, 0, 0, 0, 0, 0]]));
  const ov = { full: [0, 0, 0], single: [0, 0, 0] };
  for (let fold = 0; fold < 4; fold++) {
    const fa = [], te = []; for (let k = 0; k < Aw.length; k++) (k % 4 === fold ? te : fa).push(Aw[k]);
    const m = buildModels(fa);
    for (const key of KEYS) {
      acc[key][0] += rate((y) => m[key].fires(y), te); acc[key][1] += rate((y) => m[key].fires(y), m[key].neg.Neg);
      acc[key][2] += m[key].k; acc[key][3] += m[key].Qs.length ? m[key].Qs.reduce((s, q) => s + q.length, 0) / m[key].Qs.length : 0;
      acc[key][4] += intraJac(m[key].Qs); acc[key][5] += m[key].ms;
    }
    const of = crossOverlap(m.rep, m.gU06, te), os = crossOverlap(m.repK1, m.gU1, te);
    for (let i = 0; i < 3; i++) { ov.full[i] += of[i]; ov.single[i] += os[i]; }
  }
  const row = { word, A: Aw.length }; const cells = KEYS.map((key) => {
    const a = acc[key].map((v) => v / 4);
    agg[key].r += a[0] * 100; agg[key].f += a[1] * 100; agg[key].k += a[2]; agg[key].q += a[3]; agg[key].j += a[4]; agg[key].ms += a[5];
    row[key] = { R: +(a[0] * 100).toFixed(1), F: +(a[1] * 100).toFixed(1), K: +a[2].toFixed(1), Q: +a[3].toFixed(0) };
    return `${(a[0] * 100).toFixed(0)}/${(a[1] * 100).toFixed(0)}/${a[2].toFixed(0)}/${a[3].toFixed(0)}/${a[4].toFixed(2)}/${a[5].toFixed(0)}`.padEnd(24);
  });
  const ovf = ov.full.map((v) => v / 4), ovs = ov.single.map((v) => v / 4);
  for (let i = 0; i < 3; i++) { ovAgg.full[i] += ovf[i]; ovAgg.single[i] += ovs[i]; }
  row.ov = { fullCov: +ovf[0].toFixed(2), fullBit: +ovf[2].toFixed(2), K1cov: +ovs[0].toFixed(2), K1bit: +ovs[2].toFixed(2) };
  agg.n++; rows.push(row);
  w.write(`  ${word.padEnd(12)}${String(Aw.length).padStart(6)}   ${cells.join("")}  | full ${ovf[0].toFixed(2)}/${ovf[2].toFixed(2)} · K1 ${ovs[0].toFixed(2)}/${ovs[2].toFixed(2)}\n`);
}
const M = (o) => `${(o.r / agg.n).toFixed(0)}/${(o.f / agg.n).toFixed(0)}/${(o.k / agg.n).toFixed(1)}/${(o.q / agg.n).toFixed(0)}/${(o.j / agg.n).toFixed(2)}/${(o.ms / agg.n).toFixed(0)}`;
w.write(`  ${"MEAN".padEnd(12)}${"".padStart(6)}   ${KEYS.map((k) => M(agg[k]).padEnd(24)).join("")}\n`);
w.write(`# overlap MEAN — full(gU06↔rep): cov ${(ovAgg.full[0] / agg.n).toFixed(2)} contain ${(ovAgg.full[1] / agg.n).toFixed(2)} bit ${(ovAgg.full[2] / agg.n).toFixed(2)} | single(gU1↔repK1): cov ${(ovAgg.single[0] / agg.n).toFixed(2)} contain ${(ovAgg.single[1] / agg.n).toFixed(2)} bit ${(ovAgg.single[2] / agg.n).toFixed(2)}\n`);
w.write(`# JSON ${JSON.stringify(rows)}\n`);
