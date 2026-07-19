"use strict";

// benchmark/phase1/densBench.js — matched-metric test: greedy with OR-recall selection vs greedy with DENSITY-gate
// selection (buildPartDensity: forward-select bits to maximize the density gate's recall−λFP). Does aligning the
// selection metric with the deployed gate lower FP at equal recall? Reports TRAIN→HELD/FP + K/|Q|. Baseline: replicator.
//   node --max-old-space-size=8192 benchmark/phase1/densBench.js <dict> <corpus>

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../../src/core/parts/tokenize.js";
import { encodeWord } from "../../src/core/signatures/wordEncoder.js";
import { segmentText } from "../../src/utilities/textSegmentation/segmentText.js";
import fitClass from "../../src/core/phase1/fit.js";
import { fitClassGreedy } from "../../src/core/phase1/greedy.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGETS = (process.env.TARGETS || "time,world,state,music,century,river,government,bank,physics,philosophy,hydrogen").split(",");
const R = Number(process.env.R || 5), D = R + 1, RHO = Number(process.env.RHO || 80);
const NEGPOOL = Number(process.env.NEGPOOL || 60000), DELTA = Number(process.env.DELTA || 0.05), DBAND = Number(process.env.DBAND || 0.05);
const SAMP = Number(process.env.SAMP || 4000);

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

const samp = (arr, n) => { if (arr.length <= n) return arr; const step = arr.length / n, out = []; for (let i = 0; i < arr.length; i += step) out.push(arr[Math.floor(i)]); return out; };
const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
const meanLen = (Qs) => (Qs.length ? Qs.reduce((s, q) => s + q.length, 0) / Qs.length : 0);
const gOpt = (o) => ({ delta: DBAND, peel: true, candidates: "union", Rstar: 0.6, thObj: "diff", ...o });

const KEYS = ["rep", "greedyEns", "greedy1"];
const timed = (fn) => { const t = process.hrtime.bigint(); const m = fn(); m.ms = Number(process.hrtime.bigint() - t) / 1e6; return m; };
const w = process.stdout;
w.write(`# ensemble (AdaBoost+peel) vs SINGLE grown part (no knee, no boost) | canon=${TARGETS.length} | 4-fold | TRAIN→HELD/FP (K,|Q|,ms)\n`);
w.write(`  word          |A|    replicator                 greedy ENSEMBLE            greedy SINGLE (1 part)\n`);
const agg = Object.fromEntries(KEYS.map((k) => [k, { tr: 0, hd: 0, f: 0, k: 0, q: 0, ms: 0 }])); let N = 0;
const rows = [];
for (const word of TARGETS) {
  const Aw = classA.get(word); if (Aw.length < 40) continue;
  const acc = Object.fromEntries(KEYS.map((k) => [k, [0, 0, 0, 0, 0, 0]]));
  for (let fold = 0; fold < 4; fold++) {
    const fa = [], te = []; for (let k = 0; k < Aw.length; k++) (k % 4 === fold ? te : fa).push(Aw[k]);
    const nTr = Math.max(1, Math.floor(fa.length * 0.75)); const trS = samp(fa.slice(0, nTr), SAMP);
    const models = {
      rep: timed(() => { const f = fitClass(fa, negPool, Mglob, { rho: RHO, delta: DBAND }); return { fires: f.head.fires, neg: f.neg, Qs: f.G.map((g) => g.Qbits), k: f.G.length }; }),
      greedyEns: timed(() => { const g = fitClassGreedy(fa, negPool, Mglob, gOpt({ selectBy: "or" })); return { fires: g.fires, neg: g.neg, Qs: g.parts.map((p) => p.Qs), k: g.parts.length }; }),
      greedy1: timed(() => { const g = fitClassGreedy(fa, negPool, Mglob, gOpt({ selectBy: "or", maxRounds: 1, noKnee: true, Rstar: 0.9 })); return { fires: g.fires, neg: g.neg, Qs: g.parts.map((p) => p.Qs), k: g.parts.length }; }),
    };
    for (const key of KEYS) { const m = models[key], nS = samp(m.neg.Neg, SAMP); acc[key][0] += rate((y) => m.fires(y), trS); acc[key][1] += rate((y) => m.fires(y), te); acc[key][2] += rate((y) => m.fires(y), nS); acc[key][3] += m.k; acc[key][4] += meanLen(m.Qs); acc[key][5] += m.ms; }
  }
  const row = { word, A: Aw.length }; const cells = KEYS.map((key) => {
    const a = acc[key].map((v) => v / 4);
    agg[key].tr += a[0] * 100; agg[key].hd += a[1] * 100; agg[key].f += a[2] * 100; agg[key].k += a[3]; agg[key].q += a[4]; agg[key].ms += a[5];
    row[key] = { tr: +(a[0] * 100).toFixed(1), hd: +(a[1] * 100).toFixed(1), fp: +(a[2] * 100).toFixed(1), K: +a[3].toFixed(1), Q: +a[4].toFixed(0), ms: +a[5].toFixed(0) };
    return `${(a[0] * 100).toFixed(0)}→${(a[1] * 100).toFixed(0)}/${(a[2] * 100).toFixed(0)} (${a[3].toFixed(0)},${a[4].toFixed(0)},${a[5].toFixed(0)}ms)`.padEnd(27);
  });
  N++; rows.push(row);
  w.write(`  ${word.padEnd(12)}${String(Aw.length).padStart(6)}   ${cells.join("")}\n`);
}
const M = (o) => `${(o.tr / N).toFixed(0)}→${(o.hd / N).toFixed(0)}/${(o.f / N).toFixed(0)} (${(o.k / N).toFixed(1)},${(o.q / N).toFixed(0)},${(o.ms / N).toFixed(0)}ms)`;
w.write(`  ${"MEAN".padEnd(12)}${"".padStart(6)}   ${KEYS.map((k) => M(agg[k]).padEnd(27)).join("")}\n`);
w.write(`# JSON ${JSON.stringify(rows)}\n`);
