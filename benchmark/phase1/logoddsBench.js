"use strict";

// benchmark/phase1/logoddsBench.js — [PROTOTYPE] count-trained log-odds gate (naive Bayes) vs the greedy fit.
// The scaling bet: u_c[b]=log((p⁺_c[b]+α)/(p⁻[b]+α)) is COUNTED, not optimized — p⁻ is a GLOBAL background computed
// once and shared across all classes, p⁺_c is a per-class count. score(x)=Σ_{b∈x} u_c[b] (+background for unseen
// bits = the −p⁻ "not-other-class" signal), fire if > θ_c. Reports held rec/FP AND per-class build-ms (the whole
// point — is training O(corpus counting) instead of O(classes × fit)?).
//   node --max-old-space-size=8192 benchmark/phase1/logoddsBench.js <dict> <corpus>

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../../src/core/parts/tokenize.js";
import { encodeWord } from "../../src/core/signatures/wordEncoder.js";
import { segmentText } from "../../src/utilities/textSegmentation/segmentText.js";
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

// GLOBAL background p⁻ — computed ONCE, shared across every class (the crux of the tractability claim).
const t0g = process.hrtime.bigint();
const negRate = new Map(); for (const y of negPool) for (const b of y) negRate.set(b, (negRate.get(b) || 0) + 1);
for (const [b, c] of negRate) negRate.set(b, c / negPool.length);
const negBuildMs = Number(process.hrtime.bigint() - t0g) / 1e6;
const ALPHA = 1 / (negPool.length || 1);
const bg = (b) => Math.log(ALPHA / ((negRate.get(b) || 0) + ALPHA));   // background log-odds for a bit the class never saw

const samp = (arr, n) => { if (arr.length <= n) return arr; const step = arr.length / n, out = []; for (let i = 0; i < arr.length; i += step) out.push(arr[Math.floor(i)]); return out; };
const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
const ms = (ns) => Number(ns) / 1e6;

// count-trained log-odds fit. `full`: include the −p⁻ background penalty for unseen bits.
const fitLogOdds = (A, negNeg, full) => {
  const lambda = Math.min(3, Math.max(1, 1500 / A.length));
  const nTr = Math.max(1, Math.floor(A.length * 0.75)); const tr = A.slice(0, nTr), val = A.slice(nTr);
  const cnt = new Map(); for (const x of tr) for (const b of x) cnt.set(b, (cnt.get(b) || 0) + 1);
  const u = new Map(); for (const [b, c] of cnt) u.set(b, Math.log((c / tr.length + ALPHA) / ((negRate.get(b) || 0) + ALPHA)));
  const score = (x) => { let s = 0; for (const b of x) { if (u.has(b)) s += u.get(b); else if (full) s += bg(b); } return s; };
  const Sv = val.map(score).sort((a, b) => b - a), Sn = negNeg.map(score);
  const cands = [...new Set([...Sv, ...Sn])].sort((a, b) => a - b);
  let th = cands[0] ?? 0, bo = -Infinity;
  for (const t of cands) { const r = Sv.filter((s) => s >= t).length / (Sv.length || 1), f = Sn.length ? Sn.filter((s) => s >= t).length / Sn.length : 0; if (r - lambda * f > bo) { bo = r - lambda * f; th = t; } }
  return { fires: (x) => score(x) >= th };
};

const w = process.stdout;
w.write(`# [PROTOTYPE] count-trained log-odds (naive Bayes) vs greedy | canon=${TARGETS.length} | 4-fold | held-rec/FP (build-ms)\n`);
w.write(`# global p⁻ background built once in ${negBuildMs.toFixed(0)}ms (shared across all classes)\n`);
w.write(`  word          |A|    greedy                  logOdds+ (pos only)     logOdds (full, −p⁻)\n`);
const KEYS = ["greedy", "loPos", "loFull"];
const agg = Object.fromEntries(KEYS.map((k) => [k, { r: 0, f: 0, ms: 0 }])); let N = 0;
for (const word of TARGETS) {
  const Aw = classA.get(word); if (Aw.length < 40) continue;
  const acc = Object.fromEntries(KEYS.map((k) => [k, [0, 0, 0]]));
  for (let fold = 0; fold < 4; fold++) {
    const fa = [], te = []; for (let k = 0; k < Aw.length; k++) (k % 4 === fold ? te : fa).push(Aw[k]);
    const negS = samp(negPool, SAMP);
    let t = process.hrtime.bigint(); const g = fitClassGreedy(fa, negPool, Mglob, { delta: DBAND, peel: true, candidates: "union", Rstar: 0.6 }); const gms = ms(process.hrtime.bigint() - t);
    t = process.hrtime.bigint(); const lp = fitLogOdds(fa, negS, false); const lpms = ms(process.hrtime.bigint() - t);
    t = process.hrtime.bigint(); const lf = fitLogOdds(fa, negS, true); const lfms = ms(process.hrtime.bigint() - t);
    const models = { greedy: [g, gms], loPos: [lp, lpms], loFull: [lf, lfms] };
    for (const key of KEYS) { const [m, bm] = models[key]; acc[key][0] += rate((y) => m.fires(y), te); acc[key][1] += rate((y) => m.fires(y), negS); acc[key][2] += bm; }
  }
  const cells = KEYS.map((key) => { const a = acc[key].map((v) => v / 4); agg[key].r += a[0] * 100; agg[key].f += a[1] * 100; agg[key].ms += a[2]; return `${(a[0] * 100).toFixed(0)}/${(a[1] * 100).toFixed(0)} (${a[2].toFixed(0)}ms)`.padEnd(24); });
  N++; w.write(`  ${word.padEnd(12)}${String(Aw.length).padStart(6)}   ${cells.join("")}\n`);
}
const M = (o) => `${(o.r / N).toFixed(0)}/${(o.f / N).toFixed(0)} (${(o.ms / N).toFixed(0)}ms)`;
w.write(`  ${"MEAN".padEnd(12)}${"".padStart(6)}   ${KEYS.map((k) => M(agg[k]).padEnd(24)).join("")}\n`);
w.write(`# per-class speedup: logOdds+ ${(agg.greedy.ms / agg.loPos.ms).toFixed(0)}× · logOdds-full ${(agg.greedy.ms / agg.loFull.ms).toFixed(0)}× faster to train than greedy\n`);
