"use strict";

// benchmark/phase1/binaryGateBench.js — [PROTOTYPE] does the FULLY-BINARY signed-popcount gate hold up vs float SLOG?
// Gate:  fire iff  popcount(W⁺_c ∧ x) − popcount(W⁻_c ∧ x)  >  t_c·|x| + b_c,   with t_c,b_c tuned PER ROW (per class).
// W⁺_c = { b : u_c[b] ≥ +M },  W⁻_c = { b : u_c[b] ≤ −M }  — i.e. the log-odds u_c=log((p⁺_c+α)/(p⁻+α)) quantized to
// ternary {−1,0,+1} at margin M (shared bits, u≈0, drop out — the popcount cancellation). M is swept per row too.
// Compares: greedy (deployed float head) · SLOG-float (Σ float log-odds, per-row θ) · signedPop (integer popcount, per-row t,b,M).
// The question: binarizing the weights should keep recall (distinctive bits are near-binary) but lose FP (graded relatedness gone).
//   node --max-old-space-size=8192 benchmark/phase1/binaryGateBench.js <dict> <corpus>

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

// meaningful sample set: ~40 common content words (override via env TARGETS)
const TARGETS = (process.env.TARGETS ||
  "time,world,state,music,century,river,government,bank,physics,philosophy,hydrogen,water,energy,system,number,language," +
  "history,science,city,country,war,book,film,company,church,school,family,group,party,market,law,art,king,president," +
  "university,population,building,region,species,culture").split(",");
const R = Number(process.env.R || 5), D = R + 1;
const NEGPOOL = Number(process.env.NEGPOOL || 60000), DELTA = Number(process.env.DELTA || 0.05), DBAND = Number(process.env.DBAND || 0.05);
const SAMP = Number(process.env.SAMP || 4000);
const MAXA = Number(process.env.MAXA || 4000); // cap per-class contexts (even-stride subsample) so greedy stays tractable
const MARGINS = (process.env.MARGINS || "0,0.5,1,1.5,2").split(",").map(Number);
const TGRID = (process.env.TGRID || "0,0.02,0.05,0.1,0.15,0.2,0.3").split(",").map(Number);

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

// GLOBAL background p⁻ — counted once, shared across every class.
const negRate = new Map(); for (const y of negPool) for (const b of y) negRate.set(b, (negRate.get(b) || 0) + 1);
for (const [b, c] of negRate) negRate.set(b, c / negPool.length);
const ALPHA = 1 / (negPool.length || 1);
const bg = (b) => Math.log(ALPHA / ((negRate.get(b) || 0) + ALPHA));

const samp = (arr, n) => { if (arr.length <= n) return arr; const step = arr.length / n, out = []; for (let i = 0; i < arr.length; i += step) out.push(arr[Math.floor(i)]); return out; };
const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
const ms = (ns) => Number(ns) / 1e6;

// float SLOG (naive-Bayes log-odds, per-row θ) — the reference the binary gate must match.
const fitLogOdds = (A, negNeg) => {
  const lambda = Math.min(3, Math.max(1, 1500 / A.length));
  const nTr = Math.max(1, Math.floor(A.length * 0.75)); const tr = A.slice(0, nTr), val = A.slice(nTr);
  const cnt = new Map(); for (const x of tr) for (const b of x) cnt.set(b, (cnt.get(b) || 0) + 1);
  const u = new Map(); for (const [b, c] of cnt) u.set(b, Math.log((c / tr.length + ALPHA) / ((negRate.get(b) || 0) + ALPHA)));
  const score = (x) => { let s = 0; for (const b of x) { if (u.has(b)) s += u.get(b); else s += bg(b); } return s; };
  const Sv = val.map(score), Sn = negNeg.map(score);
  const cands = [...new Set([...Sv, ...Sn])].sort((a, b) => a - b);
  let th = cands[0] ?? 0, bo = -Infinity;
  for (const t of cands) { const r = rate((s) => s >= t, Sv), f = rate((s) => s >= t, Sn); if (r - lambda * f > bo) { bo = r - lambda * f; th = t; } }
  return { fires: (x) => score(x) >= th };
};

// FULLY-BINARY signed-popcount gate. Weights ternary at margin M; per-row (t,b) and M tuned on val+neg.
const fitSignedPop = (A, negNeg) => {
  const lambda = Math.min(3, Math.max(1, 1500 / A.length));
  const nTr = Math.max(1, Math.floor(A.length * 0.75)); const tr = A.slice(0, nTr), val = A.slice(nTr);
  const cnt = new Map(); for (const x of tr) for (const b of x) cnt.set(b, (cnt.get(b) || 0) + 1);
  const u = new Map(); for (const [b, c] of cnt) u.set(b, Math.log((c / tr.length + ALPHA) / ((negRate.get(b) || 0) + ALPHA)));
  let best = null;
  for (const M of MARGINS) {
    const Wp = new Set(), Wn = new Set();
    for (const [b, v] of u) { if (v >= M) Wp.add(b); else if (v <= -M) Wn.add(b); }
    if (Wp.size === 0) continue;
    // integer popcount score = |W⁺∧x| − |W⁻∧x|
    const pop = (x) => { let s = 0; for (const b of x) { if (Wp.has(b)) s++; else if (Wn.has(b)) s--; } return s; };
    const vP = val.map((x) => [pop(x), x.length]), vN = negNeg.map((x) => [pop(x), x.length]);
    for (const t of TGRID) {
      // z = score − t·|x|; optimal intercept b is a 1-D threshold on z
      const zP = vP.map(([s, L]) => s - t * L), zN = vN.map(([s, L]) => s - t * L);
      const cands = [...new Set([...zP, ...zN])].sort((a, b) => a - b);
      for (const b of cands) {
        const r = rate((z) => z > b, zP), f = rate((z) => z > b, zN);
        const obj = r - lambda * f;
        if (!best || obj > best.obj) best = { obj, M, t, b, Wp, Wn };
      }
    }
  }
  if (!best) return { fires: () => false, M: 0, t: 0 };
  const { Wp, Wn, t, b } = best;
  const pop = (x) => { let s = 0; for (const y of x) { if (Wp.has(y)) s++; else if (Wn.has(y)) s--; } return s; };
  return { fires: (x) => pop(x) - t * x.length > b, M: best.M, t: best.t, wp: Wp.size, wn: Wn.size };
};

const w = process.stdout;
w.write(`# [PROTOTYPE] fully-binary signed-popcount gate vs float SLOG vs greedy | canon=${TARGETS.length} words | 4-fold | held-rec/FP\n`);
w.write(`# signedPop: popcount(W⁺∧x)−popcount(W⁻∧x) > t·|x|+b, ternary weights @margin M, (t,b,M) tuned PER ROW\n`);
w.write(`  word          |A|    greedy(float head)     SLOG(float)             signedPop(binary)       binMeta\n`);
const KEYS = ["greedy", "slog", "spop"];
const agg = Object.fromEntries(KEYS.map((k) => [k, { r: 0, f: 0 }])); let N = 0; let mAgg = 0, tAgg = 0, wpAgg = 0;
for (const word of TARGETS) {
  const Aw0 = classA.get(word); if (!Aw0 || Aw0.length < 40) continue;
  const Aw = samp(Aw0, MAXA);
  const acc = Object.fromEntries(KEYS.map((k) => [k, [0, 0]])); let mm = 0, tt = 0, ww = 0;
  for (let fold = 0; fold < 4; fold++) {
    const fa = [], te = []; for (let k = 0; k < Aw.length; k++) (k % 4 === fold ? te : fa).push(Aw[k]);
    const negS = samp(negPool, SAMP);
    const g = fitClassGreedy(fa, negPool, Mglob, { delta: DBAND, peel: true, candidates: "union", Rstar: 0.6 });
    const sl = fitLogOdds(fa, negS);
    const sp = fitSignedPop(fa, negS); mm += sp.M; tt += sp.t; ww += (sp.wp || 0);
    const models = { greedy: g, slog: sl, spop: sp };
    for (const key of KEYS) { const m = models[key]; acc[key][0] += rate((y) => m.fires(y), te); acc[key][1] += rate((y) => m.fires(y), negS); }
  }
  const cells = KEYS.map((key) => { const a = acc[key].map((v) => v / 4); agg[key].r += a[0] * 100; agg[key].f += a[1] * 100; return `${(a[0] * 100).toFixed(0)}/${(a[1] * 100).toFixed(0)}`.padEnd(24); });
  mAgg += mm / 4; tAgg += tt / 4; wpAgg += ww / 4; N++;
  w.write(`  ${word.padEnd(12)}${String(Aw0.length).padStart(7)}   ${cells.join("")}M=${(mm / 4).toFixed(1)} t=${(tt / 4).toFixed(2)} |W⁺|=${(ww / 4).toFixed(0)}\n`);
}
const M = (o) => `${(o.r / N).toFixed(0)}/${(o.f / N).toFixed(0)}`;
w.write(`  ${"MEAN".padEnd(12)}${"".padStart(6)}   ${KEYS.map((k) => M(agg[k]).padEnd(24)).join("")}M=${(mAgg / N).toFixed(1)} t=${(tAgg / N).toFixed(2)} |W⁺|=${(wpAgg / N).toFixed(0)}\n`);
w.write(`# read: if signedPop FP >> SLOG FP at matched recall, binarizing the weights loses the graded (low-rank relatedness) signal.\n`);
