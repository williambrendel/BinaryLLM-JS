"use strict";

// benchmark/phase1/coreConsolidateBench.js — do the overlapping deployed parts carry more than one core?
// 4-fold cross-word CV on the canon, all configs sharing the same folds / negatives / base-parts, each
// θ-tuned on val and scored on the held fold (recHeld / confFP). Configs:
//   current        — deployed parts (recallTau@0.6 + AdaBoost + val early-stop)
//   adaptive       — extract each part's support by the adaptive elbow-cut on x* (instead of recallTau)
//   single70       — ONE part grown to ≥70% recall, accepted regardless of precision/FP
//   union          — deployed parts collapsed into one part = ⋃ Qbits
//   intersect      — one part = ⋂ Qbits
//   twoCore        — two parts: [⋂ , ⋃−⋂]  (shared core + combined tails)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../../src/core/parts/tokenize.js";
import { encodeWord } from "../../src/core/signatures/wordEncoder.js";
import { segmentText } from "../../src/utilities/textSegmentation/segmentText.js";
import buildNegSet from "../../src/core/phase1/negSet.js";
import buildAffinity from "../../src/core/phase1/affinity.js";
import replicate from "../../src/core/phase1/replicator.js";
import boost from "../../src/core/phase1/boost.js";
import mfit from "../../src/core/phase1/mfit.js";
import makeHead from "../../src/core/phase1/head.js";
import andCount from "../../src/core/math/sparse/andCount.js";
import adaptivePruneCount from "../../src/core/math/adaptiveThreshold.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGETS = (process.env.TARGETS || "time,world,state,music,century,river,government,bank,physics,philosophy,hydrogen").split(",");
const R = Number(process.env.R || 5), D = R + 1, RHO = Number(process.env.RHO || 80);
const NEGPOOL = Number(process.env.NEGPOOL || 60000), DELTA = Number(process.env.DELTA || 0.05), DBAND = Number(process.env.DBAND || 0.05);

const [dictArg, corpusArg] = process.argv.slice(2);
const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F;
const pc = new Map(); const partsOf = (w) => { let a = pc.get(w); if (!a) { a = [...new Set(encodeWord(dict, w))]; pc.set(w, a); } return a; };
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

// ── helpers ──
const rate = (pred, pool) => (pool.length ? pool.filter(pred).length / pool.length : 0);
const sortAsc = (a) => [...new Set(a)].sort((x, y) => x - y);
const mkPart = (bits, A, w, neg) => {                                       // fit m + recall-vote α for a bit-set
  const Q = sortAsc(bits); if (!Q.length) return null;
  const fit = mfit(Q, A, w, neg.Neg);
  const rec = fit.valid ? fit.recall : (fit.recallAt1 || 0), m = fit.valid ? fit.m : 1;
  const rc = Math.min(Math.max(rec, 0.501), 0.999);
  return { Qbits: Q, m, alpha: 0.5 * Math.log(rc / (1 - rc)) };
};
const prefixSelect = (G, val, neg, lam) => {                                // deployed val early-stop prefix
  let best = -Infinity, rStar = G.length;
  for (let r = 1; r <= G.length; r++) { const h = makeHead(G.slice(0, r)); const o = rate((y) => h.fires(y), val) - lam * rate((y) => h.fires(y), neg.Neg); if (o > best) { best = o; rStar = r; } }
  return G.slice(0, rStar);
};
const tuneEval = (G, val, te, neg, lam) => {                                // θ-tune on val, score on te / neg
  if (!G || !G.length) return [0, 0];
  let head = makeHead(G);
  const Sval = val.map((y) => head.S(y)), Sneg = neg.Neg.map((y) => head.S(y));
  const cands = [...new Set([0, ...Sval, ...Sneg])].sort((a, b) => a - b);
  let bTh = head.theta, bObj = -Infinity;
  for (const c of cands) { const vr = Sval.length ? Sval.filter((s) => s > c).length / Sval.length : 0; const vf = Sneg.length ? Sneg.filter((s) => s > c).length / Sneg.length : 0; if (vr - lam * vf > bObj) { bObj = vr - lam * vf; bTh = c; } }
  head = makeHead(G, bTh);
  return [rate((y) => head.fires(y), te), rate((y) => head.fires(y), neg.Neg)];
};

// adaptive-cut extraction: same AdaBoost loop, but the part support = top-adaptivePruneCount(x* weights).
const adaptiveBoost = (A, neg) => {
  const nA = A.length, pMin = nA / (nA + neg.negN); let w = new Float64Array(nA).fill(1 / nA);
  const G = [], covered = new Array(nA).fill(false);
  for (let round = 0; round < 50; round++) {
    const { u, edges } = buildAffinity(A, w, neg, {});
    const r = replicate(u, edges, { solver: "exp", rho: RHO, suppPatience: 3 });
    if (!r.Q.length) break;
    const order = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i]]).sort((a, b) => b[1] - a[1]);
    const k = Math.max(1, adaptivePruneCount(order.map((e) => e[1])));
    const Qbits = sortAsc(order.slice(0, k).map((e) => e[0]));
    const fit = mfit(Qbits, A, w, neg.Neg);
    if (!fit.valid || fit.precision < pMin) break;
    const eps = Math.min(Math.max(1 - fit.recall, 1e-10), 1 - 1e-10), alpha = 0.5 * Math.log((1 - eps) / eps);
    G.push({ Qbits, m: fit.m, alpha });
    let Wn = 0; for (let i = 0; i < nA; i++) { const h = andCount(A[i], Qbits) >= fit.m ? 1 : 0; if (h) covered[i] = true; w[i] *= Math.exp(-alpha * h); Wn += w[i]; } for (let i = 0; i < nA; i++) w[i] /= Wn;
    if (covered.filter(Boolean).length / nA > 0.99) break;
  }
  return G;
};

// single strong core: grow ONE part to ≥70% weighted recall (uniform w), accept regardless of precision.
const singleStrong = (A, neg) => {
  const w = new Float64Array(A.length).fill(1 / A.length);
  const { u, edges } = buildAffinity(A, w, neg, {});
  const r = replicate(u, edges, { solver: "exp", rho: RHO, suppPatience: 3 });
  if (!r.Q.length) return [];
  const order = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i]]).sort((a, b) => b[1] - a[1]).map((e) => e[0]);
  const cov = new Uint8Array(A.length); let wcov = 0; const Qbits = [];
  for (const b of order) { Qbits.push(b); for (let i = 0; i < A.length; i++) if (!cov[i]) { for (const t of A[i]) if (t === b) { cov[i] = 1; wcov += w[i]; break; } } if (wcov >= 0.7) break; }
  const p = mkPart(Qbits, A, w, neg); return p ? [p] : [];
};

// ── per-word 4-fold CV over all configs ──
const CFGS = ["current", "adaptive", "single70", "union", "intersect", "twoCore"];
const w = process.stdout;
w.write(`# core-consolidation CV (4-fold held-out, θ-tuned) | canon=${TARGETS.length} | recHeld%/FP%\n`);
w.write(`  word          |A|   ${CFGS.map((c) => c.padEnd(13)).join("")}\n`);
const agg = Object.fromEntries(CFGS.map((c) => [c, { r: 0, f: 0, n: 0 }]));
for (const word of TARGETS) {
  const Aw = classA.get(word); if (Aw.length < 8) continue;
  const acc = Object.fromEntries(CFGS.map((c) => [c, [0, 0]]));
  for (let fold = 0; fold < 4; fold++) {
    const fa = [], te = []; for (let k = 0; k < Aw.length; k++) (k % 4 === fold ? te : fa).push(Aw[k]);
    const nTr = Math.max(1, Math.floor(fa.length * 0.75)); const tr = fa.slice(0, nTr), val = fa.slice(nTr);
    const neg = buildNegSet(tr, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const lam = Math.min(3, Math.max(1, 1500 / fa.length)), wU = new Float64Array(tr.length).fill(1 / tr.length);
    const Gdep = boost(tr, neg, { rho: RHO, solver: "exp", recallTau: true, tauFloor: 0.6, suppPatience: 3 });
    const Gkept = prefixSelect(Gdep.G, val, neg, lam);
    const uni = sortAsc(Gkept.flatMap((g) => g.Qbits));
    let inter = Gkept.length ? [...Gkept[0].Qbits] : []; for (const g of Gkept.slice(1)) { const S = new Set(g.Qbits); inter = inter.filter((b) => S.has(b)); }
    inter = sortAsc(inter); const iset = new Set(inter); const tail = uni.filter((b) => !iset.has(b));
    const build = {
      current: Gkept,
      adaptive: prefixSelect(adaptiveBoost(tr, neg), val, neg, lam),
      single70: singleStrong(tr, neg),
      union: [mkPart(uni, tr, wU, neg)].filter(Boolean),
      intersect: [mkPart(inter, tr, wU, neg)].filter(Boolean),
      twoCore: [mkPart(inter, tr, wU, neg), mkPart(tail, tr, wU, neg)].filter(Boolean),
    };
    for (const c of CFGS) { const [r, f] = tuneEval(build[c], val, te, neg, lam); acc[c][0] += r; acc[c][1] += f; }
  }
  const cells = CFGS.map((c) => { const r = 100 * acc[c][0] / 4, f = 100 * acc[c][1] / 4; agg[c].r += r; agg[c].f += f; agg[c].n++; return `${r.toFixed(0)}/${f.toFixed(0)}`.padEnd(13); });
  w.write(`  ${word.padEnd(12)}${String(Aw.length).padStart(5)}   ${cells.join("")}\n`);
}
w.write(`  ${"MEAN".padEnd(12)}${"".padStart(5)}   ${CFGS.map((c) => `${(agg[c].r / agg[c].n).toFixed(1)}/${(agg[c].f / agg[c].n).toFixed(1)}`.padEnd(13)).join("")}\n`);
