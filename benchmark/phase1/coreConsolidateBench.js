"use strict";

// benchmark/phase1/coreConsolidateBench.js — do the overlapping deployed parts carry more than one core, and
// do the retired strategies beat them? 4-fold cross-word CV on the canon, all configs sharing the same folds /
// negatives / base-parts, each θ-tuned on val, scored on TRAIN and the held fold (recTrain / recHeld / confFP):
//   current     — deployed parts (recallTau@0.6 + AdaBoost + val early-stop)
//   adaptCore   — the adaptive-core: STRONG(0.7 recall)+dynFPMAX → ship single, else augment to 0.55 + hard-peel
//   stopStrong  — AdaBoost (strong extraction), STOP as soon as a part has recall > 0.7 (keep up to it)
//   single70    — one part grown to ≥70% recall (the "grow-to-0.7" variant)
//   union       — deployed parts collapsed into one part = ⋃ Qbits
//   intersect   — one part = ⋂ Qbits
//   twoCore     — two parts: [⋂ , ⋃−⋂]

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
const orRate = (bits, pool) => { if (!bits.length || !pool.length) return 0; const s = sortAsc(bits); return pool.filter((y) => andCount(y, s) >= 1).length / pool.length; };
const mkPart = (bits, A, w, neg) => {
  const Q = sortAsc(bits); if (!Q.length) return null;
  const fit = mfit(Q, A, w, neg.Neg);
  const rec = fit.valid ? fit.recall : (fit.recallAt1 || 0), m = fit.valid ? fit.m : 1;
  const rc = Math.min(Math.max(rec, 0.501), 0.999);
  return { Qbits: Q, m, alpha: 0.5 * Math.log(rc / (1 - rc)) };
};
const prefixSelect = (G, val, neg, lam) => {
  let best = -Infinity, rStar = G.length;
  for (let r = 1; r <= G.length; r++) { const h = makeHead(G.slice(0, r)); const o = rate((y) => h.fires(y), val) - lam * rate((y) => h.fires(y), neg.Neg); if (o > best) { best = o; rStar = r; } }
  return G.slice(0, rStar);
};
const tuneEval = (G, tr, val, te, neg, lam) => {                            // θ-tune on val, score train/held/FP
  if (!G || !G.length) return [0, 0, 0];
  let head = makeHead(G);
  const Sval = val.map((y) => head.S(y)), Sneg = neg.Neg.map((y) => head.S(y));
  const cands = [...new Set([0, ...Sval, ...Sneg])].sort((a, b) => a - b);
  let bTh = head.theta, bObj = -Infinity;
  for (const c of cands) { const vr = Sval.length ? Sval.filter((s) => s > c).length / Sval.length : 0; const vf = Sneg.length ? Sneg.filter((s) => s > c).length / Sneg.length : 0; if (vr - lam * vf > bObj) { bObj = vr - lam * vf; bTh = c; } }
  head = makeHead(G, bTh);
  return [rate((y) => head.fires(y), tr), rate((y) => head.fires(y), te), rate((y) => head.fires(y), neg.Neg)];
};

// config 1 — the adaptive-core (STRONG/dynFPMAX single vs augment-to-0.55 + HARD-PEEL). Peel done inline by
// driving peeled bits' unary to −∞ (the cleaned affinity has no `exclude` opt).
const adaptiveCore = (A, neg) => {
  const nA = A.length, wU = new Float64Array(nA).fill(1 / nA);
  const peeled = new Set(), covered = new Uint8Array(nA), cores = [];
  for (let round = 0; round < 12; round++) {
    let W = 0; const w = new Float64Array(nA); for (let i = 0; i < nA; i++) { w[i] = covered[i] ? 0.02 : 1; W += w[i]; } for (let i = 0; i < nA; i++) w[i] /= W;
    const { u, edges } = buildAffinity(A, w, neg, {});
    for (let p = 0; p < u.length; p++) if (peeled.has(neg.pPlus[p])) u[p] = -1e9;   // hard peel
    const r = replicate(u, edges, { solver: "exp", rho: RHO, suppPatience: 3 });
    const byW = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i]]).sort((a, b) => b[1] - a[1]).map((e) => e[0]);
    if (!byW.length) break;
    const dset = new Set(byW), cmap = new Map();
    for (let i = 0; i < nA; i++) for (const b of A[i]) if (dset.has(b)) { let a = cmap.get(b); if (!a) { a = []; cmap.set(b, a); } a.push(i); }
    if (round === 0) {
      const p = []; for (const b of byW) { p.push(b); if (orRate(p, A) >= 0.5) break; } const r50 = p.length;
      const dynFP = Math.min(0.25, 0.015 * r50);
      if (orRate(byW, A) >= 0.7 && orRate(byW, neg.Neg) <= dynFP) { cores.push(byW.slice()); break; }   // ship single strong+clean
    }
    const wc = new Uint8Array(nA); let wcov = 0; const weak = [];   // augment weight-first to 0.55 weighted recall
    for (const b of byW) { weak.push(b); for (const i of (cmap.get(b) || [])) if (!wc[i]) { wc[i] = 1; wcov += w[i]; } if (wcov >= 0.55) break; }
    cores.push(weak); for (const b of weak) peeled.add(b);
    const ws = new Set(weak); for (let i = 0; i < nA; i++) if (!covered[i]) { for (const b of A[i]) if (ws.has(b)) { covered[i] = 1; break; } }
    let cov = 0; for (let i = 0; i < nA; i++) if (covered[i]) cov++;
    if (cov / nA >= 0.85) break;
  }
  return cores.map((c) => mkPart(c, A, wU, neg)).filter(Boolean);
};

// config 2 — AdaBoost with STRONG (full dominant-set) extraction, STOP at the first part whose recall > 0.7.
const stopStrong = (A, neg) => {
  const G = boost(A, neg, { rho: RHO, solver: "exp", recallTau: false, suppPatience: 3 }).G;
  let cut = G.length; for (let i = 0; i < G.length; i++) if (G[i].recall > 0.7) { cut = i + 1; break; }
  return G.slice(0, cut);
};

// single70 — one part grown to ≥70% weighted recall (the "grow-to-0.7" variant).
const single70 = (A, neg) => {
  const w = new Float64Array(A.length).fill(1 / A.length);
  const { u, edges } = buildAffinity(A, w, neg, {});
  const r = replicate(u, edges, { solver: "exp", rho: RHO, suppPatience: 3 });
  if (!r.Q.length) return [];
  const order = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i]]).sort((a, b) => b[1] - a[1]).map((e) => e[0]);
  const cov = new Uint8Array(A.length); let wcov = 0; const Qbits = [];
  for (const b of order) { Qbits.push(b); for (let i = 0; i < A.length; i++) if (!cov[i]) { for (const t of A[i]) if (t === b) { cov[i] = 1; wcov += w[i]; break; } } if (wcov >= 0.7) break; }
  const p = mkPart(Qbits, A, w, neg); return p ? [p] : [];
};

// ── per-word 4-fold CV ──
const CFGS = ["current", "adaptCore", "stopStrong", "single70", "union", "intersect", "twoCore"];
const w = process.stdout;
w.write(`# core-consolidation CV (4-fold, θ-tuned) | canon=${TARGETS.length} | recTrain/recHeld/FP %\n`);
w.write(`  word          |A|   ${CFGS.map((c) => c.padEnd(15)).join("")}\n`);
const agg = Object.fromEntries(CFGS.map((c) => [c, { t: 0, r: 0, f: 0, n: 0 }]));
for (const word of TARGETS) {
  const Aw = classA.get(word); if (Aw.length < 8) continue;
  const acc = Object.fromEntries(CFGS.map((c) => [c, [0, 0, 0]]));
  for (let fold = 0; fold < 4; fold++) {
    const fa = [], te = []; for (let k = 0; k < Aw.length; k++) (k % 4 === fold ? te : fa).push(Aw[k]);
    const nTr = Math.max(1, Math.floor(fa.length * 0.75)); const tr = fa.slice(0, nTr), val = fa.slice(nTr);
    const neg = buildNegSet(tr, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const lam = Math.min(3, Math.max(1, 1500 / fa.length)), wU = new Float64Array(tr.length).fill(1 / tr.length);
    const Gkept = prefixSelect(boost(tr, neg, { rho: RHO, solver: "exp", recallTau: true, tauFloor: 0.6, suppPatience: 3 }).G, val, neg, lam);
    const uni = sortAsc(Gkept.flatMap((g) => g.Qbits));
    let inter = Gkept.length ? [...Gkept[0].Qbits] : []; for (const g of Gkept.slice(1)) { const S = new Set(g.Qbits); inter = inter.filter((b) => S.has(b)); }
    inter = sortAsc(inter); const iset = new Set(inter); const tail = uni.filter((b) => !iset.has(b));
    const build = {
      current: Gkept,
      adaptCore: adaptiveCore(tr, neg),
      stopStrong: stopStrong(tr, neg),
      single70: single70(tr, neg),
      union: [mkPart(uni, tr, wU, neg)].filter(Boolean),
      intersect: [mkPart(inter, tr, wU, neg)].filter(Boolean),
      twoCore: [mkPart(inter, tr, wU, neg), mkPart(tail, tr, wU, neg)].filter(Boolean),
    };
    for (const c of CFGS) { const [t, r, f] = tuneEval(build[c], tr, val, te, neg, lam); acc[c][0] += t; acc[c][1] += r; acc[c][2] += f; }
  }
  const cells = CFGS.map((c) => { const t = 100 * acc[c][0] / 4, r = 100 * acc[c][1] / 4, f = 100 * acc[c][2] / 4; agg[c].t += t; agg[c].r += r; agg[c].f += f; agg[c].n++; return `${t.toFixed(0)}/${r.toFixed(0)}/${f.toFixed(0)}`.padEnd(15); });
  w.write(`  ${word.padEnd(12)}${String(Aw.length).padStart(5)}   ${cells.join("")}\n`);
}
w.write(`  ${"MEAN".padEnd(12)}${"".padStart(5)}   ${CFGS.map((c) => `${(agg[c].t / agg[c].n).toFixed(0)}/${(agg[c].r / agg[c].n).toFixed(0)}/${(agg[c].f / agg[c].n).toFixed(0)}`.padEnd(15)).join("")}\n`);
w.write(`# JSON ${JSON.stringify(CFGS.map((c) => ({ cfg: c, train: +(agg[c].t / agg[c].n).toFixed(1), held: +(agg[c].r / agg[c].n).toFixed(1), fp: +(agg[c].f / agg[c].n).toFixed(1) })))}\n`);
