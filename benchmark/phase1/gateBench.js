"use strict";

// benchmark/phase1/gateBench.js — per-subpart GATE comparison. Greedy gives the subparts (bits); we swap the scoring:
//   density  : |Q_k∩x| / |x|            (baseline greedy gate)
//   cooc     : Σ_{a,b∈x∩Q_k} M_k[a,b]   (per-subpart co-occurrence — the hypothesis)
//   usum     : Σ_{b∈x∩Q_k} u_b          (per-bit log-odds — the fingerprint result)
// Each part's threshold is FP-aware (recall−λ·FP on val pos + neg), α-summed into a θ-tuned head. Report TRAIN and
// HELD recall/FP (4-fold) so any gain must hold on both. Baselines: full replicator.
//   node --max-old-space-size=8192 benchmark/phase1/gateBench.js <dict> <corpus>

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../../src/core/parts/tokenize.js";
import { encodeWord } from "../../src/core/signatures/wordEncoder.js";
import { segmentText } from "../../src/utilities/textSegmentation/segmentText.js";
import fitClass from "../../src/core/phase1/fit.js";
import { fitClassGreedy } from "../../src/core/phase1/greedy.js";
import buildNegSet, { pairKey } from "../../src/core/phase1/negSet.js";
import buildAffinity from "../../src/core/phase1/affinity.js";
import andCount from "../../src/core/math/sparse/andCount.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGETS = (process.env.TARGETS || "time,world,state,music,century,river,government,bank,physics,philosophy,hydrogen").split(",");
const R = Number(process.env.R || 5), D = R + 1, RHO = Number(process.env.RHO || 80);
const NEGPOOL = Number(process.env.NEGPOOL || 60000), DELTA = Number(process.env.DELTA || 0.05), DBAND = Number(process.env.DBAND || 0.05);
const SAMP = Number(process.env.SAMP || 4000);                        // cap on train/neg eval sets (bounds cooc pair cost)

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

const sortAsc = (a) => [...new Set(a)].sort((x, y) => x - y);
const samp = (arr, n) => { if (arr.length <= n) return arr; const step = arr.length / n, out = []; for (let i = 0; i < arr.length; i += step) out.push(arr[Math.floor(i)]); return out; };
const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);

// per-part scorers over part bits Q_k: density, cooc (ΣM), usum (Σu). Builds M_k/u_k on the subset.
const scorers = (Qk, A, wU, neg) => {
  const Qs = sortAsc(Qk), Qset = new Set(Qs);
  const subNeg = { ...neg, pPlus: Qs, pset: Qset };
  const { u, edges } = buildAffinity(A, wU, subNeg, {});
  const uMap = new Map(); Qs.forEach((b, i) => uMap.set(b, u[i]));
  const pm = new Map(); for (let e = 0; e < edges.m.length; e++) pm.set(pairKey(Qs[edges.i[e]], Qs[edges.j[e]]), edges.m[e]);
  const present = (x) => { const q = []; for (const b of x) if (Qset.has(b)) q.push(b); return q; };
  return {
    density: (x) => (x.length ? andCount(x, Qs) / x.length : 0),
    cooc: (x) => { const q = present(x); let s = 0; for (let i = 0; i < q.length; i++) for (let j = i + 1; j < q.length; j++) { const v = pm.get(pairKey(q[i], q[j])); if (v) s += v; } return s; },
    usum: (x) => { let s = 0; for (const b of x) { const v = uMap.get(b); if (v) s += v; } return s; },
  };
};
// FP-aware per-part threshold: argmax recall−λ·FP over val-pos & neg score distributions.
const partTheta = (score, valPos, negS, lambda) => {
  const pos = valPos.map(score), neg = negS.map(score);
  const cands = [...new Set([...pos, ...neg])].sort((a, b) => a - b);
  let bt = cands[0] ?? 0, bo = -Infinity;
  for (const t of cands) { const r = pos.filter((v) => v >= t).length / (pos.length || 1); const f = neg.length ? neg.filter((v) => v >= t).length / neg.length : 0; if (r - lambda * f > bo) { bo = r - lambda * f; bt = t; } }
  return bt;
};
// α-sum head over per-part gates, θ tuned on val (recall−λ·FP).
const headEval = (parts, gate, tr, val, te, negNeg, lambda) => {
  const gs = parts.map((p) => ({ fn: p.sc[gate], th: partTheta(p.sc[gate], val, negNeg, lambda), a: p.alpha }));
  const S = (x) => { let s = 0; for (const g of gs) if (g.fn(x) >= g.th) s += g.a; return s; };
  const Sv = val.map(S), Sn = negNeg.map(S), cands = [...new Set([0, ...Sv, ...Sn])].sort((a, b) => a - b);
  let th = 0, bo = -Infinity;
  for (const c of cands) { const r = Sv.filter((s) => s > c).length / (Sv.length || 1); const f = Sn.length ? Sn.filter((s) => s > c).length / Sn.length : 0; if (r - lambda * f > bo) { bo = r - lambda * f; th = c; } }
  const fires = (x) => S(x) > th;
  return [rate(fires, tr), rate(fires, te), rate(fires, negNeg)];
};

const KEYS = ["rep", "density", "cooc", "usum"];
const w = process.stdout;
w.write(`# per-subpart gate: density vs cooc(ΣM) vs usum(Σu) | canon=${TARGETS.length} | 4-fold | TRAIN→HELD/FP\n`);
w.write(`  word          |A|    replicator              density (greedy)        cooc (ΣM)               usum (Σu)\n`);
const agg = Object.fromEntries(KEYS.map((k) => [k, [0, 0, 0]])); let N = 0;
const rows = [];
for (const word of TARGETS) {
  const Aw = classA.get(word); if (Aw.length < 40) continue;
  const acc = Object.fromEntries(KEYS.map((k) => [k, [0, 0, 0]]));
  for (let fold = 0; fold < 4; fold++) {
    const fa = [], te = []; for (let k = 0; k < Aw.length; k++) (k % 4 === fold ? te : fa).push(Aw[k]);
    const nTr = Math.max(1, Math.floor(fa.length * 0.75)); const tr = fa.slice(0, nTr), val = fa.slice(nTr);
    const lambda = Math.min(3, Math.max(1, 1500 / fa.length));
    const fit = fitClass(fa, negPool, Mglob, { rho: RHO, delta: DBAND });
    const trS = samp(tr, SAMP), negS = samp(fit.neg.Neg, SAMP);
    acc.rep[0] += rate((y) => fit.head.fires(y), trS); acc.rep[1] += rate((y) => fit.head.fires(y), te); acc.rep[2] += rate((y) => fit.head.fires(y), negS);
    const g = fitClassGreedy(fa, negPool, Mglob, { delta: DBAND, peel: true, candidates: "union", Rstar: 0.6, thObj: "diff" });
    const neg = g.neg, wU = new Float64Array(tr.length).fill(1 / tr.length);
    const parts = g.parts.map((p) => ({ alpha: p.alpha, sc: scorers(p.Qs, tr, wU, neg) }));
    const negSamp = samp(neg.Neg, SAMP);
    for (const gate of ["density", "cooc", "usum"]) { const [rt, rh, f] = headEval(parts, gate, trS, val, te, negSamp, lambda); acc[gate][0] += rt; acc[gate][1] += rh; acc[gate][2] += f; }
  }
  const row = { word, A: Aw.length }; const cells = KEYS.map((key) => {
    const a = acc[key].map((v) => 100 * v / 4);
    agg[key][0] += a[0]; agg[key][1] += a[1]; agg[key][2] += a[2];
    row[key] = { tr: +a[0].toFixed(1), hd: +a[1].toFixed(1), fp: +a[2].toFixed(1) };
    return `${a[0].toFixed(0)}→${a[1].toFixed(0)}/${a[2].toFixed(0)}`.padEnd(24);
  });
  N++; rows.push(row);
  w.write(`  ${word.padEnd(12)}${String(Aw.length).padStart(6)}   ${cells.join("")}\n`);
}
w.write(`  ${"MEAN".padEnd(12)}${"".padStart(6)}   ${KEYS.map((k) => `${(agg[k][0] / N).toFixed(0)}→${(agg[k][1] / N).toFixed(0)}/${(agg[k][2] / N).toFixed(0)}`.padEnd(24)).join("")}\n`);
w.write(`# JSON ${JSON.stringify(rows)}\n`);
