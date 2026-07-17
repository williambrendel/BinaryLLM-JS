"use strict";

// benchmark/phase1/coreConsolidateBench.js — canon 4-fold CV, REAL pipeline (fitClass) for `current`; the
// alternatives reuse the real head/mfit/θ-tune. Reports recTrain / recHeld / FP and, for the deployed parts,
// K / part-size / intra-Jaccard. Configs:
//   current    — deployed fitClass head (real pipeline)
//   avgWeights — current parts, but every α = mean(α) (uniform head weights instead of AdaBoost votes)
//   adaptCore  — adaptive-core: STRONG(0.7)+dynFPMAX single, else augment-to-0.55 + hard-peel
//   stopStrong — AdaBoost (strong extraction), STOP at the first part with recall > 0.7
//   union      — one part ⋃ Qbits, α = mean(α)
//   intersect  — one part ⋂ Qbits, α = mean(α)
//   twoCore    — [⋂ , ⋃−⋂], both α = mean(α)

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
import fitClass from "../../src/core/phase1/fit.js";
import mfit from "../../src/core/phase1/mfit.js";
import makeHead from "../../src/core/phase1/head.js";
import andCount from "../../src/core/math/sparse/andCount.js";
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
const mean = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0);
const wOr = (bits, A, w) => { const s = sortAsc(bits); if (!s.length) return 0; let tw = 0, W = 0; for (let i = 0; i < A.length; i++) { W += w[i]; if (andCount(A[i], s) >= 1) tw += w[i]; } return W ? tw / W : 0; };
// weight-first prefix: add bits in replicator-weight order until weighted OR-recall ≥ target; returns the bits.
const wPrefix = (byW, A, w, target) => {
  const cov = new Uint8Array(A.length); let wc = 0, W = 0; for (let i = 0; i < A.length; i++) W += w[i]; const bits = [];
  for (const b of byW) { bits.push(b); for (let i = 0; i < A.length; i++) if (!cov[i]) { const yi = A[i]; for (let t = 0; t < yi.length; t++) if (yi[t] === b) { cov[i] = 1; wc += w[i]; break; } } if (wc / (W || 1) >= target - 1e-9) break; }
  return bits;
};
// waste-removal at equal recall: full-dominant OR-recall R_dom, then the fewest WEIGHT-FIRST bits still holding R_dom
// (drops redundant bits at the same recall ≈ same FP → the tight, discriminative core).
const wasteRemove = (byW, A, w) => { const Rdom = wOr(byW, A, w); return { tight: wPrefix(byW, A, w, Rdom), Rdom }; };
// union of cores = OR of the ORIGINAL part gates (recall ≥ current by construction — the α-sum can't fire unless a
// gate fires). No θ; reports the recall ceiling and the FP the θ-head trims.
const orGateEval = (G, tr, te, neg) => { const fire = (y) => G.some((g) => andCount(y, g.Qbits) >= g.m); return [rate(fire, tr), rate(fire, te), rate(fire, neg.Neg)]; };
const mkPartA = (bits, A, w, neg, alpha) => { const Q = sortAsc(bits); if (!Q.length) return null; const fit = mfit(Q, A, w, neg.Neg); return { Qbits: Q, m: fit.valid ? fit.m : 1, alpha }; };
const tuneEval = (G, tr, val, te, neg, lam) => {                            // θ-tune on val (fit.js contract), score
  if (!G || !G.length) return [0, 0, 0];
  let head = makeHead(G);
  const Sval = val.map((y) => head.S(y)), Sneg = neg.Neg.map((y) => head.S(y));
  const cands = [...new Set([0, ...Sval, ...Sneg])].sort((a, b) => a - b);
  let bTh = head.theta, bObj = -Infinity;
  for (const c of cands) { const vr = Sval.length ? Sval.filter((s) => s > c).length / Sval.length : 0; const vf = Sneg.length ? Sneg.filter((s) => s > c).length / Sneg.length : 0; if (vr - lam * vf > bObj) { bObj = vr - lam * vf; bTh = c; } }
  head = makeHead(G, bTh);
  return [rate((y) => head.fires(y), tr), rate((y) => head.fires(y), te), rate((y) => head.fires(y), neg.Neg)];
};

// adaptive-core (waste-removal edition, per the agreed sequence): each round extract the FULL dominant set →
// waste-remove to the tight core at equal recall R_dom → if R_dom≥0.7 AND FP≤dynFPMAX ship it single & stop;
// else take a weak learner (weight-first to just >0.55), peel its bits (u=-1e9), reweight uncovered, iterate.
const adaptiveCore = (A, neg, peelMode = "tight") => {                           // peel-amount axis: "weak"=only the shipped 0.55 learner · "tight"=the waste-removed core (R_dom) · "full"=the entire dominant set (no reuse of waste bits)
  const nA = A.length, wU = new Float64Array(nA).fill(1 / nA);
  const peeled = new Set(), covered = new Uint8Array(nA), cores = [];
  for (let round = 0; round < 12; round++) {
    let W = 0; const w = new Float64Array(nA); for (let i = 0; i < nA; i++) { w[i] = covered[i] ? 0.02 : 1; W += w[i]; } for (let i = 0; i < nA; i++) w[i] /= W;
    const { u, edges } = buildAffinity(A, w, neg, {});
    for (let p = 0; p < u.length; p++) if (peeled.has(neg.pPlus[p])) u[p] = -1e9;
    const r = replicate(u, edges, { solver: "exp", rho: RHO, suppPatience: 3 });
    const byW = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i]]).sort((a, b) => b[1] - a[1]).map((e) => e[0]);
    if (!byW.length) break;
    const { tight, Rdom } = wasteRemove(byW, A, w);                              // full dominant → tight core @ equal recall
    const r50 = wPrefix(byW, A, w, 0.5).length;                                  // 50%-recall prefix = concentration
    const dynFPMAX = Math.min(0.25, 0.015 * r50);                               // predicted ensemble FP (cap 25%)
    const tp = mkPartA(tight, A, w, neg, 1);                                     // strong core ships as its m-of-n gate
    const tightFP = tp ? rate((y) => andCount(y, tp.Qbits) >= tp.m, neg.Neg) : 1;
    if (Rdom >= 0.7 && tightFP <= dynFPMAX) { cores.push(tight); break; }        // STRONG + FP-clean → ship single, stop
    const weak = wPrefix(byW, A, w, Math.min(0.55, Rdom));                       // WEAK learner shipped to the head: weight-first to VIABLE 0.55, waste-removed (capped at R_dom)
    const peelSet = peelMode === "full" ? byW : peelMode === "weak" ? weak : tight;  // how much of this round's structure to remove from future rounds
    cores.push(weak); for (const b of peelSet) peeled.add(b);                    // PEEL → disjoint
    const cs = new Set(weak); for (let i = 0; i < nA; i++) if (!covered[i]) { for (const b of A[i]) if (cs.has(b)) { covered[i] = 1; break; } }
    let cov = 0; for (let i = 0; i < nA; i++) if (covered[i]) cov++; if (cov / nA >= 0.85) break;  // union target
  }
  return cores.map((c) => mkPartA(c, A, wU, neg, 1)).filter(Boolean);
};
const stopStrong = (A, neg) => { const G = boost(A, neg, { rho: RHO, solver: "exp", recallTau: false, suppPatience: 3 }).G; let cut = G.length; for (let i = 0; i < G.length; i++) if (G[i].recall > 0.7) { cut = i + 1; break; } return G.slice(0, cut); };

// ── per-word 4-fold CV ──
const CFGS = ["current", "currentFix", "union", "unionWaste", "unionM"];
const w = process.stdout;
w.write(`# consolidation CV (4-fold, REAL fitClass) | canon=${TARGETS.length} | recTrain/recHeld/FP %\n`);
w.write(`  word          |A|   ${CFGS.map((c) => c.padEnd(15)).join("")}   K±sd  |Q|±sd[min-max]  intraJac  headα(cv)\n`);
const agg = Object.fromEntries(CFGS.map((c) => [c, { t: 0, r: 0, f: 0, n: 0 }]));
const stats = [];
for (const word of TARGETS) {
  const Aw = classA.get(word); if (Aw.length < 8) continue;
  const acc = Object.fromEntries(CFGS.map((c) => [c, [0, 0, 0]]));
  let jac = 0, jn = 0, uniSz = 0, uwSz = 0; const alphas = [], qsizes = [], kPerFold = [];
  for (let fold = 0; fold < 4; fold++) {
    const fa = [], te = []; for (let k = 0; k < Aw.length; k++) (k % 4 === fold ? te : fa).push(Aw[k]);
    const nTr = Math.max(1, Math.floor(fa.length * 0.75)); const tr = fa.slice(0, nTr), val = fa.slice(nTr);
    const fit = fitClass(fa, negPool, Mglob, { rho: RHO, delta: DBAND });   // REAL pipeline (recallFloor 0.5)
    const fitFix = fitClass(fa, negPool, Mglob, { rho: RHO, delta: DBAND, softFloor: true });  // fix: keep 0.5 where reachable, else admit the m=1 max-recall sense-clique
    const neg = fit.neg, G = fit.G, lam = Math.min(3, Math.max(1, 1500 / fa.length)), wU = new Float64Array(tr.length).fill(1 / tr.length);
    const avgA = mean(G.map((g) => g.alpha)) || 0.5;
    const uni = sortAsc(G.flatMap((g) => g.Qbits));
    let inter = G.length ? [...G[0].Qbits] : []; for (const g of G.slice(1)) { const S = new Set(g.Qbits); inter = inter.filter((b) => S.has(b)); }
    inter = sortAsc(inter); const iset = new Set(inter); const tail = uni.filter((b) => !iset.has(b));
    // unionWaste: ONE more waste-removal on the union-of-cores as a weight-first MIN-COVER — walk the union bits in
    // discriminative (log-odds u desc) order and KEEP a bit only if it covers a NEW positive, until the union's OR-
    // recall is held. Drops genuinely redundant bits (a prefix can't — it retains covered-already middle bits).
    const { u: uUni } = buildAffinity(tr, wU, neg, {});
    const bitU = new Map(); for (let p = 0; p < uUni.length; p++) bitU.set(neg.pPlus[p], uUni[p]);
    const uniByU = [...uni].sort((a, b) => (bitU.get(b) ?? -1e9) - (bitU.get(a) ?? -1e9));
    const uniSet = new Set(uni), postings = new Map(); for (const b of uni) postings.set(b, []);
    for (let i = 0; i < tr.length; i++) for (const b of tr[i]) if (uniSet.has(b)) postings.get(b).push(i);
    const Runion = wOr(uni, tr, wU); let Wtr = 0; for (let i = 0; i < tr.length; i++) Wtr += wU[i];
    const covMC = new Uint8Array(tr.length); let wcMC = 0; const uwArr = [];
    for (const b of uniByU) { let adds = false; for (const i of postings.get(b)) if (!covMC[i]) { covMC[i] = 1; wcMC += wU[i]; adds = true; } if (adds) uwArr.push(b); if (wcMC / (Wtr || 1) >= Runion - 1e-9) break; }
    const uwBits = sortAsc(uwArr);
    uniSz += uni.length; uwSz += uwBits.length;
    const build = {
      current: G,                                                               // separate cores + θ-head (2 threshold levels: per-core m_k + cross-core θ)
      currentFix: fitFix.G,                                                      // separate cores, softFloor+balanced-α (deployed default)
      union: null,                                                              // OR of the ORIGINAL gates — the recall ceiling (handled by orGateEval)
      unionWaste: uwBits.length ? [mkPartA(uwBits, tr, wU, neg, avgA)].filter(Boolean) : [],  // waste-removed union as ONE m-of-n gate (mfit threshold) + θ-tune
      unionM: uni.length ? [mkPartA(uni, tr, wU, neg, avgA)].filter(Boolean) : [],             // FULL union as ONE m-of-n gate (mfit threshold) + θ-tune
    };
    for (const c of CFGS) { const [t, r, f] = c === "union" ? orGateEval(G, tr, te, neg) : tuneEval(build[c], tr, val, te, neg, lam); acc[c][0] += t; acc[c][1] += r; acc[c][2] += f; }  // union = raw OR ceiling; unionWaste/unionM = single m-of-n gate, θ-tuned
    kPerFold.push(G.length);                                                     // parts per fold
    for (const g of G) { alphas.push(g.alpha); qsizes.push(g.Qbits.length); }    // head weights + part sizes across all parts/folds
    for (let i = 0; i < G.length; i++) for (let j = i + 1; j < G.length; j++) { jac += jaccard(G[i].Qbits, G[j].Qbits); jn++; }
  }
  const cells = CFGS.map((c) => { const t = 100 * acc[c][0] / 4, r = 100 * acc[c][1] / 4, f = 100 * acc[c][2] / 4; agg[c].t += t; agg[c].r += r; agg[c].f += f; agg[c].n++; return `${t.toFixed(0)}/${r.toFixed(0)}/${f.toFixed(0)}`.padEnd(15); });
  const sd = (a, m) => (a.length ? Math.sqrt(mean(a.map((x) => (x - m) ** 2))) : 0);
  const Km = mean(kPerFold), Ksd = sd(kPerFold, Km), Jm = jn ? jac / jn : 0;
  const Qm = mean(qsizes), Qsd = sd(qsizes, Qm), Qmin = qsizes.length ? Math.min(...qsizes) : 0, Qmax = qsizes.length ? Math.max(...qsizes) : 0;
  const aMean = mean(alphas), aCv = aMean ? sd(alphas, aMean) / aMean : 0, aSorted = alphas.slice().sort((a, b) => b - a).map((a) => +a.toFixed(3));
  stats.push({ word, A: Aw.length, K: +Km.toFixed(1), Ksd: +Ksd.toFixed(1), Q: +Qm.toFixed(0), Qsd: +Qsd.toFixed(0), Qmin, Qmax, jac: +Jm.toFixed(2), aMean: +aMean.toFixed(3), aCv: +aCv.toFixed(2), uniBits: +(uniSz / 4).toFixed(0), uwBits: +(uwSz / 4).toFixed(0), alphas: aSorted });
  w.write(`  ${word.padEnd(12)}${String(Aw.length).padStart(5)}   ${cells.join("")}   ${Km.toFixed(0).padStart(2)}±${Ksd.toFixed(0)}  ${Qm.toFixed(0).padStart(3)}±${Qsd.toFixed(0)}[${Qmin}-${Qmax}]  ${Jm.toFixed(2)}  α${aMean.toFixed(2)}cv${aCv.toFixed(2)}  ⋃${(uniSz / 4).toFixed(0)}→${(uwSz / 4).toFixed(0)}\n`);
}
w.write(`  ${"MEAN".padEnd(12)}${"".padStart(5)}   ${CFGS.map((c) => `${(agg[c].t / agg[c].n).toFixed(0)}/${(agg[c].r / agg[c].n).toFixed(0)}/${(agg[c].f / agg[c].n).toFixed(0)}`.padEnd(15)).join("")}\n`);
w.write(`# CFGJSON ${JSON.stringify(CFGS.map((c) => ({ cfg: c, train: +(agg[c].t / agg[c].n).toFixed(1), held: +(agg[c].r / agg[c].n).toFixed(1), fp: +(agg[c].f / agg[c].n).toFixed(1) })))}\n`);
w.write(`# STATJSON ${JSON.stringify(stats)}\n`);
