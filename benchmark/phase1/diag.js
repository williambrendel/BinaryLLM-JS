"use strict";

// benchmark/phase1/diag.js — diagnose (1) why fitClass returns 0 parts for some words (music), and (2) why some
// words need a huge number of bits even after waste-removal at equal recall. Featurizes the corpus once.
//   node --max-old-space-size=8192 benchmark/phase1/diag.js <dict> <corpus>   TARGETS="music,state,world,time"

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
import andCount from "../../src/core/math/sparse/andCount.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGETS = (process.env.TARGETS || "music,state,world,time,century,hydrogen").split(",");
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

const w = process.stdout;
const sortAsc = (a) => [...new Set(a)].sort((x, y) => x - y);
const wOr = (bits, A, wt) => { const s = sortAsc(bits); if (!s.length) return 0; let tw = 0, W = 0; for (let i = 0; i < A.length; i++) { W += wt[i]; if (andCount(A[i], s) >= 1) tw += wt[i]; } return W ? tw / W : 0; };
const nToRecall = (byW, A, wt, target) => { const cov = new Uint8Array(A.length); let wc = 0, W = 0; for (let i = 0; i < A.length; i++) W += wt[i]; let n = 0; for (const b of byW) { n++; for (let i = 0; i < A.length; i++) if (!cov[i]) { const yi = A[i]; for (let t = 0; t < yi.length; t++) if (yi[t] === b) { cov[i] = 1; wc += wt[i]; break; } } if (wc / (W || 1) >= target) break; } return n; };

w.write(`# diag | corpus=${path.basename(corpusArg)} negPool=${negPool.length} rho=${RHO}\n`);
for (const word of TARGETS) {
  const A = classA.get(word); if (!A || A.length < 8) { w.write(`\n=== ${word}: |A|=${A ? A.length : 0} (skipped)\n`); continue; }
  const nTr = Math.max(1, Math.floor(A.length * 0.75)), tr = A.slice(0, nTr);
  const neg = buildNegSet(tr, negPool, Mglob, { delta: DBAND });
  const pMin = tr.length / (tr.length + neg.negN);
  w.write(`\n=== ${word} | |A|=${A.length} tr=${tr.length} negN=${neg.negN} pMin=${pMin.toFixed(3)} p⁺bits=${neg.pPlus.length} ===\n`);

  // (A) MUSIC-style: run boost, report stop reason + curve — WITHOUT and WITH softFloor
  const res = boost(tr, neg, { rho: RHO, solver: "exp", recallTau: true, tauFloor: 0.6, suppPatience: 3 });
  w.write(`  boost(default): stop=${res.stop} rounds=${res.rounds} G=${res.G.length}\n`);
  res.curve.slice(0, 4).forEach((c) => w.write(`    r${c.round}: rec=${c.recall.toFixed(2)} prec=${c.precision.toFixed(3)} (pMin=${pMin.toFixed(3)}) m=${c.m} |Q|=${c.size} unionRec=${c.unionRecall.toFixed(2)}\n`));
  const resSF = boost(tr, neg, { rho: RHO, solver: "exp", recallTau: true, tauFloor: 0.6, suppPatience: 3, softFloor: true });
  w.write(`  boost(softFloor): stop=${resSF.stop} rounds=${resSF.rounds} G=${resSF.G.length}\n`);
  resSF.curve.slice(0, 4).forEach((c) => w.write(`    r${c.round}: rec=${c.recall.toFixed(2)} prec=${c.precision.toFixed(3)} (pMin=${pMin.toFixed(3)}) m=${c.m} |Q|=${c.size} unionRec=${c.unionRecall.toFixed(2)}\n`));

  // (B) round-0 raw: dominant size, mfit at the FULL dominant + at recall-grown prefixes (huge-bits question)
  const wt = new Float64Array(tr.length).fill(1 / tr.length);
  const { u, edges } = buildAffinity(tr, wt, neg, {});
  const r = replicate(u, edges, { solver: "exp", rho: RHO, suppPatience: 3 });
  const byW = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i]]).sort((a, b) => b[1] - a[1]).map((e) => e[0]);
  const Rdom = wOr(byW, tr, wt);
  const fitFull = mfit(sortAsc(byW), tr, wt, neg.Neg);
  const fitSF = mfit(sortAsc(byW), tr, wt, neg.Neg, { softFloor: true });
  w.write(`  round0: |dominant|=${byW.length} R_dom(OR)=${Rdom.toFixed(2)} | mfit(default): valid=${fitFull.valid} m=${fitFull.m} rec=${(fitFull.recall || 0).toFixed(2)} prec=${(fitFull.precision || 0).toFixed(3)} recAt1=${(fitFull.recallAt1 || 0).toFixed(2)}\n`);
  w.write(`          mfit(softFloor,minRecall=0.25): valid=${fitSF.valid} m=${fitSF.m} rec=${(fitSF.recall || 0).toFixed(2)} prec=${(fitSF.precision || 0).toFixed(3)} recAt1=${(fitSF.recallAt1 || 0).toFixed(2)} | pMin=${pMin.toFixed(3)} → precision<pMin? ${((fitSF.precision || 0) < pMin)}\n`);
  w.write(`  bits to reach weighted-OR recall: 0.5→${nToRecall(byW, tr, wt, 0.5)} 0.6→${nToRecall(byW, tr, wt, 0.6)} 0.7→${nToRecall(byW, tr, wt, 0.7)} 0.8→${nToRecall(byW, tr, wt, 0.8)} (of ${byW.length})\n`);
  // (C) full fitClass path — does softFloor survive the fitClass wrapper (forwarding + prefix early-stop)?
  const fc = fitClass(A, negPool, Mglob, { rho: RHO, delta: DBAND });
  const fcSF = fitClass(A, negPool, Mglob, { rho: RHO, delta: DBAND, softFloor: true });
  w.write(`  fitClass(default): G=${fc.G.length} rStar=${fc.rStar}/${fc.roundsTotal} | fitClass(softFloor): G=${fcSF.G.length} rStar=${fcSF.rStar}/${fcSF.roundsTotal} valRec=${(fcSF.valRecall || 0).toFixed(2)} valFP=${(fcSF.valConfFP || 0).toFixed(2)}\n`);
  if (fcSF.G.length) { const g0 = fcSF.G[0], nn = fcSF.neg.Neg, sq = [...g0.Qbits].sort((a, b) => a - b); const gfp = nn.filter((y) => andCount(y, sq) >= g0.m).length / nn.length; const hfp = nn.filter((y) => fcSF.head.fires(y)).length / nn.length; w.write(`  PROBE: G[0] m=${g0.m} |Q|=${g0.Qbits.length} α=${g0.alpha.toFixed(3)} | rawGateFP(fcNeg,${nn.length})=${gfp.toFixed(3)} tunedHeadFP=${hfp.toFixed(3)} θ=${fcSF.theta.toFixed(3)} Σα=${fcSF.G.reduce((s, g) => s + g.alpha, 0).toFixed(2)}\n`); }

  // (D) STRUCTURE: dominant core, ⋁A (union of ALL positive signatures), and ⋁A waste-removed (discriminative min-cover)
  const orRec = (bits, pool) => { if (!bits.length) return 0; const s = sortAsc(bits); return pool.length ? pool.filter((y) => andCount(y, s) >= 1).length / pool.length : 0; };
  w.write(`  CORE(dominant): |Q|=${byW.length} posRec(OR)=${orRec(byW, tr).toFixed(2)} negFP(OR)=${orRec(byW, neg.Neg).toFixed(3)}\n`);
  const vAset = new Set(); for (const y of tr) for (const b of y) vAset.add(b);
  const vA = sortAsc([...vAset]);
  w.write(`  ⋁A(all pos sigs): |bits|=${vA.length} posRec=${orRec(vA, tr).toFixed(2)} negFP(OR)=${orRec(vA, neg.Neg).toFixed(3)}\n`);
  const bitU = new Map(); for (let p = 0; p < u.length; p++) bitU.set(neg.pPlus[p], u[p]);
  const vAByU = [...vA].sort((a, b) => (bitU.get(b) ?? -1e9) - (bitU.get(a) ?? -1e9));
  const postings = new Map(); for (const b of vA) postings.set(b, []);
  for (let i = 0; i < tr.length; i++) for (const b of tr[i]) postings.get(b).push(i);
  for (const target of [0.5, 0.7, 0.9]) {                    // discriminative (u-desc) min-cover to hold `target` positive coverage
    const cov = new Uint8Array(tr.length); let c = 0; const bits = [];
    for (const b of vAByU) { let adds = false; for (const i of postings.get(b)) if (!cov[i]) { cov[i] = 1; c++; adds = true; } if (adds) bits.push(b); if (c / tr.length >= target) break; }
    const uHi = bitU.get(bits[0]), uLo = bitU.get(bits[bits.length - 1]);
    w.write(`  ⋁A min-cover@${target}: |bits|=${bits.length} negFP(OR)=${orRec(bits, neg.Neg).toFixed(3)} u[${uHi != null ? uHi.toFixed(2) : "NA"}..${uLo != null ? uLo.toFixed(2) : "NA"}]\n`);
  }

  // (E) ⋂/⋃ analysis — why does twoCore (⋂, ⋃−⋂) ≡ union? is ⋂ super small, or does the tail alone already = union?
  const Gd = fc.G.length ? fc.G : fcSF.G;
  if (Gd.length) {
    const uniS = new Set(); for (const g of Gd) for (const b of g.Qbits) uniS.add(b);
    let interB = [...Gd[0].Qbits]; for (const g of Gd.slice(1)) { const s = new Set(g.Qbits); interB = interB.filter((b) => s.has(b)); }
    const iSet = new Set(interB); const tailB = [...uniS].filter((b) => !iSet.has(b));
    const iCov = orRec(interB, tr), tCov = orRec(tailB, tr), uCov = orRec([...uniS], tr);
    w.write(`  ⋂/⋃: K=${Gd.length} |⋂|=${interB.length} |⋃|=${uniS.size} |tail|=${tailB.length} | ⋂cov=${iCov.toFixed(2)} tailcov=${tCov.toFixed(2)} ⋃cov=${uCov.toFixed(2)} | tail≈union? Δ=${Math.abs(tCov - uCov).toFixed(3)} | ⋂ adds over tail: ${(uCov - tCov).toFixed(3)}\n`);
  }
}
