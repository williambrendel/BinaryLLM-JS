"use strict";

// ============================================================================
// benchmark/trainLayer.js  —  Pass-1 DETECT-stage trainer (§5), one layer.
//
// Grows a monotone part library by residual-directed contrast voting, commits a
// closed-form Newton value per part, and drives to the kill-test T-03 (H2: do
// gates overlap on held-out?). Correctness asserts C-02/C-03/C-06/C-08 are wired
// live, not commented. Somas (§5 COMBINE) and Pass 2 are the next build.
//
//   Gain(p,t) = Σ_c (Σ_S r_i[c])² / (Σ_S h_i[c] + γ)     [per-class denom — C-03]
//   r_i[c] = 1[c=τ_i] − P_i[c],  h_i[c] = P_i[c](1−P_i[c])
//   w*[c]  = (Σ_S r_i[c]) / (Σ_S h_i[c] + γ)
//
// Usage: node benchmark/trainLayer.js <dict> <corpus> [rounds]
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { streamSignatureChunks } from "../src/core/signatures/streamSignatures.js";
import { fuzzyEncodeWord } from "../src/core/signatures/wordEncoder.js";
import { NO_LABEL } from "../src/core/signatures/signatureDataset.js";
import { Kind } from "../src/core/parts/kind.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };
const mkRng = (s) => () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const rng = mkRng(7);

// hyperparameters
const GAMMA = 1.0, LAMBDA = Number(process.env.LAM || 0.5), KAPPA = 48, ETA = 0.4;   // ETA = shrinkage
const REFIT_EVERY = Number(process.env.REFIT || 50), REFIT_SWEEPS = 8;   // §5.7b joint refit
const WCLIP = 2.0;   // trust region: |w[c]| ≤ 2 nats (|V|-independent; fixes (D) blowup as h→0)
const MINPREV = Number(process.env.MINPREV || 1);   // [L|C] eval: exclude positions with < MINPREV words of left context.
// For r1 the window is full at pos≥1, so pos=0 (C-only, no left word) is the ONLY lost cause — excluding it lifts the
// held-out edge over unigram 0.115→0.183 (Alice); MINPREV≥2 over-excludes legit easy openings and hurts. (train + eval).
const S_LO = 30, S_MAX = Number(process.env.SMAX || 3000);   // support band: BOTH edges ABSOLUTE COUNTS.
// s_hi as a fraction made the band-pass the optimizer at scale (|S| pinned at 0.05·N). A gate's breadth is
// a property of the language, not the corpus size — so the ceiling is a count, like s_min, applied everywhere.
const DELTA = 0.02, SEED_N = 8, REFINE_B = 8, RESTARTS = 5, REFINE_ITERS = 4;
const TOPFREQ = 200;                   // classes always kept as softmax candidates

// ---------------------------------------------------------------------------
const main = async () => {
  const [dictArg, corpusArg, roundsArg] = process.argv.slice(2);
  const ROUNDS = Number(roundsArg || 40);
  const dict = loadDict(dictArg);
  const F = dict.size();
  const parts = dict.allParts();

  // ---- load records: ctx=[L|C] (C offset by F), tgt=nextWord ----------------
  const vocabIndex = new Map(), vocab = [];
  const recs = []; let nLabel = 0;
  for await (const { records } of streamSignatureChunks(dict, corpusArg, { chunkSize: 20000, radius: 1, fuzzy: true, vocabIndex, vocab })) {
    for (const r of records) {
      if (r.nextWord === NO_LABEL) continue;
      nLabel++;
      if (r.pos < MINPREV) continue;   // insufficient left evidence — lost cause, excluded from train AND eval
      const ctx = new Int32Array(r.L.length + r.C.length); let k = 0;
      for (const a of r.L) ctx[k++] = a; for (const a of r.C) ctx[k++] = a + F;
      ctx.sort();
      recs.push({ ctx, tgt: r.nextWord });
    }
  }
  const V = vocab.length, N = recs.length;
  const tgtBag = vocab.map((w) => fuzzyEncodeWord(dict, Buffer.from(w).toString("latin1").toLowerCase()));

  // ---- contiguous split (protocol step 1) -----------------------------------
  const cut = Math.floor(N * 0.85);
  const train = recs.slice(0, cut), test = recs.slice(cut);
  const Ntr = train.length;

  // ---- unigram prior b (Laplace) --------------------------------------------
  const cnt = new Float64Array(V); for (const r of train) cnt[r.tgt]++;
  const b = new Float64Array(V), expB = new Float64Array(V); let sumExpB = 0;
  for (let c = 0; c < V; c++) { b[c] = Math.log((cnt[c] + 0.5) / (Ntr + 0.5 * V)); expB[c] = Math.exp(b[c]); sumExpB += expB[c]; }
  const freqClasses = [...cnt.keys()].sort((x, y) => cnt[y] - cnt[x]).slice(0, TOPFREQ);
  // global expected atom rate q_a = Σ_c P_uni(c)·1[a∈bag_c]  (for factored qhat, §Fix2)
  const qAtom = new Float64Array(F);
  for (let c = 0; c < V; c++) { const p = cnt[c] / Ntr; if (p === 0) continue; for (const a of tgtBag[c]) qAtom[a] += p; }

  // model: committed parts, and per-train-sample sparse log-odds correction
  const pool = [];                                   // {p:Set, t, w:Map<class,val>}
  const corr = train.map(() => new Map());           // corr_i : class -> Σ w_j[c] over fired j
  const clamp = (x) => (x > 30 ? 30 : x < -30 ? -30 : x);

  // exact normaliser Z_i via expB factoring — O(|corr_i|), NO sort (the hot path)
  const Zof = (i) => { const ci = corr[i]; let Z = sumExpB; for (const [c, v] of ci) Z += expB[c] * (Math.exp(clamp(v)) - 1); return Z; };
  // (probTop removed — direction-building no longer sorts per sample; §Fix2)

  // train cross-entropy (exact Z, no sort)
  const trainLL = () => { let s = 0; for (let i = 0; i < Ntr; i++) { const tgt = train[i].tgt; s += -Math.log(expB[tgt] * Math.exp(clamp(corr[i].get(tgt) || 0)) / Zof(i)); } return s / Ntr; };
  const testLL = () => {
    let s = 0;
    for (const r of test) {
      let Z = sumExpB; const ci = new Map();
      for (const pt of pool) { if (kBits(r.ctx, pt.bits) >= pt.t) for (const [c, v] of pt.w) ci.set(c, (ci.get(c) || 0) + v); }
      for (const [c, v] of ci) Z += expB[c] * (Math.exp(clamp(v)) - 1);
      s += -Math.log(expB[r.tgt] * Math.exp(clamp(ci.get(r.tgt) || 0)) / Z);
    }
    return s / test.length;
  };

  // ---- PERSISTENT residual state: invZ[i] = 1/Z_i, maintained incrementally ----
  // corr[i] IS logits_i (sparse). On commit we touch only S_j (≤ s_hi·N), never all N.
  const invZ = new Float64Array(Ntr).fill(1 / sumExpB);   // corr all empty ⇒ Z=sumExpB(=1)
  const rebuildInvZ = () => { for (let i = 0; i < Ntr; i++) invZ[i] = 1 / Zof(i); };   // full O(N) — only after refit
  // drift guard: recompute one sample's Z from scratch, assert incremental value matches
  const driftCheck = (i) => { const exact = 1 / Zof(i); return Math.abs(exact - invZ[i]) < 1e-9 * exact ? 0 : Math.abs(exact - invZ[i]); };

  // ---- EXACT incremental train cross-entropy (not amortized) ------------------
  // llTotal = Σ_i −log P_i(τ_i). A commit changes logits only on S_j, so we update
  // it in O(|S_j|) by bracketing each touched sample. Rebuilt O(N) only after refit.
  const negLogPtgt = (i) => -Math.log(expB[train[i].tgt] * Math.exp(clamp(corr[i].get(train[i].tgt) || 0)) * invZ[i]);
  let llTotal = 0; for (let i = 0; i < Ntr; i++) llTotal += negLogPtgt(i);
  const rebuildLL = () => { llTotal = 0; for (let i = 0; i < Ntr; i++) llTotal += negLogPtgt(i); };

  // popcount(ctx ∧ p)
  // fast popcount(p ∧ ctx): part as a bitset, ctx iterated with a bit-test (no hashing)
  const WBITS = (2 * F + 31) >> 5;
  const bitsetOf = (atoms) => { const b = new Uint32Array(WBITS); for (const a of atoms) b[a >> 5] |= (1 << (a & 31)); return b; };
  const kBits = (ctx, bits) => { let k = 0; for (const a of ctx) k += (bits[a >> 5] >>> (a & 31)) & 1; return k; };
  const allIdx = Int32Array.from({ length: Ntr }, (_, i) => i);

  // SWEEP over a PRECOMPUTED k array (kArr[j] ↔ idx[j]).  withCorr=false skips the
  // per-sample corr den-correction (invZ-only den) — fast, used for SEARCH RANKING;
  // withCorr=true is exact, used at commit.  Returns {t, gain} only.
  const sweepK = (kArr, idx, psz, withCorr, hiCnt = S_MAX) => {   // hiCnt = absolute-count ceiling (scaled to idx like the floor)
    const Ntot = idx.length, floorCnt = Math.max(5, (S_LO / Ntr) * Ntot), ceilCnt = (hiCnt / Ntr) * Ntot;
    const byK = Array.from({ length: psz + 1 }, () => ({ cnt: 0, sz: 0, sz2: 0, n: new Map(), cP: withCorr ? new Map() : null, cP2: withCorr ? new Map() : null }));
    for (let j = 0; j < Ntot; j++) {
      const i = idx[j], bk = byK[kArr[j]], iz = invZ[i];
      bk.cnt++; bk.sz += iz; bk.sz2 += iz * iz;
      bk.n.set(train[i].tgt, (bk.n.get(train[i].tgt) || 0) + 1);
      if (withCorr) for (const [c, v] of corr[i]) { const cv = clamp(v); bk.cP.set(c, (bk.cP.get(c) || 0) + (Math.exp(cv) - 1) * iz); bk.cP2.set(c, (bk.cP2.get(c) || 0) + (Math.exp(2 * cv) - 1) * iz * iz); }
    }
    let aCnt = 0, aSz = 0, aSz2 = 0; const aN = new Map(), aCP = new Map(), aCP2 = new Map();
    let bestT = -1, bestGain = -Infinity;
    for (let t = psz; t >= 1; t--) {
      const bk = byK[t]; aCnt += bk.cnt; aSz += bk.sz; aSz2 += bk.sz2;
      for (const [c, v] of bk.n) aN.set(c, (aN.get(c) || 0) + v);
      if (withCorr) { for (const [c, v] of bk.cP) aCP.set(c, (aCP.get(c) || 0) + v); for (const [c, v] of bk.cP2) aCP2.set(c, (aCP2.get(c) || 0) + v); }
      if (aCnt < floorCnt || aCnt > ceilCnt) continue;   // band-pass — both edges absolute counts
      let gain = 0;
      for (const c of (withCorr ? new Set([...aN.keys(), ...aCP.keys()]) : aN.keys())) {   // C-03: per-class denom
        const sumP = expB[c] * (aSz + (withCorr ? (aCP.get(c) || 0) : 0));
        const sumP2 = expB[c] * expB[c] * (aSz2 + (withCorr ? (aCP2.get(c) || 0) : 0));
        const num = (aN.get(c) || 0) - sumP; gain += num * num / (sumP - sumP2 + GAMMA);
      }
      if (gain > bestGain) { bestGain = gain; bestT = t; }
    }
    return bestT < 0 ? null : { t: bestT, gain: bestGain };
  };

  // exact S + w* at a chosen threshold, over the full-set k array (commit only)
  const computeSW = (kArr, bestT) => {
    const S = new Set(); let sSz = 0, sSz2 = 0; const sN = new Map(), sCP = new Map(), sCP2 = new Map();
    for (let i = 0; i < Ntr; i++) { if (kArr[i] < bestT) continue; S.add(i); const iz = invZ[i]; sSz += iz; sSz2 += iz * iz; sN.set(train[i].tgt, (sN.get(train[i].tgt) || 0) + 1); for (const [c, v] of corr[i]) { const cv = clamp(v); sCP.set(c, (sCP.get(c) || 0) + (Math.exp(cv) - 1) * iz); sCP2.set(c, (sCP2.get(c) || 0) + (Math.exp(2 * cv) - 1) * iz * iz); } }
    const wstar = new Map();
    for (const c of new Set([...sN.keys(), ...sCP.keys()])) { const sumP = expB[c] * (sSz + (sCP.get(c) || 0)); const sumP2 = expB[c] * expB[c] * (sSz2 + (sCP2.get(c) || 0)); const num = (sN.get(c) || 0) - sumP; const den = sumP - sumP2 + GAMMA; const w = num / den; if (Math.abs(w) > 1e-4) wstar.set(c, w); }
    if (wstar.size > KAPPA) { const keep = [...wstar.entries()].sort((a, z) => Math.abs(z[1]) - Math.abs(a[1])).slice(0, KAPPA); wstar.clear(); for (const [c, v] of keep) wstar.set(c, v); }
    for (const [c, v] of wstar) wstar.set(c, Math.max(-WCLIP, Math.min(WCLIP, v)));   // trust region (fix for h→0 at large |V|)
    return { S, wstar };
  };

  // ---- FIND_MASK: mixed atom+label directions → contrast seed ---------------
  // C-08: aligned set uses the residual in the DIRECTION's space (atom vs label)
  const alignedSet = (idx, dir) => {
    const A = [];
    if (dir.src === "atom") { const a = dir.id; for (const i of idx) if (tgtBag[train[i].tgt].includes(a)) A.push(i); }   // r^atom space
    else { const c = dir.cls; for (const i of idx) if (train[i].tgt === c) A.push(i); }                                    // label space (r_i[c])
    return A;
  };
  const contrastSeed = (idx, A) => {
    const inA = new Set(A); const nA = A.length, nB = idx.length - nA; if (!nA || !nB) return null;
    const fA = new Map(), fB = new Map();
    for (const i of idx) { const dst = inA.has(i) ? fA : fB; for (const a of train[i].ctx) dst.set(a, (dst.get(a) || 0) + 1); }
    const cst = [];
    for (const [a, ca] of fA) { const c = ca / nA - (fB.get(a) || 0) / nB; if (c > DELTA) cst.push([a, c]); }
    cst.sort((x, y) => y[1] - x[1]);
    return new Set(cst.slice(0, SEED_N).map((e) => e[0]));
  };

  const buildDirections = (idx) => {
    // ONE pass, NO per-sample sort. Accumulate residual mass into an F-array (atoms)
    // and a class map, via the 1/Z factoring (same as the denominator fix).
    let SinvZ = 0;
    const tgtCount = new Map(), tgtP = new Map(), corrCorr = new Map();
    const tgtAtomCount = new Float64Array(F), invZByAtom = new Float64Array(F);
    for (const i of idx) {
      const iz = invZ[i], t = train[i].tgt, ci = corr[i];
      SinvZ += iz;
      tgtCount.set(t, (tgtCount.get(t) || 0) + 1);
      tgtP.set(t, (tgtP.get(t) || 0) + expB[t] * Math.exp(clamp(ci.get(t) || 0)) * iz);   // P_i[τ_i], no sort
      for (const [c, v] of ci) corrCorr.set(c, (corrCorr.get(c) || 0) + (Math.exp(clamp(v)) - 1) * iz);
      for (const a of tgtBag[t]) { tgtAtomCount[a]++; invZByAtom[a] += iz; }
    }
    // atomTotalQ[a] = Σ_i E[a|P_i] = q_a·SinvZ + Σ_{boosted c} corrCorr[c]·expB[c]·1[a∈bag_c]
    const atomTotalQ = new Float64Array(F);
    for (let a = 0; a < F; a++) atomTotalQ[a] = qAtom[a] * SinvZ;
    for (const [c, cc] of corrCorr) { const wgt = cc * expB[c]; for (const a of tgtBag[c]) atomTotalQ[a] += wgt; }
    const dirs = [];
    // Kind-stratified atom directions: Σ_i|r^atom_i[a]| ≈ tgtAtomCount + atomTotalQ − 2·q_a·invZByAtom
    const bestK = new Map();  // kind -> [atom, mass]
    for (let a = 0; a < F; a++) {
      if (tgtAtomCount[a] === 0 && atomTotalQ[a] === 0) continue;
      const mass = Math.abs(tgtAtomCount[a] + atomTotalQ[a] - 2 * qAtom[a] * invZByAtom[a]);
      const k = parts[a].kind; const bk = bestK.get(k);
      if (!bk || mass > bk[1]) bestK.set(k, [a, mass]);
    }
    for (const K of [Kind.Start, Kind.End, Kind.Mid, Kind.Whole]) { const bk = bestK.get(K); if (bk) dirs.push({ src: "atom", id: bk[0] }); }
    // label directions: EXACT Σ_i|r_i[c]| = under + over (over-prediction kept), admitted at ≥ S_LO
    const clsMass = [];
    for (const [c, cc] of tgtCount) {
      if (cc < S_LO) continue;
      const tp = tgtP.get(c) || 0, totalP = expB[c] * (SinvZ + (corrCorr.get(c) || 0));
      clsMass.push([c, (cc - tp) + (totalP - tp)]);   // under + over
    }
    clsMass.sort((a, z) => z[1] - a[1]);              // one sort over admitted classes, not per-sample
    for (const [c] of clsMass.slice(0, 4)) dirs.push({ src: "label", cls: c });
    return dirs;
  };

  // ---- REFINE: ADD/DROP with incremental k via an inverted index ------------
  const refine = (p0, D) => {
    const nD = D.length;
    const atomToD = new Map();   // context atom -> D-positions containing it
    for (let j = 0; j < nD; j++) for (const a of train[D[j]].ctx) { let l = atomToD.get(a); if (!l) atomToD.set(a, (l = [])); l.push(j); }
    let p = new Set(p0);
    const k = new Int16Array(nD);
    for (const a of p) { const l = atomToD.get(a); if (l) for (const j of l) k[j]++; }
    let cur = sweepK(k, D, p.size, false); if (!cur) return null;
    for (let iter = 0; iter < REFINE_ITERS; iter++) {
      // S-conditioned shortlist: S = {j : k[j] >= cur.t}
      const score = new Map();
      for (let j = 0; j < nD; j++) { if (k[j] < cur.t) continue; for (const a of train[D[j]].ctx) if (!p.has(a)) score.set(a, (score.get(a) || 0) + 1); }
      const addCand = [...score.entries()].sort((x, y) => y[1] - x[1]).slice(0, REFINE_B).map((e) => e[0]);
      let best = cur, bestMove = null, bestK = null;
      for (const a of addCand) { const l = atomToD.get(a) || []; const k2 = k.slice(); for (const j of l) k2[j]++; const r = sweepK(k2, D, p.size + 1, false); if (r && r.gain > best.gain + LAMBDA) { best = r; bestMove = ["add", a]; bestK = k2; } }
      if (p.size > 1) for (const a of p) { const l = atomToD.get(a) || []; const k2 = k.slice(); for (const j of l) k2[j]--; const r = sweepK(k2, D, p.size - 1, false); if (r && r.gain > best.gain + LAMBDA) { best = r; bestMove = ["drop", a]; bestK = k2; } }
      if (!bestMove) break;
      if (bestMove[0] === "add") p.add(bestMove[1]); else p.delete(bestMove[1]);
      k.set(bestK); cur = best;
    }
    return { p, gain: cur.gain };
  };

  // ---- commit: exact w*/Gain on FULL train (bit-test k), verify vs λ ---------
  const commitPart = (p) => {
    const bits = bitsetOf(p), psz = p.size;
    const kArr = new Int16Array(Ntr); for (let i = 0; i < Ntr; i++) kArr[i] = kBits(train[i].ctx, bits);
    const r = sweepK(kArr, allIdx, psz, true, 2 * S_MAX);   // commit ceiling = 2× search (absolute), avoids integer-t-skip starvation
    if (!r) return null;
    const { S, wstar } = computeSW(kArr, r.t);
    return { p, t: r.t, w: wstar, S, bits, gain: r.gain, accepted: r.gain > LAMBDA };
  };

  // ---- §5.7b JOINT REFIT: coordinate descent over gates ---------------------
  // Each gate's w is re-fit (per-class Newton) against the residual left by ALL
  // OTHER gates — so co-firing gates stop double-counting the shared residual.
  // Convex; Gauss-Seidel sweeps converge. corr and gate.w are updated in lockstep.
  const jointRefit = () => {
    rebuildInvZ();   // refit rewrites all w → full O(N) rebuild (also resets float drift)
    for (let s = 0; s < REFIT_SWEEPS; s++) {
      for (const g of pool) {
        const S = g.S, w = g.w, cls = [...w.keys()]; if (!cls.length) continue;
        const cidx = new Map(); cls.forEach((c, k) => cidx.set(c, k));
        const n = new Float64Array(cls.length), sP = new Float64Array(cls.length), sPP = new Float64Array(cls.length);
        for (const i of S) {
          const iz = invZ[i], ci = corr[i];
          for (let k = 0; k < cls.length; k++) { const pic = expB[cls[k]] * Math.exp(clamp(ci.get(cls[k]) || 0)) * iz; sP[k] += pic; sPP[k] += pic * pic; }
          const kt = cidx.get(train[i].tgt); if (kt !== undefined) n[kt]++;
        }
        // Newton coordinate delta, then clip the RESULTING w to the trust region and
        // apply the SAME effective delta to corr (keeps corr = Σ gate.w consistent).
        const eff = new Float64Array(cls.length);
        for (let k = 0; k < cls.length; k++) { const c = cls[k]; const grad = (sP[k] - n[k]) + GAMMA * w.get(c); const hess = (sP[k] - sPP[k]) + GAMMA; const tgt = Math.max(-WCLIP, Math.min(WCLIP, w.get(c) - grad / hess)); eff[k] = tgt - w.get(c); }
        for (const i of S) {
          const ci = corr[i]; let Zi = 1 / invZ[i];
          for (let k = 0; k < cls.length; k++) { const d = eff[k]; if (!d) continue; const c = cls[k], oldc = ci.get(c) || 0, newc = oldc + d; Zi += expB[c] * (Math.exp(clamp(newc)) - Math.exp(clamp(oldc))); ci.set(c, newc); }
          invZ[i] = 1 / Zi;
        }
        for (let k = 0; k < cls.length; k++) w.set(cls[k], w.get(cls[k]) + eff[k]);
      }
    }
  };

  // =========================================================================
  const w = process.stdout;
  w.write(`# trainLayer — dict=${path.basename(dictArg)} corpus=${path.basename(corpusArg)} | F=${F} V=${V} | N=${N} (train ${Ntr} / test ${test.length}) | fuzzy r1\n`);
  w.write(`# [L|C] eval filter: MINPREV=${MINPREV} — excluded ${nLabel - N}/${nLabel} (${(100 * (nLabel - N) / nLabel).toFixed(1)}%) positions with < ${MINPREV} left words (BOS lost-cause, train+eval)\n`);
  const b0 = trainLL(), t0 = testLL();
  w.write(`baselines: unigram trainLL=${b0.toFixed(3)}  testLL=${t0.toFixed(3)}\n\n`);

  const n_D = Math.min(5000, Ntr);   // search subsample (§5.8): commit is still on full N
  const tOverP = [], gainCurves = [], c02 = [], ckpt = [];
  let prevLL = b0;   // llBefore = last round's llAfter (corr only changes at commit) — one trainLL/round
  for (let round = 0; round < ROUNDS; round++) {
    if (round > 0 && round % 50 === 0) {   // cheap insurance: incremental invZ AND llTotal must match a from-scratch recompute
      const d = driftCheck(round % Ntr); if (d > 0) w.write(`  ⚠️ invZ drift=${d.toExponential(1)} (incremental residual diverged)\n`);
      const exLL = trainLL(); if (Math.abs(exLL - llTotal / Ntr) > 1e-6) w.write(`  ⚠️ llTotal drift: incr=${(llTotal / Ntr).toFixed(6)} exact=${exLL.toFixed(6)}\n`);
    }   // deterministic idx — no RNG consumed
    // subsample D
    const D = []; { const seen = new Set(); const cap = Math.min(n_D, Ntr); while (D.length < cap) { const i = (rng() * Ntr) | 0; if (!seen.has(i)) { seen.add(i); D.push(i); } } }
    const dirs = buildDirections(D);
    // C-06/C-08 structural assert: atom dirs carry an atom id (< F namespace), label dirs a class id (< V)
    for (const d of dirs) { if (d.src === "atom") console.assert(d.id < F || (d.id >= F && d.id < 2 * F), "C-06: atom dir id in atom space"); else console.assert(d.cls < V, "C-08: label dir id in class space"); }
    let bestCand = null;
    for (let r = 0; r < RESTARTS && r < dirs.length; r++) {
      const A = alignedSet(D, dirs[r]); if (A.length < 5) continue;
      const seed = contrastSeed(D, A); if (!seed || !seed.size) continue;
      const ref = refine(seed, D); if (!ref) continue;
      if (!bestCand || ref.gain > bestCand.gain) bestCand = ref;
    }
    if (!bestCand) { w.write(`round ${round}: no candidate\n`); continue; }
    const llBefore = prevLL;   // cached (corr unchanged since last commit)
    const committed = commitPart(bestCand.p);
    if (!committed) { w.write(`round ${round}: no valid threshold on full set — stop\n`); break; }
    if (!committed.accepted) { w.write(`round ${round}: STOP — best full-set Gain_N=${committed.gain.toFixed(3)} ≤ λ=${LAMBDA}  → ${committed.gain < 0.05 ? "≈0: CORPUS EXHAUSTED" : "meaningful: λ WAS BINDING"}\n`); break; }
    // apply η·w (shrinkage / trust region) to corr of fired train samples
    const wEta = new Map([...committed.w].map(([c, v]) => [c, ETA * v]));
    for (const i of committed.S) {   // incremental: touch only S_j, update llTotal in O(|S_j|) — exact
      const ci = corr[i], tgt = train[i].tgt;
      llTotal += Math.log(expB[tgt] * Math.exp(clamp(ci.get(tgt) || 0)) * invZ[i]);   // remove old −logP(τ_i)
      let zi = 1 / invZ[i];
      for (const [c, v] of wEta) { const oldc = ci.get(c) || 0, newc = oldc + v; zi += expB[c] * (Math.exp(clamp(newc)) - Math.exp(clamp(oldc))); ci.set(c, newc); }
      invZ[i] = 1 / zi;
      llTotal -= Math.log(expB[tgt] * Math.exp(clamp(ci.get(tgt) || 0)) * invZ[i]);    // add new −logP(τ_i)
    }
    pool.push({ p: committed.p, t: committed.t, w: wEta, bits: committed.bits, S: committed.S });
    const llAfter = llTotal / Ntr; prevLL = llAfter;   // O(1) read — no O(N) pass
    // C-02: ΔLL_train ≈ Gain·η(1−η/2)  (shrinkage-aware Newton prediction)
    const dLL = (llBefore - llAfter) * Ntr;   // total nats
    const expDrop = committed.gain * ETA * (1 - ETA / 2);
    c02.push([expDrop, dLL]);
    tOverP.push(committed.t / committed.p.size);
    if (pool.length % REFIT_EVERY === 0) { jointRefit(); rebuildLL(); prevLL = llTotal / Ntr; w.write(`  [joint refit @K=${pool.length}: trainLL→${prevLL.toFixed(3)}]\n`); }
    // T-03 checkpoint every 10 parts: held-out fire DISTRIBUTION (median, ≤2 frac), vs K
    if (pool.length % 10 === 0) {
      const fires = test.map((r) => { let k = 0; for (const pt of pool) if (kBits(r.ctx, pt.bits) >= pt.t) k++; return k; });
      fires.sort((a, b) => a - b);
      const med = fires[fires.length >> 1], mean = fires.reduce((a, b) => a + b, 0) / fires.length, le2 = fires.filter((x) => x <= 2).length / fires.length;
      ckpt.push({ K: pool.length, mean, med, le2, tll: testLL() });
    }
    if (round % 10 === 0 || round === ROUNDS - 1) w.write(`round ${String(round).padStart(3)}: |pool|=${pool.length} |p|=${committed.p.size} t=${committed.t} |S|=${committed.S.size} gain=${committed.gain.toFixed(2)} ΔLL=${dLL.toFixed(1)} trainLL=${llAfter.toFixed(3)}\n`);
  }

  // final joint refit + validation: does removing double-counting reduce loss?
  const trPre = trainLL(), tllPre = testLL();
  jointRefit();
  const trPost = trainLL(), tllPost = testLL();

  // =========================================================================
  const fireDist = (samples) => { const f = samples.map((r) => { let k = 0; for (const pt of pool) if (kBits(r.ctx, pt.bits) >= pt.t) k++; return k; }); f.sort((a, b) => a - b); return f; };
  const ftr = fireDist(train), fte = fireDist(test);
  const stat = (f) => ({ mean: f.reduce((a, b) => a + b, 0) / f.length, med: f[f.length >> 1], le2: f.filter((x) => x <= 2).length / f.length, p90: f[Math.floor(f.length * 0.9)], max: f[f.length - 1] });
  const str = stat(ftr), ste = stat(fte);

  w.write(`\n================ RESULTS ================\n`);
  w.write(`final: |pool|=${pool.length}  trainLL=${trainLL().toFixed(3)}  testLL=${testLL().toFixed(3)}  (unigram test ${t0.toFixed(3)})\n\n`);

  // C-02 FIRST — numerics gate. Shrinkage-aware band 0.7–0.95 = (D) working; 0.3, >1, wrong sign = bug.
  const okC02 = c02.filter(([e]) => e > 2).map(([e, d]) => d / e);
  const ratio = okC02.length ? okC02.reduce((a, b) => a + b, 0) / okC02.length : NaN;
  const nNeg = c02.filter(([e, d]) => d < 0).length;
  w.write(`C-02 (ΔLL ≈ Gain·η(1−η/2), η=${ETA}):  mean ratio = ${ratio.toFixed(2)}  over ${okC02.length} parts  |  wrong-sign parts: ${nNeg}\n`);
  w.write(`   → ${ratio >= 0.55 && ratio <= 1.1 && nNeg === 0 ? "GREEN — numerics sound ((D) over-shoot expected below 1)" : "RED — residual/Hessian bug; do NOT trust T-03"}\n`);
  w.write(`C-03 (per-class denom): Σ_c num²/(den_c+γ), den_c=Σ_S P_i[c](1−P_i[c]) — structural in sweepK\n`);
  w.write(`§5.7b JOINT REFIT (final): trainLL ${trPre.toFixed(3)}→${trPost.toFixed(3)}   testLL ${tllPre.toFixed(3)}→${tllPost.toFixed(3)}   (drop ⇒ double-counting removed)\n\n`);

  w.write(`T-03 (H2 — gate overlap), as a DISTRIBUTION, held-out vs train:\n`);
  w.write(`   HELD-OUT: mean=${ste.mean.toFixed(2)} median=${ste.med} p90=${ste.p90} max=${ste.max}  |  frac firing ≤2 = ${(100 * ste.le2).toFixed(0)}%\n`);
  w.write(`   TRAIN:    mean=${str.mean.toFixed(2)} median=${str.med} p90=${str.p90} max=${str.max}  |  frac firing ≤2 = ${(100 * str.le2).toFixed(0)}%\n`);
  w.write(`   slope vs K (the real read — rising ⇒ H2 alive; flat at 1–2 while K grows ⇒ partition):\n`);
  w.write(`     K:      ` + ckpt.map((c) => String(c.K).padStart(6)).join("") + `\n`);
  w.write(`     mean:   ` + ckpt.map((c) => c.mean.toFixed(1).padStart(6)).join("") + `\n`);
  w.write(`     median: ` + ckpt.map((c) => String(c.med).padStart(6)).join("") + `\n`);
  w.write(`     ≤2 %:   ` + ckpt.map((c) => (100 * c.le2).toFixed(0).padStart(6)).join("") + `\n`);
  w.write(`     testLL: ` + ckpt.map((c) => c.tll.toFixed(3).padStart(6)).join("") + `   (unigram ${t0.toFixed(3)} — does adding low-gain parts help or hurt held-out?)\n`);
  w.write(`   → verdict deferred: read the SLOPE, not K=${pool.length}. (train≫test ⇒ memorisation.)\n\n`);

  const meanTP = tOverP.reduce((a, b) => a + b, 0) / (tOverP.length || 1);
  const interior = tOverP.filter((x) => x >= 0.2 && x <= 0.4).length;
  w.write(`D-02 (t/|p| interior?):  mean=${meanTP.toFixed(2)}  |  in [0.2,0.4]: ${interior}/${tOverP.length}  pinned@1.0: ${tOverP.filter((x) => x > 0.95).length}  (interior ⇒ not flat)\n`);
};

main();
