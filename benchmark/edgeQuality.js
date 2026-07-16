"use strict";

// benchmark/edgeQuality.js — Monte-Carlo edge-quality graph over the global p⁺ (no mask).
//
// p⁺ = ⋁A (global OR, unmasked). Each iteration: shuffle p⁺'s bit order, chunk into equal
// parts. Each chunk Q is a part proposal; its mask is the REVERSE formula D_τ = p⁺ ∖ Q
// (M_τ = ¬D_τ = Q ∨ ¬p⁺), its negative channel p⁻ comes from the peel, and its quality is
// the local classifier's info gain separating A from a negative sample. That scalar quality
// is accumulated onto every bit-bit edge WITHIN the chunk (symmetric, no double count).
//
// After many iterations we ask: is a bit's edge-mass flat or peaked (row entropy)? What are
// the mean/std edge weights? Do LOW-strength bits recover the mask (low d(b)) and HIGH-strength
// bits the discriminative vocabulary? Which bits are broad CONNECTORS vs clique SPECIALISTS?
// Writes the averaged edge matrix for a later dominant-set pass.
//
// Usage: TARGET=bank R=5 REP=sparse ITERS=300 CHUNK=14 node --max-old-space-size=8192 benchmark/edgeQuality.js english.txt <corpus>

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { encodeWord } from "../src/core/signatures/wordEncoder.js";
import { fuzzyEncode } from "../src/core/parts/encoding/fuzzyEncode.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";
import peelNegative from "../src/core/discovery/peelNegative.js";
import fitThreshold from "../src/core/discovery/thresholdFit.js";
import pruneClique from "../src/core/discovery/pruneClique.js";
import { adaptedGateScore } from "../src/core/gate/adaptedGate.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGET = process.env.TARGET || "bank";
const R = Number(process.env.R || 5), D = R + 1, DD = Number(process.env.DELTA || 0.001);
const REP = process.env.REP || "sparse";
const ITERS = Number(process.env.ITERS || 300);
const CHUNK = Number(process.env.CHUNK || 14);
const NEG = Number(process.env.NEG || 1500);
const KAPPA = Number(process.env.KAPPA || 2), M = Number(process.env.M || 2);
const PRUNE = Number(process.env.PRUNE || 0);                 // attach the deterministic prune per chunk
const PRUNEMIN = Number(process.env.PRUNEMIN || 2);          // prune floor (1 ⇒ can drop to a single survivor)
const VALIDATE = Number(process.env.VALIDATE || 0);         // # held-out random parts to test connectivity vs true gain
const MASKCUT = Number(process.env.MASKCUT || 0.15);        // survival below this ⇒ mask bit (strip from p⁺ for masked span)
const lcg = (seed) => () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 0x100000000;

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F;
  const enc = REP === "sparse" ? encodeWord : fuzzyEncode;
  const pc = new Map(); const partsOf = (w) => { let a = pc.get(w); if (!a) { a = [...new Set(enc(dict, w))]; pc.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const ws = tokenizeStream(line.slice(seg.start, seg.end)).filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws);
  }
  const band = (dd) => (dd === 1 ? 2 : dd <= 3 ? 1 : 0);
  const stmp = new Int32Array(DIM).fill(-1); let sp = 0;
  const feat = (s, t) => { sp++; const y = []; for (let dd = 1; dd <= D; dd++) { const q = t - dd; if (q < 0) break; const off = band(dd) * F; for (const a of partsOf(s[q])) { const b = a + off; if (stmp[b] !== sp) { stmp[b] = sp; y.push(b); } } } return y.sort((a, b) => a - b); };

  const A = [], neg = []; let ni = 0, negTot = 0;
  for (const s of sents) for (let t = 1; t < s.length; t++) {
    if (s[t] === TARGET) A.push(feat(s, t));
    else { negTot++; if (ni++ % Math.max(1, Math.floor((sents.length * 8) / (NEG * 12))) === 0 && neg.length < NEG * 4) neg.push(feat(s, t)); }
  }
  const Neg = neg.filter((_, j) => j % Math.max(1, Math.floor(neg.length / NEG)) === 0);

  // global p⁺ (bits with ≥ PMINSUP support in A — drops fuzzy singletons, keeps the P×P matrix tractable)
  const PMINSUP = Number(process.env.PMINSUP || 1);
  const cntA = new Map(); for (const y of A) for (const b of y) cntA.set(b, (cntA.get(b) || 0) + 1);
  const pset = new Set(); for (const [b, c] of cntA) if (c >= PMINSUP) pset.add(b);
  const Pbits = Uint32Array.from(pset).sort(); const P = Pbits.length;
  const idxOf = new Map(); Pbits.forEach((b, i) => idxOf.set(b, i));

  // per-bit discriminativeness d(b) (for validation only — the accumulator never sees it)
  const cntN = new Map();
  for (const y of Neg) for (const b of y) if (pset.has(b)) cntN.set(b, (cntN.get(b) || 0) + 1);
  const epsD = 1 / (Neg.length + 1);
  const dOf = (b) => ((cntA.get(b) || 0) / A.length + epsD) / ((cntN.get(b) || 0) / Neg.length + epsD);

  // symmetric accumulators (upper triangle a<b), flattened
  const Wsum = new Float64Array(P * P), Wcnt = new Float64Array(P * P);
  const rnd = lcg(Number(process.env.SEED || 20260714) >>> 0);
  const order = Int32Array.from({ length: P }, (_, i) => i);
  const SNAP = Number(process.env.SNAP || 25);

  // per-bit strength from the current accumulator (averaged edges) — snapshotted to track convergence
  const computeStrength = () => { const s = new Float64Array(P); for (let a = 0; a < P; a++) { let sum = 0, n = 0; for (let b = 0; b < P; b++) { if (b === a) continue; const p = a < b ? a * P + b : b * P + a; if (Wcnt[p] > 0) { sum += Wsum[p] / Wcnt[p]; n++; } } s[a] = n ? sum / n : 0; } return s; };
  const topEdgeSet = (N) => { const es = []; for (let a = 0; a < P; a++) for (let b = a + 1; b < P; b++) { const p = a * P + b; if (Wcnt[p] > 0) es.push([p, Wsum[p] / Wcnt[p]]); } es.sort((x, y) => y[1] - x[1]); return new Set(es.slice(0, N).map((e) => e[0])); };
  const pearson = (x, y) => { const n = x.length; let mx = 0, my = 0; for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; } mx /= n; my /= n; let c = 0, vx = 0, vy = 0; for (let i = 0; i < n; i++) { const dx = x[i] - mx, dy = y[i] - my; c += dx * dy; vx += dx * dx; vy += dy * dy; } return c / Math.sqrt(vx * vy); };
  const rankOf = (arr) => { const idx = [...arr].map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]); const r = new Array(arr.length); idx.forEach(([, i], k) => { r[i] = k; }); return r; };
  const spearman = (x, y) => pearson(rankOf([...x]), rankOf([...y]));      // rank corr — robust to d's heavy tail
  const jaccard = (a, b) => { let inter = 0; for (const e of a) if (b.has(e)) inter++; return inter / (a.size + b.size - inter || 1); };
  const snaps = [];

  const w = process.stdout;
  w.write(`# edgeQuality — ${path.basename(corpusArg)} | TARGET=${TARGET} REP=${REP} | |A|=${A.length} |Neg|=${Neg.length} |p⁺|=${P} | ITERS=${ITERS} CHUNK=${CHUNK}\n`);
  const t0 = process.hrtime.bigint();

  const appearCnt = new Float64Array(P), surviveCnt = new Float64Array(P);  // for the prune-rate mask detector
  const emergeMap = new Map();                                            // survivor-set key → {count, gain, Q} (emergence)
  const allParts = [];                                                    // every pruned part: {g, Q, draw} — for good-part structure
  let qSum = 0, qCnt = 0;                                                 // baseline: mean part quality
  for (let it = 0; it < ITERS; it++) {
    for (let i = P - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); const tmp = order[i]; order[i] = order[j]; order[j] = tmp; }
    for (let start = 0; start < P; start += CHUNK) {
      const end = Math.min(start + CHUNK, P);
      const chunkIdx = []; for (let k = start; k < end; k++) chunkIdx.push(order[k]);
      if (chunkIdx.length < 2) continue;
      let survIdx = chunkIdx;                                             // indices (into Pbits) that keep an edge
      if (PRUNE) {
        const Qfull = chunkIdx.map((i) => Pbits[i]).sort((a, b) => a - b);
        const pr = pruneClique(Qfull, A, Neg, { minBits: PRUNEMIN });     // hold part fixed, drop unearned bits
        const surv = new Set(pr.Q);
        for (const i of chunkIdx) appearCnt[i]++;
        survIdx = chunkIdx.filter((i) => surv.has(Pbits[i]));
        for (const i of survIdx) surviveCnt[i]++;
      }
      if (survIdx.length < 2) { if (survIdx.length) qCnt++; continue; }   // pruned to a singleton → no edges
      const Q = survIdx.map((i) => Pbits[i]).sort((a, b) => a - b);
      const Qset = new Set(Q);
      const Dtau = [...Pbits].filter((b) => !Qset.has(b));                 // reverse formula: D_τ = p⁺ ∖ Q
      const { pMinus } = peelNegative(Q, Neg, A, { Dtau, kappa: KAPPA, m: M });
      const scoreFn = (x) => adaptedGateScore(x, { pPlusAdapted: Q, pMinus, dropSet: Dtau });
      const quality = fitThreshold(scoreFn, A, Neg).gain;                 // quality of the (pruned) local classifier
      qSum += quality; qCnt++;
      if (PRUNE) { const key = Q.join(","); let e = emergeMap.get(key); if (!e) { e = { count: 0, gain: 0, Q }; emergeMap.set(key, e); } e.count++; e.gain += quality; allParts.push({ g: quality, Q, draw: it }); }
      for (let u = 0; u < survIdx.length; u++) for (let v = u + 1; v < survIdx.length; v++) {
        let a = survIdx[u], b = survIdx[v]; if (a > b) { const t = a; a = b; b = t; }
        const p = a * P + b; Wsum[p] += quality; Wcnt[p] += 1;           // accumulate onto SURVIVOR edges only
      }
    }
    if ((it + 1) % SNAP === 0) snaps.push({ iter: it + 1, s: computeStrength(), top: topEdgeSet(Math.max(100, 2 * P)) });
  }
  const baseline = qSum / qCnt;                                           // E[part quality] over random parts
  const secs = Number(process.hrtime.bigint() - t0) / 1e9;

  // ── per-node edge statistics ──
  //   strength = MEAN edge (frequency-normalized)   deg   = #distinct partners (frequency-sensitive)
  //   totMass  = Σ edge weight (freq × quality)     wdeg  = Σ co-occurrence count (pure survival freq)
  const strength = new Float64Array(P), entropy = new Float64Array(P), deg = new Int32Array(P), totMass = new Float64Array(P), wdeg = new Float64Array(P);
  let gSum = 0, gSq = 0, gN = 0, gMin = Infinity, gMax = -Infinity;
  const avg = (a, b) => { const p = a < b ? a * P + b : b * P + a; return Wcnt[p] > 0 ? Wsum[p] / Wcnt[p] : 0; };
  for (let a = 0; a < P; a++) {
    let s = 0, n = 0, tm = 0, wd = 0; const row = [];
    for (let b = 0; b < P; b++) { if (b === a) continue; const p = a < b ? a * P + b : b * P + a; if (Wcnt[p] > 0) { const e = Wsum[p] / Wcnt[p]; s += e; row.push(e); tm += Wsum[p]; wd += Wcnt[p]; if (a < b) { gSum += e; gSq += e * e; gN++; if (e < gMin) gMin = e; if (e > gMax) gMax = e; } } }
    n = row.length; deg[a] = n; strength[a] = n ? s / n : 0; totMass[a] = tm; wdeg[a] = wd;
    const tot = row.reduce((x, y) => x + y, 0);
    let H = 0; if (tot > 0) { for (const e of row) { const pr = e / tot; if (pr > 0) H -= pr * Math.log(pr); } H /= Math.log(Math.max(2, n)); }
    entropy[a] = H;                                                       // normalized [0,1]: 1=flat, 0=peaked
  }
  const gMean = gSum / gN, gStd = Math.sqrt(gSq / gN - gMean * gMean);
  const order2 = Array.from({ length: P }, (_, i) => i);

  // PULL coefficient: corr( bit's edges , partners' strengths ). A passenger/mask bit's edges
  // are DETERMINED by whoever it is chunked with → high pull; a driver bit adds its own synergy
  // signal orthogonal to partner strength → lower pull. Answers "is a mask bit pulled by others?"
  const pull = new Float64Array(P);
  for (let a = 0; a < P; a++) { const ex = [], ey = []; for (let b = 0; b < P; b++) { if (b === a) continue; const p = a < b ? a * P + b : b * P + a; if (Wcnt[p] > 0) { ex.push(Wsum[p] / Wcnt[p]); ey.push(strength[b]); } } pull[a] = ex.length > 2 ? pearson(ex, ey) : 0; }

  // correlation strength ↔ log d(b): does the accumulator recover discriminativeness / the mask?
  const ld = Pbits.map((b) => Math.log(dOf(b)));
  const corr = pearson([...strength], [...ld]);

  // PRUNE-RATE mask detector: fraction of appearances a bit SURVIVES the per-chunk prune.
  // A mask bit is dropped from most chunks it lands in → low survival; a driver earns its place.
  const survRate = new Float64Array(P);
  for (let i = 0; i < P; i++) survRate[i] = appearCnt[i] > 0 ? surviveCnt[i] / appearCnt[i] : 0;
  const corrSurv = PRUNE ? pearson([...survRate], [...ld]) : NaN;

  // which per-node edge statistic recovers the mask (correlates with log d)?
  const corrDeg = pearson([...deg], [...ld]), corrTot = pearson([...totMass], [...ld]), corrWdeg = pearson([...wdeg], [...ld]);

  // mask vs discriminative groups (by the held-out d(b) the accumulator never saw)
  const DMASK = Number(process.env.DMASK || 1.3), DHI = Number(process.env.DHI || 3);
  const grp = (pred) => { const ix = order2.filter(pred); const mean = (f) => ix.reduce((a, i) => a + f(i), 0) / (ix.length || 1); return { n: ix.length, mS: mean((i) => strength[i]), mP: mean((i) => pull[i]), mSurv: mean((i) => survRate[i]), mDeg: mean((i) => deg[i]), mTot: mean((i) => totMass[i]), ix }; };
  const maskG = grp((i) => dOf(Pbits[i]) < DMASK), discG = grp((i) => dOf(Pbits[i]) >= DHI);
  const sepBelow = maskG.ix.filter((i) => strength[i] < discG.mS).length / (maskG.n || 1);

  // SPARSITY of the edge matrix: is the mass concentrated (structure) or spread (noise)?
  const edges = []; for (let a = 0; a < P; a++) for (let b = a + 1; b < P; b++) { const p = a * P + b; if (Wcnt[p] > 0) edges.push(Wsum[p] / Wcnt[p]); }
  edges.sort((x, y) => y - x); const total = edges.reduce((a, b) => a + b, 0);
  const massShare = (frac) => { const k = Math.max(1, Math.floor(edges.length * frac)); let s = 0; for (let i = 0; i < k; i++) s += edges[i]; return s / total; };
  const aboveThr = edges.filter((e) => e > gMean + 2 * gStd).length / edges.length;
  let gAcc = 0; { const asc = edges.slice().reverse(); let cum = 0; for (let i = 0; i < asc.length; i++) { cum += asc[i]; gAcc += cum; } } const gini = (edges.length + 1 - 2 * (gAcc / total)) / edges.length;

  w.write(`\n# ran ${secs.toFixed(1)}s | PRUNE=${PRUNE ? "on" : "off"} | baseline E[part-gain]=${baseline.toFixed(5)} | edge weight: mean=${gMean.toFixed(5)} std=${gStd.toFixed(5)} max=${gMax.toFixed(5)}\n`);
  // SPEARMAN (rank) corr with d — the honest signal (Pearson is washed out by d's heavy tail)
  const dArr = [...Pbits].map((b) => dOf(b));
  const spCorr = (arr) => spearman([...arr], dArr);
  const spS = spCorr(strength), spDeg = spCorr(deg), spTot = spCorr(totMass), spSurv = PRUNE ? spCorr(survRate) : NaN;
  w.write(`# spearman( node stat , d(b) ) — which edge statistic recovers the mask (rank corr):\n`);
  w.write(`#   strength(mean) ${spS.toFixed(3)}   degree ${spDeg.toFixed(3)}   totalMass ${spTot.toFixed(3)}${PRUNE ? `   survival ${spSurv.toFixed(3)}` : ""}\n`);
  const best = [["strength", spS], ["degree", spDeg], ["totalMass", spTot], ...(PRUNE ? [["survival", spSurv]] : [])].sort((a, b) => b[1] - a[1])[0];
  w.write(`#   → ${best[0]} (ρ=${best[1].toFixed(2)}) best recovers discriminativeness${best[1] > 0.25 ? " — mask bits ARE findable" : ""}\n`);

  // convergence: strength (fast — each bit in every partition) vs top-edge set (slow)
  if (snaps.length >= 2) {
    const fin = snaps[snaps.length - 1];
    w.write(`\n# CONVERGENCE (vs final@${fin.iter}it) — each pair sampled ~${(ITERS * CHUNK / P).toFixed(1)}× (≈ITERS·CHUNK/|p⁺|); each bit in all ${ITERS} parts\n   iter   corr(strength)  jaccard(top-edges)\n`);
    for (const sn of snaps) w.write(`   ${String(sn.iter).padStart(4)}      ${pearson(sn.s, fin.s).toFixed(4)}         ${jaccard(sn.top, fin.top).toFixed(3)}\n`);
  }

  w.write(`\n# SPARSITY — top 1%/5%/10% of edges hold ${(100 * massShare(0.01)).toFixed(0)}%/${(100 * massShare(0.05)).toFixed(0)}%/${(100 * massShare(0.1)).toFixed(0)}% of mass | >μ+2σ: ${(100 * aboveThr).toFixed(1)}% of edges | Gini=${gini.toFixed(3)}\n`);
  w.write(`# MASK (d<${DMASK}, n=${maskG.n}) vs DISC (d≥${DHI}, n=${discG.n}) node stats:\n`);
  w.write(`#   strength ${maskG.mS.toFixed(4)} vs ${discG.mS.toFixed(4)} | degree ${maskG.mDeg.toFixed(0)} vs ${discG.mDeg.toFixed(0)} | totalMass ${maskG.mTot.toFixed(2)} vs ${discG.mTot.toFixed(2)}${PRUNE ? ` | survival ${(100 * maskG.mSurv).toFixed(0)}% vs ${(100 * discG.mSurv).toFixed(0)}%` : ""}\n`);
  w.write(`#   → discriminative bits have ${(discG.mDeg / (maskG.mDeg || 1)).toFixed(1)}× the degree and ${(discG.mTot / (maskG.mTot || 1)).toFixed(1)}× the edge mass of mask bits\n`);
  // is the survival mask the same as the FREQUENCY (low-d) mask?
  { let sM = 0, dM = 0, inter = 0; for (let i = 0; i < P; i++) { const s = survRate[i] < MASKCUT, d = dOf(Pbits[i]) < DMASK; if (s) sM++; if (d) dM++; if (s && d) inter++; }
    w.write(`# MASK vs FREQUENCY: survival-mask=${sM} bits, freq-mask(d<${DMASK})=${dM} bits, overlap=${inter} (Jaccard ${(inter / (sM + dM - inter)).toFixed(2)}) — ${(100 * inter / (dM || 1)).toFixed(0)}% of freq-mask bits are survival-masked, but survival-mask is ${(sM / (dM || 1)).toFixed(1)}× larger (also strips rare specialists)\n\n`); }

  const show = (title, arr) => { w.write(`  ${title}\n    bit      d(b)   strength  entropy  pull  ${PRUNE ? "surv%  " : ""}deg\n`); for (const i of arr) w.write(`    ${String(Pbits[i]).padStart(6)}  ${dOf(Pbits[i]).toFixed(2).padStart(6)}   ${strength[i].toFixed(4)}   ${entropy[i].toFixed(3)}  ${pull[i].toFixed(2).padStart(5)}  ${PRUNE ? `${(100 * survRate[i]).toFixed(0).padStart(4)}%  ` : ""}${deg[i]}\n`); };
  show("TOP-12 by strength (discriminative core):", order2.slice().sort((a, b) => strength[b] - strength[a]).slice(0, 12));
  show("BOTTOM-12 by strength (mask candidates):", order2.slice().sort((a, b) => strength[a] - strength[b]).slice(0, 12));
  const sThr = gMean + gStd; const hi = order2.filter((i) => strength[i] > sThr);
  show("CONNECTORS (strength>μ+σ, flat/high-entropy, low pull):", hi.slice().sort((a, b) => (entropy[b] - pull[b]) - (entropy[a] - pull[a])).slice(0, 8));
  show("SPECIALISTS (strength>μ+σ, peaked/low-entropy):", hi.slice().sort((a, b) => entropy[a] - entropy[b]).slice(0, 8));

  // ── VALIDATE: does the matrix CONNECTIVITY score a part like its true classifier gain? ──
  if (VALIDATE > 0) {
    const rv = lcg((Number(process.env.SEED || 20260714) + 1) >>> 0);
    const cliqueSum = [], density = [], gains = [], SZ = CHUNK;          // FIXED size: raw Σ is meaningful
    for (let t = 0; t < VALIDATE; t++) {
      const sel = new Set(); while (sel.size < Math.min(SZ, P)) sel.add(Math.floor(rv() * P));
      const idxs = [...sel];
      let cs = 0, cn = 0; for (let u = 0; u < idxs.length; u++) for (let v = u + 1; v < idxs.length; v++) { cs += avg(idxs[u], idxs[v]); cn++; }   // each pair once — no double count
      cliqueSum.push(cs); density.push(cn ? cs / cn : 0);
      const Q = idxs.map((i) => Pbits[i]).sort((a, b) => a - b), Qset = new Set(Q);
      const Dtau = [...Pbits].filter((b) => !Qset.has(b));
      const { pMinus } = peelNegative(Q, Neg, A, { Dtau, kappa: KAPPA, m: M });
      gains.push(fitThreshold((x) => adaptedGateScore(x, { pPlusAdapted: Q, pMinus, dropSet: Dtau }), A, Neg).gain);
    }
    const gs = [...gains].sort((a, b) => a - b), pct = (q) => gs[Math.floor(q * (gs.length - 1))];
    w.write(`\n# VALIDATE — clique-edge SUM (Σ_{a<b∈Q} W_ab, no double count) vs TRUE gain, ${VALIDATE} RANDOM parts of size ${SZ}:\n`);
    w.write(`#   clique-sum: pearson=${pearson(cliqueSum, gains).toFixed(3)} spearman=${spearman(cliqueSum, gains).toFixed(3)}  |  density: pearson=${pearson(density, gains).toFixed(3)}\n`);
    w.write(`#   RANDOM-part gain: median=${pct(0.5).toFixed(3)} p90=${pct(0.9).toFixed(3)} p99=${pct(0.99).toFixed(3)} max=${pct(1).toFixed(3)} | baseline=${baseline.toFixed(3)} — random parts are mostly weak; a thin tail is strong\n`);
  }

  // ── EMERGENCE: do some survivor-parts recur across rounds, and do they score higher? ──
  if (PRUNE && emergeMap.size) {
    const parts = [...emergeMap.values()].map((e) => ({ count: e.count, mg: e.gain / e.count, Q: e.Q }));
    const recur = parts.filter((p) => p.count >= 2), once = parts.filter((p) => p.count === 1);
    const mean = (arr, f) => arr.reduce((a, x) => a + f(x), 0) / (arr.length || 1);
    const spC = spearman(parts.map((p) => p.count), parts.map((p) => p.mg));
    w.write(`\n# EMERGENCE — ${parts.length} distinct survivor-parts from ${qCnt} prunes | ${recur.length} recur (≥2×), ${once.length} one-off\n`);
    w.write(`#   mean gain: recurring=${mean(recur, (p) => p.mg).toFixed(4)}  one-off=${mean(once, (p) => p.mg).toFixed(4)}  baseline=${baseline.toFixed(4)}\n`);
    w.write(`#   spearman( recurrence-count , part gain ) = ${spC.toFixed(3)}   ← >0 ⇒ emerging parts score higher than the rest\n`);
    w.write(`#   mean size: recurring=${mean(recur, (p) => p.Q.length).toFixed(1)} vs one-off=${mean(once, (p) => p.Q.length).toFixed(1)}  (size confounds raw recurrence)\n`);
    // control the confound: within each size bucket, does recurrence still track gain?
    const bySize = new Map(); for (const p of parts) { const s = p.Q.length; if (!bySize.has(s)) bySize.set(s, []); bySize.get(s).push(p); }
    w.write(`   within-size recurrence↔gain (confound-controlled):\n`);
    for (const [s, ps] of [...bySize.entries()].sort((a, b) => a[0] - b[0])) { if (ps.length < 30) continue; const spS = spearman(ps.map((p) => p.count), ps.map((p) => p.mg)); w.write(`     size ${s}: n=${String(ps.length).padStart(4)}  recur=${String(ps.filter((p) => p.count >= 2).length).padStart(3)}  spearman(count,gain)=${spS.toFixed(3)}\n`); }
    const top = parts.sort((a, b) => b.count - a.count || b.mg - a.mg).slice(0, 6);
    w.write(`   top emerging parts (count × mean-gain):  {bit:d(b) …}\n`);
    for (const p of top) w.write(`    ${String(p.count).padStart(3)}×  g=${p.mg.toFixed(3)}   {${p.Q.map((b) => `${b}:${dOf(b).toFixed(1)}`).join("  ")}}\n`);
  }

  // ── GOOD PARTS structure: same draw? overlap? span p⁺? ──
  if (PRUNE && allParts.length) {
    const GOODFRAC = Number(process.env.GOODFRAC || 0.01);
    const sorted = allParts.slice().sort((a, b) => b.g - a.g);
    const K = Math.max(2, Math.floor(sorted.length * GOODFRAC)), good = sorted.slice(0, K), thr = good[good.length - 1].g;
    // same draw?
    const perDraw = new Map(); for (const p of good) perDraw.set(p.draw, (perDraw.get(p.draw) || 0) + 1);
    let samePairs = 0, totPairs = 0; for (let i = 0; i < good.length; i++) for (let j = i + 1; j < good.length; j++) { totPairs++; if (good[i].draw === good[j].draw) samePairs++; }
    // overlap + span
    const union = new Set(); let sumSz = 0; for (const p of good) { sumSz += p.Q.length; for (const b of p.Q) union.add(b); }
    const jac = (a, b) => { const A2 = new Set(a); let inter = 0; for (const x of b) if (A2.has(x)) inter++; return inter / (a.length + b.length - inter || 1); };
    let jsum = 0, jn = 0; const cap = Math.min(good.length, 80); for (let i = 0; i < cap; i++) for (let j = i + 1; j < cap; j++) { jsum += jac(good[i].Q, good[j].Q); jn++; }
    const freq = new Map(); for (const p of good) for (const b of p.Q) freq.set(b, (freq.get(b) || 0) + 1);
    // d-stratified span: what fraction of HIGH-d vs LOW-d bits do good parts cover?
    const DMASK = Number(process.env.DMASK || 1.3), DHI = Number(process.env.DHI || 3);
    let hiTot = 0, hiCov = 0, loTot = 0, loCov = 0; for (const b of Pbits) { const d = dOf(b); if (d >= DHI) { hiTot++; if (union.has(b)) hiCov++; } else if (d < DMASK) { loTot++; if (union.has(b)) loCov++; } }
    const topBits = [...freq.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
    w.write(`\n# GOOD PARTS — top ${(100 * GOODFRAC).toFixed(0)}% by gain (${good.length} parts, gain≥${thr.toFixed(3)}, mean size ${(sumSz / good.length).toFixed(1)}):\n`);
    w.write(`#   SAME DRAW? ${perDraw.size} distinct draws hold them (of ${ITERS} rounds); ${(100 * samePairs / (totPairs || 1)).toFixed(1)}% of good-part pairs share a draw ⇒ ${samePairs / (totPairs || 1) < 0.05 ? "spread across draws, ≤1 per draw" : "cluster in lucky draws"}\n`);
    w.write(`#   OVERLAP? |⋃bits|=${union.size} vs Σ|Q|=${sumSz} (${(sumSz / union.size).toFixed(2)}× reuse) | mean pairwise Jaccard=${(jsum / jn).toFixed(3)} ⇒ ${jsum / jn < 0.1 ? "largely disjoint" : "overlapping cores"}\n`);
    w.write(`#   SPAN p⁺? cover ${union.size}/${P} = ${(100 * union.size / P).toFixed(0)}% of p⁺ | of HIGH-d(≥${DHI}): ${(100 * hiCov / (hiTot || 1)).toFixed(0)}%  vs  LOW-d(<${DMASK}): ${(100 * loCov / (loTot || 1)).toFixed(0)}%\n`);
    // MASKED span: strip mask bits (survival < MASKCUT) from p⁺, recompute coverage against the discriminative core
    const kept = new Set(); let maskN = 0; for (let i = 0; i < P; i++) { if (survRate[i] < MASKCUT) maskN++; else kept.add(Pbits[i]); }
    let covKept = 0, unionInMask = 0; for (const b of union) { if (kept.has(b)) covKept++; else unionInMask++; }
    w.write(`#   MASKED span? mask=survival<${MASKCUT} strips ${maskN}/${P} bits ⇒ masked p⁺ = ${kept.size} bits:\n`);
    w.write(`#     good parts cover ${covKept}/${kept.size} = ${(100 * covKept / (kept.size || 1)).toFixed(0)}% of the MASKED p⁺ (vs ${(100 * union.size / P).toFixed(0)}% of full p⁺) | ${unionInMask} good-part bits fell in the mask\n`);
    w.write(`#   most-reused bits in good parts:  ${topBits.map(([b, c]) => `${b}:d${dOf(b).toFixed(1)}×${c}`).join("  ")}\n`);
  }

  // ── MAX-POOL of good parts vs the monolithic p⁺ classifier (does decomposition beat p⁺?) ──
  if (Number(process.env.MAXPOOL || 0) && PRUNE && allParts.length) {
    const GOODFRAC = Number(process.env.GOODFRAC || 0.01);
    const good = allParts.slice().sort((a, b) => b.g - a.g).slice(0, Math.max(2, Math.floor(allParts.length * GOODFRAC)));
    // each good part → gate with its reverse-formula mask D_τ,k = p⁺∖Q_k, peeled p⁻_k, fitted t_k
    const gates = good.map((p) => { const Qset = new Set(p.Q), Dtau = [...Pbits].filter((b) => !Qset.has(b)); const { pMinus } = peelNegative(p.Q, Neg, A, { Dtau, kappa: KAPPA, m: M }); const t = fitThreshold((x) => adaptedGateScore(x, { pPlusAdapted: p.Q, pMinus, dropSet: Dtau }), A, Neg).t; return { Q: p.Q, pMinus, Dtau, t }; });
    const mpMargin = (x) => { let best = -Infinity; for (const g of gates) { const s = adaptedGateScore(x, { pPlusAdapted: g.Q, pMinus: g.pMinus, dropSet: g.Dtau }) - g.t; if (s > best) best = s; } return best; };
    // monolithic p⁺: (|p⁺∧x| − |p⁻∧x|)/|x|, global peel, no mask
    const pAll = [...Pbits]; const { pMinus: pM } = peelNegative(pAll, Neg, A, { Dtau: [], kappa: KAPPA, m: M });
    const pScore = (x) => adaptedGateScore(x, { pPlusAdapted: pAll, pMinus: pM, dropSet: [] });
    // HARD negatives: the most p⁺-like contexts (easy negatives are 0 FP for both — meaningless)
    const hardNeg = neg.slice().sort((a, b) => pScore(b) - pScore(a)).slice(0, Math.min(2000, neg.length));
    const fpAtRec = (fn, r) => { const sA = A.map(fn).sort((a, b) => b - a); const thr = sA[Math.min(sA.length - 1, Math.max(0, Math.floor(r * A.length) - 1))]; let fp = 0; for (const y of hardNeg) if (fn(y) > thr) fp++; return fp / hardNeg.length; };
    w.write(`\n# MAX-POOL(${gates.length} good parts) vs monolithic p⁺ — FP%% at matched recall (vs ${hardNeg.length} HARD negatives):\n`);
    w.write(`   recall    p⁺ FP%    maxpool FP%    winner\n`);
    for (const r of [0.1, 0.2, 0.3, 0.5, 0.7]) { const fp0 = fpAtRec(pScore, r), fp1 = fpAtRec(mpMargin, r); w.write(`   ${(100 * r).toFixed(0).padStart(4)}%    ${(100 * fp0).toFixed(2).padStart(5)}%    ${(100 * fp1).toFixed(2).padStart(6)}%      ${fp1 < fp0 ? "max-pool" : "p⁺"}\n`); }

    // RESIDUALIZED ensemble: hard-calibrate thresholds, then greedily accept a part only while its
    // NEW TP beats its NEW FP (leave-one-out FP — the §5.0 fix the naive top-1% pool lacks).
    const gatesR = good.map((p) => { const Qset = new Set(p.Q), Dtau = [...Pbits].filter((b) => !Qset.has(b)); const { pMinus } = peelNegative(p.Q, hardNeg, A, { Dtau, kappa: KAPPA, m: M }); const sf = (x) => adaptedGateScore(x, { pPlusAdapted: p.Q, pMinus, dropSet: Dtau }); const t = fitThreshold(sf, A, hardNeg).t; return { sf, t }; });
    const firesA = gatesR.map((g) => A.map((x) => g.sf(x) > g.t)), firesN = gatesR.map((g) => hardNeg.map((x) => g.sf(x) > g.t));
    const posCov = new Array(A.length).fill(false), negFired = new Array(hardNeg.length).fill(false), used = new Array(gatesR.length).fill(false);
    let accepted = 0;
    while (true) {
      let best = -1, bestVal = 0;
      for (let k = 0; k < gatesR.length; k++) { if (used[k]) continue; let ntp = 0, nfp = 0; for (let i = 0; i < A.length; i++) if (firesA[k][i] && !posCov[i]) ntp++; for (let j = 0; j < hardNeg.length; j++) if (firesN[k][j] && !negFired[j]) nfp++; const val = ntp - nfp; if (val > bestVal) { bestVal = val; best = k; } }
      if (best < 0) break;
      used[best] = true; accepted++;
      for (let i = 0; i < A.length; i++) if (firesA[best][i]) posCov[i] = true;
      for (let j = 0; j < hardNeg.length; j++) if (firesN[best][j]) negFired[j] = true;
    }
    const recR = posCov.filter(Boolean).length / A.length, fpR = negFired.filter(Boolean).length / hardNeg.length, fpPatR = fpAtRec(pScore, recR);
    w.write(`# RESIDUALIZED ensemble (${accepted}/${gates.length} parts accepted): recall ${(100 * recR).toFixed(1)}%  FP ${(100 * fpR).toFixed(2)}%  vs  p⁺ FP ${(100 * fpPatR).toFixed(2)}% at the same recall  →  ${fpR < fpPatR ? "RESIDUALIZED WINS" : "p⁺ still ahead"}\n`);
  }

  // dump the averaged edge matrix + bit metadata for a later dominant-set pass
  const outAvg = new Float32Array(P * P);
  for (let a = 0; a < P; a++) for (let b = a + 1; b < P; b++) { const e = avg(a, b); outAvg[a * P + b] = e; outAvg[b * P + a] = e; }
  const dir = path.join(REPO, "benchmark");
  fs.writeFileSync(path.join(dir, `edges_${TARGET}.f32`), Buffer.from(outAvg.buffer));
  fs.writeFileSync(path.join(dir, `edges_${TARGET}.meta.json`), JSON.stringify({ target: TARGET, rep: REP, P, iters: ITERS, chunk: CHUNK, bits: [...Pbits], strength: [...strength], entropy: [...entropy], d: [...Pbits].map(dOf) }));
  w.write(`\n# edge matrix → benchmark/edges_${TARGET}.f32 (${P}×${P} f32) + edges_${TARGET}.meta.json  (for dominant-set)\n`);
};

main();
