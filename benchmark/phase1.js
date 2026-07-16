"use strict";

// benchmark/phase1.js — Phase-1 StQP+Boost driver. Featurizes A (3F canon) + a confusable Neg pool,
// then runs the §9.1 test items. Starts with T1 (M density → data structure). Reuses canon featurization.
// Usage: TARGET=bank R=5 T=1 node --max-old-space-size=8192 benchmark/phase1.js english.txt <corpus>

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { encodeWord } from "../src/core/signatures/wordEncoder.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";
import buildNegSet from "../src/core/phase1/negSet.js";
import buildAffinity from "../src/core/phase1/affinity.js";
import replicate from "../src/core/phase1/replicator.js";
import boost from "../src/core/phase1/boost.js";
import makeHead from "../src/core/phase1/head.js";
import fitClass from "../src/core/phase1/fit.js";
import mfit from "../src/core/phase1/mfit.js";
import andCount from "../src/core/math/sparse/andCount.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGET = process.env.TARGET || "bank";
const R = Number(process.env.R || 5), D = R + 1, NEGPOOL = Number(process.env.NEGPOOL || 60000);

const main = () => {
  const _t0 = process.hrtime.bigint();
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

  // ── ADAPTCV: deployed boost vs adaptive-core (§8) — BOTH through the α-sum head (head.fires), 4-fold
  //    cross-word held-out. The real comparison: does FP-aware disjoint spawning beat post-hoc θ-tuning? ──
  if (process.env.ADAPTCV === "1") {
    const RHO = Number(process.env.RHO || 80), DDx = Number(process.env.DELTA || 0.05), DBx = Number(process.env.DBAND || 0.05);
    const TARGETS = (process.env.TARGETS || "time,world,state,music,century,river,government,bank,physics,philosophy,hydrogen").split(",");
    const tset = new Set(TARGETS), NEGP = Number(process.env.NEGPOOL || 60000);
    const classA = new Map(TARGETS.map((t) => [t, []])), negPool = [], bitCount = new Map();
    let Ntot = 0, ni = 0; const addC = (y) => { Ntot++; for (const b of y) bitCount.set(b, (bitCount.get(b) || 0) + 1); return y; };
    for (const s of sents) for (let t = 1; t < s.length; t++) { if (tset.has(s[t])) classA.get(s[t]).push(addC(feat(s, t))); else if (ni++ % Math.max(1, Math.floor((sents.length * 8) / NEGP)) === 0) negPool.push(addC(feat(s, t))); }
    const Mg = new Set(); for (const [b, c] of bitCount) if (c / Ntot > DDx) Mg.add(b);
    const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
    const cfgs = [["deployed (boost)", {}], ["adaptive-core", { core: "adaptive" }]];
    const w = process.stdout;
    w.write(`# ADAPTCV (deployed vs adaptive-core, α-sum head, 4-fold held-out) | words=${TARGETS.length} negPool=${negPool.length}\n`);
    w.write(`  word          |A|    ${cfgs.map(([n]) => `${n} (K rec%/FP%)`.padEnd(26)).join("")}\n`);
    const agg = cfgs.map(() => ({ rec: 0, fp: 0, k: 0, nw: 0 }));
    for (const word of TARGETS) {
      const Aw = classA.get(word); if (Aw.length < 8) continue;
      const cells = cfgs.map((c, ci) => {
        let rS = 0, fS = 0, kS = 0;
        for (let f = 0; f < 4; f++) {
          const fa = [], te = []; for (let k = 0; k < Aw.length; k++) (k % 4 === f ? te : fa).push(Aw[k]);
          const fit = fitClass(fa, negPool, Mg, { rho: RHO, delta: DBx, ...cfgs[ci][1] });
          rS += rate((y) => fit.head.fires(y), te); fS += rate((y) => fit.head.fires(y), fit.neg.Neg); kS += fit.G.length;
        }
        const rec = 100 * rS / 4, fp = 100 * fS / 4, k = kS / 4; agg[ci].rec += rec; agg[ci].fp += fp; agg[ci].k += k; agg[ci].nw++;
        return `${k.toFixed(0)} ${rec.toFixed(1)}/${fp.toFixed(1)}`.padEnd(26);
      });
      w.write(`  ${word.padEnd(12)}${String(Aw.length).padStart(5)}    ${cells.join("")}\n`);
    }
    w.write(`  ${"MEAN".padEnd(12)}${"".padStart(5)}    ${agg.map((a) => `${(a.k / a.nw).toFixed(0)} ${(a.rec / a.nw).toFixed(1)}/${(a.fp / a.nw).toFixed(1)}`.padEnd(26)).join("")}\n`);
    return;
  }

  // ── NORMCV: 4-fold cross-word CV of affinity normalization (tanh/sym) vs deployed, on held-out rec/FP.
  //    Featurizes ALL target words in one corpus pass. suppPatience applied uniformly for tractable runtime. ──
  if (process.env.NORMCV === "1") {
    const RHO = Number(process.env.RHO || 80), DDx = Number(process.env.DELTA || 0.05), DBx = Number(process.env.DBAND || 0.05);
    const TARGETS = (process.env.TARGETS || "time,world,state,music,century,river,government,bank,physics,philosophy,hydrogen").split(",");
    const tset = new Set(TARGETS), NEGP = Number(process.env.NEGPOOL || 60000);
    const classA = new Map(TARGETS.map((t) => [t, []])), negPool = [], bitCount = new Map();
    let Ntot = 0, ni = 0; const addC = (y) => { Ntot++; for (const b of y) bitCount.set(b, (bitCount.get(b) || 0) + 1); return y; };
    for (const s of sents) for (let t = 1; t < s.length; t++) { if (tset.has(s[t])) classA.get(s[t]).push(addC(feat(s, t))); else if (ni++ % Math.max(1, Math.floor((sents.length * 8) / NEGP)) === 0) negPool.push(addC(feat(s, t))); }
    const Mg = new Set(); for (const [b, c] of bitCount) if (c / Ntot > DDx) Mg.add(b);
    const orRec = (G, pool) => (pool.length ? pool.filter((y) => G.some((g) => andCount(y, g.Qbits) >= g.m)).length / pool.length : 0);
    const cfgs = [["deployed", {}], ["tanh T=2", { normEdges: "tanh", normT: 2 }], ["sym", { normEdges: "sym" }]];
    const w = process.stdout;
    w.write(`# NORMCV (4-fold cross-word, held-out) | words=${TARGETS.length} negPool=${negPool.length} | held rec% / FP%\n`);
    w.write(`  word          |A|   ${cfgs.map(([n]) => `${n}(rec/FP)`.padEnd(20)).join("")}\n`);
    const agg = cfgs.map(() => ({ rec: 0, fp: 0, nw: 0 }));
    for (const word of TARGETS) {
      const Aw = classA.get(word); if (Aw.length < 8) continue;
      const line = cfgs.map((c, ci) => {
        let rSum = 0, fSum = 0;
        for (let f = 0; f < 4; f++) {
          const trr = [], tee = []; for (let k = 0; k < Aw.length; k++) (k % 4 === f ? tee : trr).push(Aw[k]);
          const neg = buildNegSet(trr, negPool, Mg, { delta: DBx, maskNodes: new Set() });
          const r = boost(trr, neg, { rho: RHO, recallTau: true, tauFloor: 0.6, maxRounds: 50, suppPatience: 3, ...cfgs[ci][1] });
          rSum += orRec(r.G, tee); fSum += orRec(r.G, neg.Neg);
        }
        const rec = 100 * rSum / 4, fp = 100 * fSum / 4; agg[ci].rec += rec; agg[ci].fp += fp; agg[ci].nw++;
        return `${rec.toFixed(1)}/${fp.toFixed(1)}`.padEnd(20);
      });
      w.write(`  ${word.padEnd(12)}${String(Aw.length).padStart(5)}   ${line.join("")}\n`);
    }
    w.write(`  ${"MEAN".padEnd(12)}${"".padStart(5)}   ${agg.map((a) => `${(a.rec / a.nw).toFixed(1)}/${(a.fp / a.nw).toFixed(1)}`.padEnd(20)).join("")}\n`);
    return;
  }

  // ── XCLASS: cross-class PART overlap (do bank/state/time parts share bits?) + which band is the culprit. ──
  if (process.env.XCLASS === "1") {
    const RHO = Number(process.env.RHO || 80), DDx = Number(process.env.DELTA || 0.05), DBx = Number(process.env.DBAND || 0.05);
    const TARGETS = (process.env.TARGETS || "bank,state,time").split(","), tset = new Set(TARGETS);
    const classA = new Map(TARGETS.map((t) => [t, []])), negPool = [], bitCount = new Map();
    let Ntot = 0, ni = 0; const NEGP = Number(process.env.NEGPOOL || 60000);
    const addC = (y) => { Ntot++; for (const b of y) bitCount.set(b, (bitCount.get(b) || 0) + 1); return y; };
    for (const s of sents) for (let t = 1; t < s.length; t++) { if (tset.has(s[t])) classA.get(s[t]).push(addC(feat(s, t))); else if (ni++ % Math.max(1, Math.floor((sents.length * 8) / NEGP)) === 0) negPool.push(addC(feat(s, t))); }
    const Mglob = new Set(); for (const [b, c] of bitCount) if (c / Ntot > DDx) Mglob.add(b);
    const G = new Map(); for (const t of TARGETS) { const fit = fitClass(classA.get(t), negPool, Mglob, { rho: RHO, delta: DBx }); G.set(t, fit.G.map((g) => g.Qbits)); }
    const inB = (b, bd) => b >= bd * F && b < (bd + 1) * F, restr = (y, bd) => (bd < 0 ? y : y.filter((b) => inB(b, bd)));
    const jac = (a, b) => { const B = new Set(b); let it = 0; for (const x of a) if (B.has(x)) it++; const u = a.length + b.length - it; return u ? it / u : 0; };
    const mn = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0);
    // global-dedup compression headroom: Σ core bits vs unique bits across ALL words' cores (codebook ceiling)
    let sumBits = 0; const uniqAll = new Set(); const blk = new Map();
    for (const t of TARGETS) for (const q of G.get(t)) { sumBits += q.length; for (const b of q) uniqAll.add(b); }
    // recurring-block headroom: count bits shared by ≥2 distinct cores (candidates for a shared codebook entry)
    const bitCores = new Map(); for (const t of TARGETS) G.get(t).forEach((q, k) => { for (const b of q) { const key = `${t}#${k}`; if (!bitCores.has(b)) bitCores.set(b, new Set()); bitCores.get(b).add(key); } });
    let shared = 0; for (const [, s] of bitCores) if (s.size >= 2) shared++;
    process.stdout.write(`# GLOBAL CORE COMPRESSION | Σ|Q| across all cores=${sumBits}b  unique bits=${uniqAll.size}b  → dedup ceiling ${(sumBits / uniqAll.size).toFixed(2)}×  | bits in ≥2 cores=${shared} (${(100 * shared / uniqAll.size).toFixed(0)}% of unique — codebook candidates)\n`);
    process.stdout.write(`# CROSS-CLASS PART OVERLAP (Jaccard×100) | ${TARGETS.map((t) => `${t}:${G.get(t).length}p`).join(" ")}\n  band     intra(per class)          cross\n`);
    for (const [bd, nm] of [[-1, "3F-all"], [2, "F2-adj"], [1, "F1-near"], [0, "F0-far"]]) {
      const intra = TARGETS.map((t) => { const P = G.get(t), v = []; for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) v.push(jac(restr(P[i], bd), restr(P[j], bd))); return mn(v); });
      const cv = []; for (let ti = 0; ti < TARGETS.length; ti++) for (let tj = ti + 1; tj < TARGETS.length; tj++) { for (const a of G.get(TARGETS[ti])) for (const b of G.get(TARGETS[tj])) cv.push(jac(restr(a, bd), restr(b, bd))); }
      process.stdout.write(`  ${nm.padEnd(7)}  ${TARGETS.map((t, k) => `${t.slice(0, 4)}:${(100 * intra[k]).toFixed(0)}`).join(" ").padEnd(24)}  ${(100 * mn(cv)).toFixed(1)}\n`);
    }
    return;
  }

  // ── CLASS OVERLAP: intra- vs cross-class signature overlap (Jaccard) for the whole 3F signature and per
  // F-band (F2=adjacent dd1 · F1=near dd2-3 · F0=far dd≥4). Deterministic strided pair sampling. ──
  if (process.env.OVERLAP === "1") {
    const TARGETS = (process.env.TARGETS || "bank,state,time").split(","), tset = new Set(TARGETS);
    const ctx = new Map(TARGETS.map((t) => [t, []]));
    for (const s of sents) for (let t = 1; t < s.length; t++) if (tset.has(s[t])) ctx.get(s[t]).push(feat(s, t));
    const inB = (b, bd) => b >= bd * F && b < (bd + 1) * F;
    const restrict = (y, bd) => (bd < 0 ? y : y.filter((b) => inB(b, bd)));
    const jac = (a, b) => { if (!a.length && !b.length) return 0; const B = new Set(b); let it = 0; for (const x of a) if (B.has(x)) it++; const un = a.length + b.length - it; return un ? it / un : 0; };
    const stat = (arr) => { if (!arr.length) return [0, 0]; const m = arr.reduce((s, x) => s + x, 0) / arr.length; return [m, Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / arr.length)]; };
    const SAMPLE = Number(process.env.SAMPLE || 3000);
    const bands = [[-1, "3F-all"], [2, "F2-adj"], [1, "F1-near"], [0, "F0-far"]];
    process.stdout.write(`# CLASS OVERLAP (Jaccard×100, mean±sd) | ${TARGETS.map((t) => `${t}:${ctx.get(t).length}`).join(" ")}\n  band     ${TARGETS.map((t) => `intra-${t}`.padEnd(11)).join(" ")}  cross\n`);
    for (const [bd, nm] of bands) {
      const intra = TARGETS.map((t) => { const C = ctx.get(t), v = []; for (let i = 0; i < SAMPLE; i++) { const a = C[(i * 7) % C.length], b = C[(i * 13 + 1) % C.length]; if (a !== b) v.push(jac(restrict(a, bd), restrict(b, bd))); } return stat(v); });
      const cv = []; for (let ti = 0; ti < TARGETS.length; ti++) for (let tj = ti + 1; tj < TARGETS.length; tj++) { const Ci = ctx.get(TARGETS[ti]), Cj = ctx.get(TARGETS[tj]); for (let i = 0; i < SAMPLE; i++) cv.push(jac(restrict(Ci[(i * 7) % Ci.length], bd), restrict(Cj[(i * 13 + 1) % Cj.length], bd))); }
      const [cm, csd] = stat(cv);
      process.stdout.write(`  ${nm.padEnd(7)}  ${intra.map(([m, sd]) => `${(100 * m).toFixed(1)}±${(100 * sd).toFixed(1)}`.padEnd(11)).join(" ")}  ${(100 * cm).toFixed(1)}±${(100 * csd).toFixed(1)}\n`);
    }
    return;
  }

  const DD = Number(process.env.DELTA || 0.001), DBAND = Number(process.env.DBAND || 0.05);   // δ always positive
  const bitCount = new Map(); let Ntot = 0;
  const addC = (y) => { Ntot++; for (const b of y) bitCount.set(b, (bitCount.get(b) || 0) + 1); return y; };
  const A = [], negPool = []; let ni = 0, negTot = 0;
  for (const s of sents) for (let t = 1; t < s.length; t++) {
    if (s[t] === TARGET) A.push(addC(feat(s, t)));
    else { negTot++; if (ni++ % Math.max(1, Math.floor((sents.length * 8) / NEGPOOL)) === 0) negPool.push(addC(feat(s, t))); }
  }
  const Mglob = new Set(); for (const [b, c] of bitCount) if (c / Ntot > DD) Mglob.add(b);   // common-word bits

  const w = process.stdout;

  // ── DOMSET: pure-M dominant set (tiny tight cliques, recall<50%) + augment-to-N%-recall each AdaBoost round,
  //    vs deployed — through the real α-sum head. Tests the "shift-concentrate then extend the cores" idea. ──
  if (process.env.DOMSET === "1") {
    const RHO = Number(process.env.RHO || 80);
    const fitA = [], test = []; for (let k = 0; k < A.length; k++) (k % 4 === 3 ? test : fitA).push(A[k]);
    const rate = (pred, pool) => (pool.length ? pool.filter(pred).length / pool.length : 0);
    const SH = (process.env.DOMSHIFT || "").split(",").filter(Boolean).map(Number);
    const cfgs = SH.length
      ? [["deployed (dc, aug60)", {}], ...SH.map((s) => [`domset ×${s} +aug55`, { domset: true, domShift: s, recallTau: true, tauFloor: 0.55 }])]
      : [
        ["deployed (dc, aug60)", {}],
        ["domset raw (tiny, no aug)", { domset: true, recallTau: false }],
        ["domset + aug50", { domset: true, recallTau: true, tauFloor: 0.5 }],
        ["domset + aug55", { domset: true, recallTau: true, tauFloor: 0.55 }],
        ["domset + aug60", { domset: true, recallTau: true, tauFloor: 0.6 }],
      ];
    w.write(`# DOMSET (pure-M cliques + augment, via head) | TARGET=${TARGET} |A|=${A.length}\n`);
    w.write(`  config                       K  mean|Q|  held rec% / FP%\n`);
    for (const [nm, o] of cfgs) {
      const fit = fitClass(fitA, negPool, Mglob, { rho: RHO, delta: DBAND, ...o });
      const sz = fit.G.length ? fit.G.reduce((s, g) => s + g.Qbits.length, 0) / fit.G.length : 0;
      w.write(`  ${nm.padEnd(28)}${String(fit.G.length).padStart(2)}  ${sz.toFixed(0).padStart(5)}   ${(100 * rate((y) => fit.head.fires(y), test)).toFixed(1).padStart(5)}% / ${(100 * rate((y) => fit.head.fires(y), fit.neg.Neg)).toFixed(1).padStart(5)}%\n`);
    }
    return;
  }

  // ── HEADCMP: the FP reconciliation — OR-union (any part fires) vs the DEPLOYED α-sum head (θ-tuned) on the
  //    SAME held split. Shows why my sweeps report 20–30% (OR proxy) but the real classifier is far lower. ──
  if (process.env.HEADCMP === "1") {
    const RHO = Number(process.env.RHO || 80);
    const fitA = [], test = []; for (let k = 0; k < A.length; k++) (k % 4 === 3 ? test : fitA).push(A[k]);
    const fit = fitClass(fitA, negPool, Mglob, { rho: RHO, delta: DBAND, ...(process.env.SOLVER ? { solver: process.env.SOLVER } : {}) });
    const Neg = fit.neg.Neg;
    const rate = (pred, pool) => (pool.length ? pool.filter(pred).length / pool.length : 0);
    const orFire = (y) => fit.G.some((g) => andCount(y, g.Qbits) >= g.m);
    w.write(`# HEADCMP (OR-union vs deployed α-sum head) | TARGET=${TARGET} |A|=${A.length} K=${fit.G.length} rStar=${fit.rStar} θ=${fit.head.theta.toFixed(2)} negN=${Neg.length}\n`);
    w.write(`  OR-union (any part fires) : recHeld ${(100 * rate(orFire, test)).toFixed(1)}%   confFP ${(100 * rate(orFire, Neg)).toFixed(1)}%\n`);
    w.write(`  α-sum head (θ-tuned)      : recHeld ${(100 * rate((y) => fit.head.fires(y), test)).toFixed(1)}%   confFP ${(100 * rate((y) => fit.head.fires(y), Neg)).toFixed(1)}%\n`);
    return;
  }

  // ── SHIFTAUG: κ diagonal shift (concentrate → over-tight cores) THEN augment bits weight-first to 55% recall
  //    to buy recall back. Raw shifted core (control) vs shift+aug55, swept over κ, through the real head. ──
  if (process.env.SHIFTAUG === "1") {
    const RHO = Number(process.env.RHO || 80);
    const fitA = [], test = []; for (let k = 0; k < A.length; k++) (k % 4 === 3 ? test : fitA).push(A[k]);
    const rate = (pred, pool) => (pool.length ? pool.filter(pred).length / pool.length : 0);
    const KS = (process.env.KS || "0,15,30,45,60").split(",").map(Number);
    w.write(`# SHIFTAUG (shift + augment-to-55% recall, via head) | TARGET=${TARGET} |A|=${A.length}\n`);
    w.write(`  κ    raw over-tight core (K |Q| rec/FP)      shift+aug55 (K |Q| rec/FP)\n`);
    for (const kap of KS) {
      const shift = kap ? { shiftDiag: kap } : {};
      const raw = fitClass(fitA, negPool, Mglob, { rho: RHO, delta: DBAND, solver: "exp", recallTau: false, ...shift });
      const aug = fitClass(fitA, negPool, Mglob, { rho: RHO, delta: DBAND, solver: "exp", recallTau: true, tauFloor: 0.55, ...shift });
      const stat = (fit) => { const sz = fit.G.length ? fit.G.reduce((s, g) => s + g.Qbits.length, 0) / fit.G.length : 0; return `${String(fit.G.length).padStart(2)} ${sz.toFixed(0).padStart(3)} ${(100 * rate((y) => fit.head.fires(y), test)).toFixed(1)}/${(100 * rate((y) => fit.head.fires(y), fit.neg.Neg)).toFixed(1)}`; };
      w.write(`  ${String(kap).padStart(3)}  ${stat(raw).padEnd(35)} ${stat(aug)}\n`);
    }
    return;
  }

  // ── ABLATE: what does each pipeline component independently buy (rec/FP via the real head + fit time)?
  //    Isolates θ-tuning, support-early-stop, solver, and adaptive-core vs boost. ──
  if (process.env.ABLATE === "1") {
    const RHO = Number(process.env.RHO || 80);
    const fitA = [], test = []; for (let k = 0; k < A.length; k++) (k % 4 === 3 ? test : fitA).push(A[k]);
    const rate = (pred, pool) => (pool.length ? pool.filter(pred).length / pool.length : 0);
    const cfgs = [
      ["full (dc+θtune+earlystop)", {}],
      ["− θ-tune (½Σα thresh)", { tuneTheta: false }],
      ["− support early-stop", { suppPatience: 0 }],
      ["solver=exp (Pelillo-exp)", { solver: "exp" }],
      ["core=adaptive", { core: "adaptive" }],
      ["OR-union (no head, ref)", { tuneTheta: false, _or: true }],
    ];
    w.write(`# ABLATE | TARGET=${TARGET} |A|=${A.length}   (component → held rec% / confFP% / fit ms)\n`);
    for (const [nm, o] of cfgs) {
      const t0 = process.hrtime.bigint(); const fit = fitClass(fitA, negPool, Mglob, { rho: RHO, delta: DBAND, ...o }); const ms = Number(process.hrtime.bigint() - t0) / 1e6;
      const pred = o._or ? (y) => fit.G.some((g) => andCount(y, g.Qbits) >= g.m) : (y) => fit.head.fires(y);
      w.write(`  ${nm.padEnd(28)} ${(100 * rate(pred, test)).toFixed(1).padStart(5)}% / ${(100 * rate(pred, fit.neg.Neg)).toFixed(1).padStart(5)}% / ${ms.toFixed(0).padStart(5)}ms  (K=${fit.G.length})\n`);
    }
    return;
  }

  // ── PROF: per-word phase breakdown of the boost loop (where does extraction time go?). ──
  if (process.env.PROF === "1") {
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const prof = { affinity: 0, replicate: 0, support: 0, mfit: 0, reweight: 0, edges: 0, iters: 0, supp: 0, n: 0 };
    const t0 = process.hrtime.bigint();
    const res = boost(A, neg, { rho: 80, solver: "exp", recallTau: true, tauFloor: 0.6, prof });
    const tot = Number(process.hrtime.bigint() - t0) / 1e6;
    const pct = (x) => `${(100 * x / tot).toFixed(0)}%`;
    w.write(`# PROF (boost phase breakdown) | TARGET=${TARGET} |A|=${A.length} rounds=${res.rounds} n(P⁺)=${prof.n} edges/round≈${prof.edges}\n`);
    w.write(`  total ${tot.toFixed(0)}ms | replicate ${prof.replicate.toFixed(0)}ms ${pct(prof.replicate)} | affinity ${prof.affinity.toFixed(0)}ms ${pct(prof.affinity)} | mfit ${prof.mfit.toFixed(0)}ms ${pct(prof.mfit)} | support ${prof.support.toFixed(0)}ms ${pct(prof.support)} | reweight ${prof.reweight.toFixed(0)}ms ${pct(prof.reweight)}\n`);
    w.write(`  replicator: avg ${(prof.iters / res.rounds).toFixed(0)} iters/round, converged support ${(prof.supp / res.rounds).toFixed(0)} of ${prof.n} bits (${(100 * prof.supp / res.rounds / prof.n).toFixed(1)}% active) → matvec loops all ${prof.edges} edges but ~${(100 * (prof.supp / res.rounds) ** 2 / prof.edges).toFixed(1)}% touch two live bits\n`);
    return;
  }

  // ── SOLVERS: is M's spectrum vs ρ making the objective non-concave (⇒ slow convergence)? Estimate λ_max(M),
  //    check ρ>λ_max concavity, and compare std/exp/hybrid convergence + quality. ──
  if (process.env.SOLVERS === "1") {
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const RHO = Number(process.env.RHO || 80);
    const wU = new Float64Array(A.length).fill(1 / A.length);
    const { edges, n } = buildAffinity(A, wU, neg, { commonSet: neg.DtauSel });     // round-0 graph
    const { i, j, m } = edges;
    const matM = (x, out) => { out.fill(0); for (let e = 0; e < m.length; e++) { out[i[e]] += m[e] * x[j[e]]; out[j[e]] += m[e] * x[i[e]]; } };
    // power iteration for λ_max(M) (deterministic start), then on (λ_max·I − M) for the most-negative eigenvalue
    const pit = (shift) => { let x = new Float64Array(n).fill(1 / Math.sqrt(n)); const y = new Float64Array(n); let lam = 0;
      for (let t = 0; t < 300; t++) { matM(x, y); if (shift) for (let p = 0; p < n; p++) y[p] = shift * x[p] - y[p]; let nrm = 0; for (let p = 0; p < n; p++) nrm += y[p] * y[p]; nrm = Math.sqrt(nrm) || 1; for (let p = 0; p < n; p++) x[p] = y[p] / nrm; }
      matM(x, y); let num = 0, den = 0; for (let p = 0; p < n; p++) { num += x[p] * (shift ? shift * x[p] - y[p] : y[p]); den += x[p] * x[p]; } return num / (den || 1); };
    const lamMax = pit(0); const lamMinShifted = pit(Math.abs(lamMax) * 2 + 1); const lamMin = (Math.abs(lamMax) * 2 + 1) - lamMinShifted;
    w.write(`# SOLVERS (spectrum + convergence) | TARGET=${TARGET} |A|=${A.length} n(P⁺)=${n} edges=${m.length}\n`);
    w.write(`  round-0 M spectrum: λ_max≈${lamMax.toFixed(1)}  λ_min≈${lamMin.toFixed(1)}  |  ρ=${RHO} → (M−ρI) ${RHO > lamMax ? "NEG-DEF ✓ concave (should converge)" : "INDEFINITE ✗ non-concave (ρ<λ_max ⇒ slow/oscillating)"}\n`);
    // single round-0 replicate per solver: iters to converge + support
    for (const s of ["exp", "std", "hybrid"]) {
      const uw = buildAffinity(A, wU, neg, { commonSet: neg.DtauSel });
      const t0 = process.hrtime.bigint(); const r = replicate(uw.u, uw.edges, { solver: s, rho: RHO }); const ms = Number(process.hrtime.bigint() - t0) / 1e6;
      w.write(`  [round0] ${s.padEnd(6)}: ${r.iters} iters (${r.iters >= 500 ? "HIT CAP" : "converged"})  ${ms.toFixed(1)}ms  |supp|=${r.Q.length}  f=${r.f.toFixed(3)}\n`);
    }
    // full boost per solver: parts quality + total time
    const orRec = (G, pool) => (pool.length ? pool.filter((y) => G.some((g) => andCount(y, g.Qbits) >= g.m)).length / pool.length : 0);
    for (const s of ["exp", "std", "hybrid"]) {
      const prof = { affinity: 0, replicate: 0, support: 0, mfit: 0, reweight: 0, edges: 0, iters: 0, supp: 0, n: 0 };
      const t0 = process.hrtime.bigint(); const r = boost(A, neg, { rho: RHO, solver: s, recallTau: true, tauFloor: 0.6, prof }); const ms = Number(process.hrtime.bigint() - t0) / 1e6;
      w.write(`  [boost]  ${s.padEnd(6)}: ${ms.toFixed(0)}ms  avg ${(prof.iters / (r.rounds || 1)).toFixed(0)} iters/round  K=${r.G.length}  rec=${(100 * orRec(r.G, A)).toFixed(1)}%  FP=${(100 * orRec(r.G, neg.Neg)).toFixed(2)}%\n`);
    }
    return;
  }

  // ── NORM: does normalizing the affinity change part quality? tanh-squash (tame outlier edges) + symmetric
  //    degree-normalization, vs the deployed baseline, on the held split. ──
  if (process.env.NORM === "1") {
    const RHO = Number(process.env.RHO || 80);
    const tr = [], te = []; for (let k = 0; k < A.length; k++) (k % 4 === 3 ? te : tr).push(A[k]);
    const neg = buildNegSet(tr, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const orRec = (G, pool) => (pool.length ? pool.filter((y) => G.some((g) => andCount(y, g.Qbits) >= g.m)).length / pool.length : 0);
    const base = { rho: RHO, recallTau: true, tauFloor: 0.6, maxRounds: 50 };
    const cfgs = [
      ["deployed (no norm)", {}],
      ["tanh T=1", { normEdges: "tanh", normT: 1 }],
      ["tanh T=2", { normEdges: "tanh", normT: 2 }],
      ["tanh T=4", { normEdges: "tanh", normT: 4 }],
      ["sym (degree-norm)", { normEdges: "sym" }],
      ["sym + κ=30", { normEdges: "sym", shiftDiag: 30 }],
    ];
    w.write(`# NORM (affinity normalization) | TARGET=${TARGET} tr=${tr.length} te=${te.length} negN=${neg.negN}\n`);
    w.write(`  config                 K  mean|Q|  recTr  recHeld    FP\n`);
    for (const [nm, o] of cfgs) {
      const r = boost(tr, neg, { ...base, ...o });
      const meanSz = r.G.length ? r.G.reduce((s, g) => s + g.Qbits.length, 0) / r.G.length : 0;
      w.write(`  ${nm.padEnd(22)}${String(r.G.length).padStart(2)}  ${meanSz.toFixed(0).padStart(5)}  ${(100 * orRec(r.G, tr)).toFixed(1).padStart(5)}% ${(100 * orRec(r.G, te)).toFixed(1).padStart(5)}% ${(100 * orRec(r.G, neg.Neg)).toFixed(1).padStart(5)}%\n`);
    }
    return;
  }

  // ── KSWEEP: fine-tune the diagonal shift κ to trace the recall/FP frontier. Two families: raw-support+peel
  //    (κ = the only concentration knob) and recallTau@0.6+peel (κ concentrates, recallTau buys recall back). ──
  if (process.env.KSWEEP === "1") {
    const RHO = Number(process.env.RHO || 80);
    const tr = [], te = []; for (let k = 0; k < A.length; k++) (k % 4 === 3 ? te : tr).push(A[k]);
    const neg = buildNegSet(tr, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const orRec = (G, pool) => (pool.length ? pool.filter((y) => G.some((g) => andCount(y, g.Qbits) >= g.m)).length / pool.length : 0);
    const KS = (process.env.KS || "-20,0,15,30,45,60").split(",").map(Number);
    const run = (label, base) => {
      w.write(`  ${label}   κ:   K  mean|Q|  recTr  recHeld    FP     (FP/rec efficiency)\n`);
      for (const kappa of KS) {
        const r = boost(tr, neg, { rho: RHO, peel: true, maxRounds: 60, ...base, ...(kappa ? { shiftDiag: kappa } : {}) });
        const meanSz = r.G.length ? r.G.reduce((s, g) => s + g.Qbits.length, 0) / r.G.length : 0;
        const rh = orRec(r.G, te), fp = orRec(r.G, neg.Neg);
        w.write(`     κ=${String(kappa).padStart(3)}  ${String(r.G.length).padStart(2)}  ${meanSz.toFixed(0).padStart(5)}  ${(100 * orRec(r.G, tr)).toFixed(1).padStart(5)}% ${(100 * rh).toFixed(1).padStart(5)}% ${(100 * fp).toFixed(1).padStart(5)}%   ${rh > 0 ? (fp / rh).toFixed(3) : "-"}\n`);
      }
    };
    w.write(`# KSWEEP (κ frontier) | TARGET=${TARGET} |A|=${A.length} tr=${tr.length} te=${te.length} negN=${neg.negN}\n`);
    run("raw+peel   ", {});
    run("recallTau@0.6+peel", { recallTau: true, tauFloor: 0.6 });
    return;
  }

  // ── CORES: loose bundles (deployed) vs MANY TIGHT cores (concentration + peel). Does a union of small precise
  //    cliques hit the same recall at LOWER FP? Held-out split checks generalization. ──
  if (process.env.CORES === "1") {
    const RHO = Number(process.env.RHO || 80);
    // deterministic 75/25 train/held split of A
    const tr = [], te = []; for (let k = 0; k < A.length; k++) (k % 4 === 3 ? te : tr).push(A[k]);
    const neg = buildNegSet(tr, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const orRec = (G, pool) => (pool.length ? pool.filter((y) => G.some((g) => andCount(y, g.Qbits) >= g.m)).length / pool.length : 0);
    const perCore = (G, pool) => { if (!G.length || !pool.length) return [0, 0]; let rs = 0, fs = 0; for (const g of G) { rs += pool.filter((y) => andCount(y, g.Qbits) >= g.m).length / pool.length; } return rs / G.length; };
    const cfgs = [
      ["loose recallTau (deployed)", { solver: "exp", recallTau: true, tauFloor: 0.6, maxRounds: 50 }],
      ["loose +peel", { solver: "exp", recallTau: true, tauFloor: 0.6, peel: true, maxRounds: 50 }],
      ["tight recallTau@0.3 +peel", { solver: "exp", recallTau: true, tauFloor: 0.3, peel: true, maxRounds: 60 }],
      ["tight coreT=5 +peel", { solver: "exp", coreT: 5, peel: true, maxRounds: 60 }],
      ["tight coreT=10 +peel", { solver: "exp", coreT: 10, peel: true, maxRounds: 60 }],
    ];
    w.write(`# CORES (loose bundles vs many tight cliques) | TARGET=${TARGET} |A|=${A.length} tr=${tr.length} te=${te.length} negN=${neg.negN}\n`);
    w.write(`  config                       K  mean|Q| perCoreFP  UNION: recTr recHeld   FP    stop\n`);
    for (const [nm, o] of cfgs) {
      const r = boost(tr, neg, { rho: RHO, ...o });
      const meanSz = r.G.length ? r.G.reduce((s, g) => s + g.Qbits.length, 0) / r.G.length : 0;
      const pcf = perCore(r.G, neg.Neg);
      w.write(`  ${nm.padEnd(28)}${String(r.G.length).padStart(2)}  ${meanSz.toFixed(0).padStart(5)}   ${(100 * pcf).toFixed(1).padStart(5)}%   ${(100 * orRec(r.G, tr)).toFixed(1).padStart(5)}% ${(100 * orRec(r.G, te)).toFixed(1).padStart(5)}% ${(100 * orRec(r.G, neg.Neg)).toFixed(1).padStart(5)}%   ${r.stop}\n`);
    }
    return;
  }

  // ── SOLVER2: head-to-head of the user-researched adaptations on OUR signed sparse payoff π = u + Mx − ρx:
  //    small classic shift, continuous-time Euler, matrix-splitting (DC), Frank–Wolfe. All O(nnz), all measured to
  //    a common L1 tol (1e-6, cap 2000) so we can see which — if any — beats exp's 500-iter crawl. ──
  if (process.env.SOLVER2 === "1") {
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const RHO = Number(process.env.RHO || 80), TOL = 1e-6, CAP = 2000, n2 = A.length;
    const wU = new Float64Array(n2).fill(1 / n2);
    const { edges, u, n } = buildAffinity(A, wU, neg, { commonSet: neg.DtauSel });
    const { i, j, m } = edges, nnz = m.length;
    const piOf = (x, out) => { out.fill(0); for (let e = 0; e < nnz; e++) { out[i[e]] += m[e] * x[j[e]]; out[j[e]] += m[e] * x[i[e]]; } for (let p = 0; p < n; p++) out[p] += u[p] - RHO * x[p]; };  // π = u+Mx−ρx
    const Mx = (x, out) => { out.fill(0); for (let e = 0; e < nnz; e++) { out[i[e]] += m[e] * x[j[e]]; out[j[e]] += m[e] * x[i[e]]; } };
    const fObj = (x) => { const y = new Float64Array(n); Mx(x, y); let q = 0, lin = 0, nrm = 0; for (let p = 0; p < n; p++) { q += x[p] * y[p]; lin += u[p] * x[p]; nrm += x[p] * x[p]; } return lin + 0.5 * (q - RHO * nrm); };
    const supp = (x) => { const q = []; for (let p = 0; p < n; p++) if (x[p] > 1e-6 / n) q.push(p); return q; };
    let minM = 0; for (let e = 0; e < nnz; e++) if (m[e] < minM) minM = m[e];
    const l1 = (a, b) => { let s = 0; for (let p = 0; p < n; p++) s += Math.abs(a[p] - b[p]); return s; };
    w.write(`# SOLVER2 (user-researched adaptations) | TARGET=${TARGET} |A|=${n2} n=${n} nnz=${nnz} min M=${minM.toFixed(1)}\n`);

    // baseline exp (deployed)
    { const t0 = process.hrtime.bigint(); const r = replicate(u, edges, { solver: "exp", rho: RHO, maxIter: CAP, eps: TOL }); const ms = Number(process.hrtime.bigint() - t0) / 1e6;
      w.write(`  exp (deployed)      : ${String(r.iters).padStart(4)} it ${r.iters >= CAP ? "(CAP)" : "(conv)"}  ${ms.toFixed(0).padStart(5)}ms  |supp|=${String(r.Q.length).padStart(4)}  f=${r.f.toFixed(3)}\n`); }
    // small classic shift on the PURE-M dominant-set (drop u & −ρ): std with α=|min M|+ε
    { const alpha = Math.abs(minM) + 1e-6; let x = new Float64Array(n).fill(1 / n); const pi = new Float64Array(n), pr = new Float64Array(n); let it = 0;
      for (; it < CAP; it++) { pr.set(x); Mx(x, pi); let den = 0; for (let p = 0; p < n; p++) den += x[p] * (pi[p] + alpha); if (den <= 0) break; for (let p = 0; p < n; p++) x[p] = x[p] * (pi[p] + alpha) / den; if (l1(x, pr) < TOL) { it++; break; } }
      w.write(`  classic shift α=${alpha.toFixed(1)} (pure M): ${String(it).padStart(4)} it ${it >= CAP ? "(CAP)" : "(conv)"}         |supp|=${String(supp(x).length).padStart(4)}  f=${fObj(x).toFixed(3)}  [homogeneous, no u/ρ]\n`); }
    // continuous-time forward Euler on OUR payoff: x += dt·x·(π − ⟨x,π⟩)
    for (const dt of [0.05, 0.2, 1.0]) { let x = new Float64Array(n).fill(1 / n); const pi = new Float64Array(n), pr = new Float64Array(n); let it = 0;
      for (; it < CAP; it++) { pr.set(x); piOf(x, pi); let g = 0; for (let p = 0; p < n; p++) g += x[p] * pi[p]; let S = 0; for (let p = 0; p < n; p++) { x[p] = Math.max(x[p] + dt * x[p] * (pi[p] - g), 1e-12); S += x[p]; } for (let p = 0; p < n; p++) x[p] /= S; if (l1(x, pr) < TOL) { it++; break; } }
      const t0 = process.hrtime.bigint(); piOf(x, pi); const ms = Number(process.hrtime.bigint() - t0) / 1e6;
      w.write(`  continuous dt=${String(dt).padEnd(4)}    : ${String(it).padStart(4)} it ${it >= CAP ? "(CAP)" : "(conv)"}  ${ms.toFixed(0).padStart(5)}ms  |supp|=${String(supp(x).length).padStart(4)}  f=${fObj(x).toFixed(3)}\n`); }
    // matrix-splitting (DC): num=(M⁺x)+u⁺+β, den=(M⁻x)+u⁻+ρx+⟨x,π⟩+β
    { const beta = Math.abs(minM) + 1e-6; let x = new Float64Array(n).fill(1 / n); const pos = new Float64Array(n), negp = new Float64Array(n), pi = new Float64Array(n), pr = new Float64Array(n); let it = 0;
      for (; it < CAP; it++) { pr.set(x); pos.fill(0); negp.fill(0);
        for (let e = 0; e < nnz; e++) { const w2 = m[e]; if (w2 > 0) { pos[i[e]] += w2 * x[j[e]]; pos[j[e]] += w2 * x[i[e]]; } else { negp[i[e]] += -w2 * x[j[e]]; negp[j[e]] += -w2 * x[i[e]]; } }
        for (let p = 0; p < n; p++) { pos[p] += Math.max(u[p], 0); negp[p] += Math.max(-u[p], 0) + RHO * x[p]; pi[p] = pos[p] - negp[p]; }
        let g = 0; for (let p = 0; p < n; p++) g += x[p] * pi[p]; let S = 0;
        for (let p = 0; p < n; p++) { const num = pos[p] + beta, den = negp[p] + g + beta; x[p] = den > 0 ? x[p] * num / den : x[p]; S += x[p]; }
        for (let p = 0; p < n; p++) x[p] /= S; if (l1(x, pr) < TOL) { it++; break; } }
      w.write(`  matrix-split β=${beta.toFixed(1)}   : ${String(it).padStart(4)} it ${it >= CAP ? "(CAP)" : "(conv)"}         |supp|=${String(supp(x).length).padStart(4)}  f=${fObj(x).toFixed(3)}\n`); }
    // Frank–Wolfe (away-step-free) with exact line search on the concave quadratic
    { let x = new Float64Array(n).fill(1 / n); const pi = new Float64Array(n), d = new Float64Array(n), Md = new Float64Array(n); let it = 0;
      for (; it < CAP; it++) { piOf(x, pi); let jm = 0; for (let p = 1; p < n; p++) if (pi[p] > pi[jm]) jm = p;   // argmax gradient
        for (let p = 0; p < n; p++) d[p] = (p === jm ? 1 : 0) - x[p]; let gd = 0; for (let p = 0; p < n; p++) gd += pi[p] * d[p]; if (gd <= TOL) { it++; break; }
        Mx(d, Md); let curv = 0, nrm = 0; for (let p = 0; p < n; p++) { curv += d[p] * Md[p]; nrm += d[p] * d[p]; } curv -= RHO * nrm;   // dᵀ(M−ρI)d
        let gam = curv < 0 ? -gd / curv : 1; gam = Math.max(0, Math.min(1, gam));
        for (let p = 0; p < n; p++) x[p] += gam * d[p]; if (gam < 1e-9) { it++; break; } }
      const t0 = process.hrtime.bigint(); piOf(x, pi); const ms = Number(process.hrtime.bigint() - t0) / 1e6;
      w.write(`  frank-wolfe         : ${String(it).padStart(4)} it ${it >= CAP ? "(CAP)" : "(conv)"}  ${ms.toFixed(0).padStart(5)}ms  |supp|=${String(supp(x).length).padStart(4)}  f=${fObj(x).toFixed(3)}  [≥|supp| iters: adds 1 bit/step]\n`); }
    // does the better-optimum continuous solver yield better PARTS in full boost? (vs deployed exp, ± early-stop)
    const orRec = (G, pool) => (pool.length ? pool.filter((y) => G.some((g) => andCount(y, g.Qbits) >= g.m)).length / pool.length : 0);
    w.write(`  — full boost (recallTau) —\n`);
    for (const [nm, o] of [["exp (deployed)", { solver: "exp" }], ["continuous dt=0.2", { solver: "cont", dt: 0.2 }], ["cont + suppPatience=3", { solver: "cont", dt: 0.2, suppPatience: 3 }]]) {
      const prof = { affinity: 0, replicate: 0, support: 0, mfit: 0, reweight: 0, edges: 0, iters: 0, supp: 0, n: 0 };
      const t0 = process.hrtime.bigint(); const r = boost(A, neg, { rho: RHO, recallTau: true, tauFloor: 0.6, prof, ...o }); const ms = Number(process.hrtime.bigint() - t0) / 1e6;
      w.write(`    ${nm.padEnd(22)}: ${ms.toFixed(0).padStart(5)}ms  avg ${(prof.iters / (r.rounds || 1)).toFixed(0).padStart(3)} it/rnd  K=${r.G.length}  rec=${(100 * orRec(r.G, A)).toFixed(1)}%  FP=${(100 * orRec(r.G, neg.Neg)).toFixed(2)}%\n`);
    }
    return;
  }

  // ── WARMSTART: seed the replicator's x0 from M's top-positive eigenvector (sparse power iteration) instead of
  //    uniform. The clique ↔ top eigenvector (Motzkin-Straus); a good x0 may cut the iteration count. ──
  if (process.env.WARMSTART === "1") {
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const RHO = Number(process.env.RHO || 80), PISTEP = Number(process.env.PISTEP || 50);
    const wU = new Float64Array(A.length).fill(1 / A.length);
    const { edges, u, n } = buildAffinity(A, wU, neg, { commonSet: neg.DtauSel });
    const { i, j, m } = edges, nnz = m.length;
    const mv = (v, y) => { y.fill(0); for (let e = 0; e < nnz; e++) { y[i[e]] += m[e] * v[j[e]]; y[j[e]] += m[e] * v[i[e]]; } };
    let sigma;
    if (process.env.TIGHTSIG === "1") {                    // σ = |λ_min|·1.05 (tight) — let M's structure show through
      let v = new Float64Array(n).fill(1 / Math.sqrt(n)); const y = new Float64Array(n); let lam = 0;
      for (let t = 0; t < 200; t++) { mv(v, y); let nrm = 0; for (let p = 0; p < n; p++) nrm += y[p] * y[p]; nrm = Math.sqrt(nrm) || 1; for (let p = 0; p < n; p++) v[p] = y[p] / nrm; }
      mv(v, y); for (let p = 0; p < n; p++) lam += v[p] * y[p];   // Rayleigh: largest-|λ| (the most negative)
      sigma = Math.abs(lam) * 1.05;
    } else {                                                // Gershgorin σ = max_i Σ_j|M_ij| (conservative, safe PSD)
      const absRow = new Float64Array(n); for (let e = 0; e < nnz; e++) { absRow[i[e]] += Math.abs(m[e]); absRow[j[e]] += Math.abs(m[e]); }
      sigma = 0; for (let p = 0; p < n; p++) if (absRow[p] > sigma) sigma = absRow[p];
    }
    const eig = (K) => {                                    // K shifted power steps, deterministic start
      let v = new Float64Array(n); for (let p = 0; p < n; p++) v[p] = 1 + 0.1 * Math.sin(p);   // break symmetry, no RNG
      const y = new Float64Array(n);
      for (let t = 0; t < K; t++) { y.fill(0); for (let e = 0; e < nnz; e++) { y[i[e]] += m[e] * v[j[e]]; y[j[e]] += m[e] * v[i[e]]; } let nrm = 0; for (let p = 0; p < n; p++) { y[p] += sigma * v[p]; nrm += y[p] * y[p]; } nrm = Math.sqrt(nrm) || 1; for (let p = 0; p < n; p++) v[p] = y[p] / nrm; }
      return v;
    };
    const toSimplex = (v) => {                              // pick the heavier signed lobe, project to Δ
      let sp = 0, sn = 0; for (let p = 0; p < n; p++) { if (v[p] > 0) sp += v[p]; else sn -= v[p]; }
      const sgn = sp >= sn ? 1 : -1; const x0 = new Float64Array(n); let S = 0;
      for (let p = 0; p < n; p++) { const val = sgn * v[p]; if (val > 0) { x0[p] = val; S += val; } }
      if (S <= 0) return new Float64Array(n).fill(1 / n); for (let p = 0; p < n; p++) x0[p] /= S; return x0;
    };
    const t0 = process.hrtime.bigint(); const vEig = eig(PISTEP); const x0 = toSimplex(vEig); const eigMs = Number(process.hrtime.bigint() - t0) / 1e6;
    let nzX0 = 0; for (let p = 0; p < n; p++) if (x0[p] > 0) nzX0++;
    w.write(`# WARMSTART (eigvec x0 vs uniform) | TARGET=${TARGET} |A|=${A.length} n=${n} | σ=${sigma.toFixed(0)} PISTEP=${PISTEP} eigvec ${eigMs.toFixed(1)}ms, |x0>0|=${nzX0}\n`);
    for (const [nm, o] of [["uniform x0", {}], [`eigvec x0`, { x0 }]]) {
      const t1 = process.hrtime.bigint(); const r = replicate(u, edges, { solver: "exp", rho: RHO, ...o }); const ms = Number(process.hrtime.bigint() - t1) / 1e6;
      w.write(`  [round0 exp] ${nm.padEnd(12)}: ${String(r.iters).padStart(3)} iters ${r.iters >= 500 ? "(CAP)" : "(conv)"}  ${ms.toFixed(1).padStart(6)}ms  |supp|=${r.Q.length}  f=${r.f.toFixed(3)}\n`);
    }
    // does eigvec x0 land ON the clique? overlap of top-|x0| bits with the uniform-run support
    const base = replicate(u, edges, { solver: "exp", rho: RHO });
    const baseSet = new Set(base.Q); let hit = 0, tot = 0; for (let p = 0; p < n; p++) if (x0[p] > 0) { tot++; if (baseSet.has(p)) hit++; }
    w.write(`  eigvec-x0 support ∩ converged clique: ${hit}/${tot} of x0 bits are in the final support (|clique|=${base.Q.length})\n`);
    return;
  }

  // ── MATVEC: how sparse is M, and is a DENSE matvec actually faster (contiguous, no gather)? Same dynamics ⇒
  //    same iteration count, so this only compares cost-per-matvec + memory. ──
  if (process.env.MATVEC === "1") {
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const wU = new Float64Array(A.length).fill(1 / A.length);
    const { edges, u, n } = buildAffinity(A, wU, neg, { commonSet: neg.DtauSel });
    const { i, j, m } = edges, nnz = m.length;
    const dens = nnz / (n * (n - 1) / 2);   // fraction of the upper triangle populated
    const denseBytes = n * n * 8, sparseBytes = nnz * (4 + 4 + 8);
    // sparse matvec (as in replicator.payoff, the M·x part)
    const x = new Float64Array(n).fill(1 / n), out = new Float64Array(n);
    const REP = 200;
    const spMV = () => { out.fill(0); for (let e = 0; e < nnz; e++) { const a = i[e], b = j[e], w = m[e]; out[a] += w * x[b]; out[b] += w * x[a]; } };
    let t0 = process.hrtime.bigint(); for (let r = 0; r < REP; r++) spMV(); const spMs = Number(process.hrtime.bigint() - t0) / 1e6 / REP;
    // dense matvec — only materialize if it fits (guard huge n)
    let deMs = NaN, built = false;
    if (denseBytes < 1.5e9) {
      const D = new Float64Array(n * n); for (let e = 0; e < nnz; e++) { D[i[e] * n + j[e]] = m[e]; D[j[e] * n + i[e]] = m[e]; } built = true;
      t0 = process.hrtime.bigint(); for (let r = 0; r < REP; r++) { out.fill(0); for (let a = 0; a < n; a++) { let s = 0; const row = a * n; for (let b = 0; b < n; b++) s += D[row + b] * x[b]; out[a] = s; } } deMs = Number(process.hrtime.bigint() - t0) / 1e6 / REP;
    }
    w.write(`# MATVEC (sparsity + dense vs sparse cost) | TARGET=${TARGET} |A|=${A.length} n=${n}\n`);
    w.write(`  M: nnz=${nnz} (upper-tri), density=${(100 * dens).toFixed(2)}%  |  mem: sparse ${(sparseBytes / 1e6).toFixed(1)}MB  vs dense ${(denseBytes / 1e6).toFixed(0)}MB (${(denseBytes / sparseBytes).toFixed(0)}×)\n`);
    w.write(`  matvec: sparse ${spMs.toFixed(3)}ms  vs dense ${built ? deMs.toFixed(3) + "ms (" + (deMs / spMs).toFixed(0) + "× slower)" : "SKIPPED (>1.5GB)"}\n`);
    w.write(`  ⇒ dynamics identical either way ⇒ same ~500 iters; dense only changes cost/mem. Iteration count is landscape-bound, not sparsity-bound.\n`);
    return;
  }

  // ── EXPCH: channel treatment — exp-solver temperature (eta) sweep + exp-unary (odds-ratio) domain. Does
  //    sharper exponentiation escape the flat 500-iter landscape, and does the u-vs-M split help quality? ──
  if (process.env.EXPCH === "1") {
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const RHO = Number(process.env.RHO || 80);
    const wU = new Float64Array(A.length).fill(1 / A.length);
    const uw0 = buildAffinity(A, wU, neg, { commonSet: neg.DtauSel });
    w.write(`# EXPCH (temperature + exp-unary) | TARGET=${TARGET} |A|=${A.length} n=${uw0.u.length}\n`);
    for (const [nm, o] of [["eta=1 (deployed)", { eta: 1 }], ["eta=2", { eta: 2 }], ["eta=4", { eta: 4 }], ["eta=8", { eta: 8 }], ["expU β=1", { eta: 1, expU: 1 }]]) {
      const t0 = process.hrtime.bigint(); const r = replicate(uw0.u, uw0.edges, { solver: "exp", rho: RHO, ...o }); const ms = Number(process.hrtime.bigint() - t0) / 1e6;
      w.write(`  [round0] ${nm.padEnd(16)}: ${String(r.iters).padStart(3)} iters ${r.iters >= 500 ? "(CAP)" : "(conv)"}  ${ms.toFixed(1).padStart(6)}ms  |supp|=${r.Q.length}  f=${r.f.toFixed(3)}\n`);
    }
    const orRec = (G, pool) => (pool.length ? pool.filter((y) => G.some((g) => andCount(y, g.Qbits) >= g.m)).length / pool.length : 0);
    for (const [nm, o] of [["eta=1 (deployed)", {}], ["eta=4", { eta: 4 }], ["eta=8", { eta: 8 }], ["expU β=1", { expU: 1 }], ["expU β=2", { expU: 2 }]]) {
      const prof = { affinity: 0, replicate: 0, support: 0, mfit: 0, reweight: 0, edges: 0, iters: 0, supp: 0, n: 0 };
      const t0 = process.hrtime.bigint(); const r = boost(A, neg, { rho: RHO, solver: "exp", recallTau: true, tauFloor: 0.6, prof, ...o }); const ms = Number(process.hrtime.bigint() - t0) / 1e6;
      w.write(`  [boost]  ${nm.padEnd(16)}: ${ms.toFixed(0).padStart(5)}ms  avg ${(prof.iters / (r.rounds || 1)).toFixed(0).padStart(3)} it/rnd  K=${r.G.length}  rec=${(100 * orRec(r.G, A)).toFixed(1)}%  FP=${(100 * orRec(r.G, neg.Neg)).toFixed(2)}%\n`);
    }
    return;
  }

  // ── SHIFT: Pelillo/Bomze matrix offset → make the STANDARD/HYBRID replicator well-defined & convergent.
  //    Constant γ·eeᵀ (maximizer-preserving, enables std) vs diagonal κ·I (Bomze, changes solution). ──
  if (process.env.SHIFT === "1") {
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const RHO = Number(process.env.RHO || 80);
    const wU = new Float64Array(A.length).fill(1 / A.length);
    const uw0 = buildAffinity(A, wU, neg, { commonSet: neg.DtauSel });
    let mU = Infinity, mM = 0; for (let p = 0; p < uw0.u.length; p++) if (uw0.u[p] < mU) mU = uw0.u[p]; for (let e = 0; e < uw0.edges.m.length; e++) if (uw0.edges.m[e] < mM) mM = uw0.edges.m[e];
    const gAuto = RHO - mU - mM + 1e-9;
    w.write(`# SHIFT (Pelillo/Bomze offset) | TARGET=${TARGET} |A|=${A.length} n=${uw0.u.length} | auto γ=${gAuto.toFixed(0)} (ρ=${RHO}, min u=${mU.toFixed(1)}, min M=${mM.toFixed(1)})\n`);
    // round-0 convergence + monotonicity of f (fundamental theorem ⇒ f non-decreasing for the fixed-shift std)
    const runs = [
      ["exp (deployed)", { solver: "exp", rho: RHO }],
      ["std adaptive", { solver: "std", rho: RHO }],
      ["hybrid adaptive", { solver: "hybrid", rho: RHO }],
      [`std γ=auto`, { solver: "std", rho: RHO, shiftConst: "auto" }],
      [`hybrid γ=auto`, { solver: "hybrid", rho: RHO, shiftConst: "auto" }],
    ];
    for (const [nm, o] of runs) {
      const t0 = process.hrtime.bigint(); const r = replicate(uw0.u, uw0.edges, o); const ms = Number(process.hrtime.bigint() - t0) / 1e6;
      w.write(`  [round0] ${nm.padEnd(16)}: ${String(r.iters).padStart(3)} iters ${r.iters >= 500 ? "(CAP)" : "(conv)"}  ${ms.toFixed(1).padStart(6)}ms  |supp|=${r.Q.length}  f=${r.f.toFixed(3)}\n`);
    }
    // full boost: quality + time for the shift variants + diagonal-κ sweep
    const orRec = (G, pool) => (pool.length ? pool.filter((y) => G.some((g) => andCount(y, g.Qbits) >= g.m)).length / pool.length : 0);
    const bruns = [
      ["exp (deployed)", { solver: "exp" }],
      ["std γ=auto", { solver: "std", shiftConst: "auto" }],
      ["hybrid γ=auto", { solver: "hybrid", shiftConst: "auto" }],
      ["exp κ=+40 diag", { solver: "exp", shiftDiag: 40 }],
      ["exp κ=−40 diag", { solver: "exp", shiftDiag: -40 }],
    ];
    for (const [nm, o] of bruns) {
      const prof = { affinity: 0, replicate: 0, support: 0, mfit: 0, reweight: 0, edges: 0, iters: 0, supp: 0, n: 0 };
      const t0 = process.hrtime.bigint(); const r = boost(A, neg, { rho: RHO, recallTau: true, tauFloor: 0.6, prof, ...o }); const ms = Number(process.hrtime.bigint() - t0) / 1e6;
      w.write(`  [boost]  ${nm.padEnd(16)}: ${ms.toFixed(0).padStart(5)}ms  avg ${(prof.iters / (r.rounds || 1)).toFixed(0).padStart(3)} it/rnd  K=${r.G.length}  rec=${(100 * orRec(r.G, A)).toFixed(1)}%  FP=${(100 * orRec(r.G, neg.Neg)).toFixed(2)}%\n`);
    }
    return;
  }

  // ── MAXITER: does the replicator need its full 500-iter budget? Sweep the cap, compare parts vs baseline. ──
  if (process.env.MAXITER === "1") {
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const cfg = { rho: 80, solver: "exp", recallTau: true, tauFloor: 0.6 };
    const orRec = (G, pool) => (pool.length ? pool.filter((y) => G.some((g) => andCount(y, g.Qbits) >= g.m)).length / pool.length : 0);
    const t0 = process.hrtime.bigint(); const base = boost(A, neg, cfg); const baseMs = Number(process.hrtime.bigint() - t0) / 1e6;
    const baseRec = orRec(base.G, A), baseFP = orRec(base.G, neg.Neg);
    const keyOf = (G) => G.map((g) => `${g.m}:${g.Qbits.join(",")}`).join("|");
    const baseKey = keyOf(base.G);
    w.write(`# MAXITER (replicator iteration-cap sweep) | TARGET=${TARGET} |A|=${A.length}\n`);
    w.write(`  baseline maxIter=500: ${baseMs.toFixed(0)}ms  K=${base.G.length}  rec=${(100 * baseRec).toFixed(1)}%  FP=${(100 * baseFP).toFixed(2)}%\n`);
    for (const mi of [50, 100, 200]) {
      const t1 = process.hrtime.bigint(); const r = boost(A, neg, { ...cfg, maxIter: mi }); const ms = Number(process.hrtime.bigint() - t1) / 1e6;
      const rec = orRec(r.G, A), fp = orRec(r.G, neg.Neg), exact = keyOf(r.G) === baseKey;
      w.write(`  maxIter=${mi}: ${ms.toFixed(0)}ms (${(baseMs / ms).toFixed(2)}×)  K=${r.G.length}  rec=${(100 * rec).toFixed(1)}%  FP=${(100 * fp).toFixed(2)}%  bit-exact=${exact}\n`);
    }
    for (const sp of [3, 5]) {                                           // support-stability early-stop (adaptive)
      const t1 = process.hrtime.bigint(); const r = boost(A, neg, { ...cfg, suppPatience: sp }); const ms = Number(process.hrtime.bigint() - t1) / 1e6;
      const rec = orRec(r.G, A), fp = orRec(r.G, neg.Neg), exact = keyOf(r.G) === baseKey;
      w.write(`  suppPatience=${sp}: ${ms.toFixed(0)}ms (${(baseMs / ms).toFixed(2)}×)  K=${r.G.length}  rec=${(100 * rec).toFixed(1)}%  FP=${(100 * fp).toFixed(2)}%  bit-exact=${exact}\n`);
    }
    return;
  }

  // ── OPT: verify the shared negPool inverted index gives IDENTICAL parts (scan vs index), and time both. ──
  if (process.env.OPT === "1") {
    // build the inverted index once (shared across all words): bit → sorted list of negPool ctx indices
    const ti = process.hrtime.bigint();
    const negLen = new Int32Array(negPool.length); for (let i = 0; i < negPool.length; i++) negLen[i] = negPool[i].length;
    const negIndex = new Map();
    for (let i = 0; i < negPool.length; i++) for (const b of negPool[i]) { let a = negIndex.get(b); if (!a) { a = []; negIndex.set(b, a); } a.push(i); }
    const idxMs = Number(process.hrtime.bigint() - ti) / 1e6;
    let idxEntries = 0; for (const [, a] of negIndex) idxEntries += a.length;      // total posting entries = total neg bits
    const heapAfterIdx = process.memoryUsage().heapUsed;
    const eqQ = (a, b) => a.length === b.length && a.every((x, k) => x === b[k]);
    const partsKey = (G) => G.map((g) => g.Qbits.join(",") + `|${g.m}|${g.alpha.toFixed(6)}`).join(";");
    // scan
    const t1 = process.hrtime.bigint(); const fitScan = fitClass(A, negPool, Mglob, { rho: 80, delta: DBAND }); const scanMs = Number(process.hrtime.bigint() - t1) / 1e6;
    // index
    const t2 = process.hrtime.bigint(); const fitIdx = fitClass(A, negPool, Mglob, { rho: 80, delta: DBAND, negIndex, negLen }); const idxFitMs = Number(process.hrtime.bigint() - t2) / 1e6;
    const negSame = fitScan.neg.Neg.length === fitIdx.neg.Neg.length && fitScan.neg.Neg.every((y, k) => eqQ(y, fitIdx.neg.Neg[k]));
    const partsSame = partsKey(fitScan.G) === partsKey(fitIdx.G);
    // disk: multi-DS stores Σ|Q| (overlapping parts, redundant) vs disjoint union (adaptive-core) vs union+bitmask
    let sumQ2 = 0; for (const g of fitScan.G) sumQ2 += g.Qbits.length;
    const uniq2 = new Set(fitScan.G.flatMap((g) => g.Qbits)).size;
    w.write(`# OPT (neg inverted index) | TARGET=${TARGET} |A|=${A.length} negPool=${negPool.length}\n`);
    w.write(`  index RAM: ${idxEntries} posting entries (~${(idxEntries * 8 / 1e6).toFixed(0)}MB packed) — a RAM COST, not a saving\n`);
    w.write(`  disk: multi-DS Σ|Q|=${sumQ2}b (~${(2 * sumQ2 / 1024).toFixed(1)}KB) vs disjoint union=${uniq2}b (~${(2 * uniq2 / 1024).toFixed(1)}KB) → ${(sumQ2 / (uniq2 || 1)).toFixed(1)}× smaller\n`);
    w.write(`  index build (once, shared): ${idxMs.toFixed(0)}ms\n`);
    w.write(`  buildNegSet path: scan-in-fit ${scanMs.toFixed(0)}ms  vs  index-in-fit ${idxFitMs.toFixed(0)}ms\n`);
    w.write(`  IDENTICAL: Neg=${negSame ? "YES" : "NO"}  parts=${partsSame ? "YES" : "NO"}  (K=${fitScan.G.length})\n`);
    return;
  }

  // ── PERF: featurization time (shared, one-time) vs per-word extraction time; on-DISK part size. ──
  if (process.env.PERF === "1") {
    const featMs = Number(process.hrtime.bigint() - _t0) / 1e6;
    const t1 = process.hrtime.bigint();
    const fit = fitClass(A, negPool, Mglob, { rho: 80, delta: DBAND });
    const extractMs = Number(process.hrtime.bigint() - t1) / 1e6;
    // on-disk size: per part = Σ delta-varint(Qbits, ~2B each) + m(1B) + α(2B quant); header K(2B)+θ(4B)
    let sumQ = 0; for (const g of fit.G) sumQ += g.Qbits.length;
    const diskB = fit.G.reduce((s, g) => s + 2 * g.Qbits.length + 3, 0) + 6;
    const uniq = new Set(fit.G.flatMap((g) => g.Qbits)).size;
    const mem = process.memoryUsage();
    w.write(`# PERF | TARGET=${TARGET} |A|=${A.length} negPool=${negPool.length}\n`);
    w.write(`  featurize (once, shared): ${featMs.toFixed(0)}ms   |   extract (per-word): ${extractMs.toFixed(0)}ms   parts=${fit.G.length}\n`);
    w.write(`  parts on disk: Σ|Q|=${sumQ} bit-ids (union ${uniq})  → ~${(diskB / 1024).toFixed(1)} KB (delta-varint 2B/bit + m,α)\n`);
    w.write(`  runtime RSS=${(mem.rss / 1e6).toFixed(0)}MB heap=${(mem.heapUsed / 1e6).toFixed(0)}MB\n`);
    return;
  }

  // ── CV mode: k-fold ban vs keep vs keep+earlystop, mean±sd. The honest test of the unified rule. ──
  if (process.env.CV === "1") {
    const K = Number(process.env.K || 5), RHO = Number(process.env.RHO || 80);
    const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
    const cfgs = [
      ["ban", { maskNodes: Mglob }, {}, false],
      ["keep", { maskNodes: new Set() }, {}, false],
      ["banU-keepE", { maskNodes: new Set() }, { commonCoefU: 0, commonCoefM: 1 }, false],
      ["keepU-banE", { maskNodes: new Set() }, { commonCoefU: 1, commonCoefM: 0 }, false],
      ["keep+es", { maskNodes: new Set() }, {}, true],
      ["keepU-banE+es", { maskNodes: new Set() }, { commonCoefM: 0 }, true],
    ];
    const agg = new Map(cfgs.map(([n]) => [n, { rec: [], fp: [], r: [] }]));
    for (let f = 0; f < K; f++) {
      const test = A.filter((_, i) => i % K === f), rest = A.filter((_, i) => i % K !== f);
      const nv = Math.floor(rest.length * 0.75), tr = rest.slice(0, nv), val = rest.slice(nv);
      for (const [name, negOpts, bOpts, early] of cfgs) {
        const nt = buildNegSet(tr, negPool, Mglob, { delta: DBAND, ...negOpts });
        const G = boost(tr, nt, { rho: RHO, solver: "exp", ...bOpts }).G;
        let rStar = G.length;
        if (early && G.length) { let best = -Infinity; for (let r = 1; r <= G.length; r++) { const h = makeHead(G.slice(0, r)); const vr = rate((y) => h.fires(y), val), vfp = rate((y) => h.fires(y), nt.Neg); if (vr - vfp > best) { best = vr - vfp; rStar = r; } } }
        const head = makeHead(G.slice(0, rStar));
        agg.get(name).rec.push(rate((y) => head.fires(y), test)); agg.get(name).fp.push(rate((y) => head.fires(y), nt.Neg)); agg.get(name).r.push(rStar);
      }
    }
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
    const sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length); };
    w.write(`# ${K}-FOLD CV | TARGET=${TARGET} |A|=${A.length} (ρ=${RHO}, δ=${DBAND})\n   config    test-rec         test-confFP       r*\n`);
    for (const [name] of cfgs) { const a = agg.get(name); w.write(`   ${name.padEnd(11)} ${(100 * mean(a.rec)).toFixed(1)}%±${(100 * sd(a.rec)).toFixed(1)}     ${(100 * mean(a.fp)).toFixed(1)}%±${(100 * sd(a.fp)).toFixed(1)}     ${mean(a.r).toFixed(0)}\n`); }
    return;
  }

  // ── FP DIAG: fit deployed keep+es, break confFP down per part (find the m=1 mop-up culprits). ──
  if (process.env.FPDIAG === "1") {
    const RHO = Number(process.env.RHO || 80);
    const fit = fitClass(A, negPool, Mglob, { rho: RHO, delta: DBAND });
    const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
    const Neg = fit.neg.Neg, head = fit.head;
    const mHist = new Map();
    w.write(`# FP DIAG | TARGET=${TARGET} |A|=${A.length} r*=${fit.rStar}/${fit.roundsTotal} | HEAD fp=${(100 * rate((y) => head.fires(y), Neg)).toFixed(1)}% θ=${head.theta.toFixed(2)}\n`);
    w.write(`   part  m  |Q|  alpha  partRec  partFP  cumHeadFP(drop-this)\n`);
    fit.G.forEach((g, k) => {
      mHist.set(g.m, (mHist.get(g.m) || 0) + 1);
      const pr = rate((y) => andCount(y, g.Qbits) >= g.m, fit.neg ? A : A);
      const pf = rate((y) => andCount(y, g.Qbits) >= g.m, Neg);
      const without = makeHead(fit.G.filter((_, j) => j !== k));
      const dfp = rate((y) => without.fires(y), Neg);
      w.write(`   ${String(k).padStart(4)}  ${String(g.m).padStart(1)}  ${String(g.Qbits.length).padStart(3)}  ${g.alpha.toFixed(2).padStart(5)}  ${(100 * pr).toFixed(0).padStart(6)}%  ${(100 * pf).toFixed(1).padStart(5)}%  ${(100 * dfp).toFixed(1).padStart(6)}%\n`);
    });
    w.write(`# m-histogram: ${[...mHist.entries()].sort((a, b) => a[0] - b[0]).map(([m, c]) => `m=${m}:${c}`).join("  ")}\n`);
    return;
  }

  // ── FP-CV: does val θ-tuning (with FP weight λ) lower the confFP floor? deployed fitClass, k-fold. ──
  if (process.env.FPCV === "1") {
    const K = Number(process.env.K || 5), RHO = Number(process.env.RHO || 80);
    const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
    const cfgs = [
      ["es(θ=½Σα)", { tuneTheta: false, fpWeight: 1 }],
      ["+θtune λ=1", { tuneTheta: true, fpWeight: 1 }],
      ["+θtune λ=2", { tuneTheta: true, fpWeight: 2 }],
      ["+θtune λ=3", { tuneTheta: true, fpWeight: 3 }],
    ];
    const agg = new Map(cfgs.map(([n]) => [n, { rec: [], fp: [] }]));
    for (let f = 0; f < K; f++) {
      const test = A.filter((_, i) => i % K === f), rest = A.filter((_, i) => i % K !== f);
      for (const [name, o] of cfgs) {
        const fit = fitClass(rest, negPool, Mglob, { rho: RHO, delta: DBAND, ...o });
        agg.get(name).rec.push(rate((y) => fit.head.fires(y), test));
        agg.get(name).fp.push(rate((y) => fit.head.fires(y), fit.neg.Neg));
      }
    }
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length, sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length); };
    w.write(`# FP-CV | TARGET=${TARGET} |A|=${A.length} (ρ=${RHO}, δ=${DBAND})\n   config       test-rec        test-confFP\n`);
    for (const [name] of cfgs) { const a = agg.get(name); w.write(`   ${name.padEnd(12)} ${(100 * mean(a.rec)).toFixed(1)}%±${(100 * sd(a.rec)).toFixed(1)}   ${(100 * mean(a.fp)).toFixed(1)}%±${(100 * sd(a.fp)).toFixed(1)}\n`); }
    return;
  }

  // ── PART: partition the round-1 dominant set into k disjoint sub-parts (weight round-robin) and show
  // the OR-ensemble reconstructs the whole part's recall — more parts, each weaker, disjoint, same span. ──
  if (process.env.PART === "1") {
    const RHO = Number(process.env.RHO || 80), kP = Number(process.env.KP || 4);
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const ww = new Float64Array(A.length).fill(1 / A.length);
    const { u, edges } = buildAffinity(A, ww, neg, { commonSet: neg.DtauSel, commonCoef: 1 });
    const r = replicate(u, edges, { solver: "exp", rho: RHO });
    const Qw = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i]]).sort((a, b) => b[1] - a[1]);
    const whole = Qw.map((e) => e[0]).sort((a, b) => a - b);
    const rec = (bits, m) => A.filter((y) => andCount(y, bits) >= m).length / A.length;
    const fpr = (bits, m) => neg.Neg.filter((y) => andCount(y, bits) >= m).length / neg.Neg.length;
    const subs = Array.from({ length: kP }, () => []); Qw.forEach(([b], i) => subs[i % kP].push(b)); subs.forEach((s) => s.sort((a, b) => a - b));
    const orRec = A.filter((y) => subs.some((s) => andCount(y, s) >= 1)).length / A.length;
    const orFP = neg.Neg.filter((y) => subs.some((s) => andCount(y, s) >= 1)).length / neg.Neg.length;
    const subRec = subs.map((s) => rec(s, 1));
    w.write(`# PARTITION | TARGET=${TARGET} |A|=${A.length} ρ=${RHO} | whole |Q|=${whole.length} split into k=${kP} disjoint sub-parts\n`);
    w.write(`  whole part:   m=1 rec=${(100 * rec(whole, 1)).toFixed(1)}% fp=${(100 * fpr(whole, 1)).toFixed(1)}%   |   m=2 rec=${(100 * rec(whole, 2)).toFixed(1)}% fp=${(100 * fpr(whole, 2)).toFixed(1)}%\n`);
    w.write(`  ${kP} sub-parts:  each |Q|≈${Math.round(whole.length / kP)}  per-sub m=1 rec=[${(100 * Math.min(...subRec)).toFixed(0)}–${(100 * Math.max(...subRec)).toFixed(0)}%]  (weak, disjoint: Jaccard=0)\n`);
    w.write(`  OR-ensemble:  rec=${(100 * orRec).toFixed(1)}% fp=${(100 * orFP).toFixed(1)}%  → reconstructs whole m=1 (Δrec=${(100 * (orRec - rec(whole, 1))).toFixed(1)}pt)\n`);
    return;
  }

  // ── RT-CV: generic tauSupp (strong part) vs recall-based tauSupp (weak ~50% part). Reports part size,
  // count, α-flatness (cv), and whether OR-ed parts span the same bit set (union/|⋁A|, Jaccard). ──
  if (process.env.RTCV === "1") {
    const K = Number(process.env.K || 5), RHO = Number(process.env.RHO || 80);
    const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
    const cfgs = [["generic", { recallTau: false }], ["recallTau", { recallTau: true }]];
    w.write(`# RT-CV | TARGET=${TARGET} |A|=${A.length} |⋁A|~ (ρ=${RHO})\n   config     K   |Q|med  α-cv  Jacc  union/⋁A  train-rec held-rec confFP\n`);
    for (const [name, o] of cfgs) {
      const K_ = [], q = [], acv = [], jc = [], us = [], trc = [], rec = [], fp = [];
      for (let f = 0; f < K; f++) {
        const test = A.filter((_, i) => i % K === f), dev = A.filter((_, i) => i % K !== f);
        const fit = fitClass(dev, negPool, Mglob, { rho: RHO, delta: DBAND, ...o });
        const tr = dev.slice(0, Math.floor(dev.length * 0.75));
        K_.push(fit.G.length); trc.push(rate((y) => fit.head.fires(y), tr)); rec.push(rate((y) => fit.head.fires(y), test)); fp.push(rate((y) => fit.head.fires(y), fit.neg.Neg));
        const sz = fit.G.map((g) => g.Qbits.length).sort((a, b) => a - b); q.push(sz[sz.length >> 1] || 0);
        const al = fit.G.map((g) => g.alpha), am = mean(al) || 1e-9; acv.push(Math.sqrt(mean(al.map((x) => (x - am) ** 2))) / am);
        let js = 0, np = 0; const uni = new Set();
        for (let i = 0; i < fit.G.length; i++) { for (const b of fit.G[i].Qbits) uni.add(b); const Bi = new Set(fit.G[i].Qbits); for (let j = i + 1; j < fit.G.length; j++) { let it = 0; for (const b of fit.G[j].Qbits) if (Bi.has(b)) it++; js += it / (fit.G[i].Qbits.length + fit.G[j].Qbits.length - it || 1); np++; } }
        if (np) jc.push(js / np); us.push(uni.size / (fit.neg.pPlus.length || 1));
      }
      w.write(`   ${name.padEnd(10)} ${mean(K_).toFixed(0).padStart(2)}  ${mean(q).toFixed(0).padStart(5)}   ${(jc.length ? mean(acv) : 0).toFixed(2)}  ${(jc.length ? mean(jc) : 0).toFixed(2)}   ${(100 * mean(us)).toFixed(0)}%     ${(100 * mean(trc)).toFixed(1)}%   ${(100 * mean(rec)).toFixed(1)}%  ${(100 * mean(fp)).toFixed(1)}%\n`);
    }
    return;
  }

  // ── CORE-CV: prune each part to its high-weight core (x*·|Q| ≥ coreT). Does trimming the tail make
  // parts smaller + more distinct (lower Jaccard) and lower FP, or does recall collapse? ──
  if (process.env.CORECV === "1") {
    const K = Number(process.env.K || 5), RHO = Number(process.env.RHO || 80);
    const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
    const Ts = (process.env.CORETS || "0,1,2,4").split(",").map(Number);
    w.write(`# CORE-CV | TARGET=${TARGET} |A|=${A.length} (ρ=${RHO}, δ=${DBAND})\n   coreT  K   |Q|med  Jacc  union/Σ|Q|  train-rec  held-rec  confFP\n`);
    for (const T of Ts) {
      const rec = [], fp = [], trc = [], q = [], jc = [], uf = [], ks = [];
      for (let f = 0; f < K; f++) {
        const test = A.filter((_, i) => i % K === f), dev = A.filter((_, i) => i % K !== f);
        const fit = fitClass(dev, negPool, Mglob, { rho: RHO, delta: DBAND, coreT: T });
        const tr = dev.slice(0, Math.floor(dev.length * 0.75));
        rec.push(rate((y) => fit.head.fires(y), test)); fp.push(rate((y) => fit.head.fires(y), fit.neg.Neg)); trc.push(rate((y) => fit.head.fires(y), tr)); ks.push(fit.G.length);
        const sz = fit.G.map((g) => g.Qbits.length).sort((a, b) => a - b); q.push(sz[sz.length >> 1]);
        let js = 0, np = 0; const uni = new Set(); let sum = 0;
        for (let i = 0; i < fit.G.length; i++) { sum += fit.G[i].Qbits.length; for (const b of fit.G[i].Qbits) uni.add(b); const Bi = new Set(fit.G[i].Qbits); for (let j = i + 1; j < fit.G.length; j++) { let inter = 0; for (const b of fit.G[j].Qbits) if (Bi.has(b)) inter++; js += inter / (fit.G[i].Qbits.length + fit.G[j].Qbits.length - inter || 1); np++; } }
        if (np) jc.push(js / np); uf.push(uni.size / (sum || 1));
      }
      w.write(`   ${String(T).padEnd(5)}  ${mean(ks).toFixed(0).padStart(2)}  ${mean(q).toFixed(0).padStart(5)}  ${(jc.length ? mean(jc) : 0).toFixed(2)}   ${mean(uf).toFixed(2)}      ${(100 * mean(trc)).toFixed(1)}%   ${(100 * mean(rec)).toFixed(1)}%   ${(100 * mean(fp)).toFixed(1)}%\n`);
    }
    return;
  }

  // ── XSTAR: the replicator equilibrium shape per round — is x* a flat plateau + cliff, or a slope? And
  // where does the recall-floor m* land on the recall(m) curve — before/after its cliff? ──
  if (process.env.XSTAR === "1") {
    const RHO = Number(process.env.RHO || 80), ROUNDS = Number(process.env.ROUNDS || 3);
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const nA = A.length; let ww = new Float64Array(nA).fill(1 / nA);
    const covered = new Array(nA).fill(false);
    w.write(`# X* DIST | TARGET=${TARGET} |A|=${nA} ρ=${RHO} (deployed keep)\n`);
    for (let rnd = 0; rnd < ROUNDS; rnd++) {
      const { u, edges } = buildAffinity(A, ww, neg, { commonSet: neg.DtauSel, commonCoef: 1 });
      const r = replicate(u, edges, { solver: "exp", rho: RHO });
      if (!r.Q.length) { w.write(`  round ${rnd + 1}: empty\n`); break; }
      const all = [...r.x].sort((a, b) => b - a), k = r.Q.length, xs = all.slice(0, k);
      const pct = (arr, q) => arr[Math.min(arr.length - 1, Math.floor(q * arr.length))];
      // x* shape: within-support spread (×k units: 1.0 = uniform) + boundary drop factor
      const wS = `max=${(xs[0] * k).toFixed(2)} p25=${(pct(xs, 0.25) * k).toFixed(2)} med=${(pct(xs, 0.5) * k).toFixed(2)} p75=${(pct(xs, 0.75) * k).toFixed(2)} min=${(xs[k - 1] * k).toFixed(2)}`;
      const bDrop = all[k] > 0 ? (all[k - 1] / all[k]).toExponential(1) : "∞";
      // recall(m) curve
      const Qbits = r.Q.map((p) => neg.pPlus[p]).sort((a, b) => a - b);
      const cA = A.map((y) => andCount(y, Qbits));
      const recAt = (m) => cA.filter((c) => c >= m).length / nA;
      const fit = mfit(Qbits, A, ww, neg.Neg);
      let curve = ""; for (let m = 1; m <= Math.min(Qbits.length, 14); m++) curve += `${(100 * recAt(m)).toFixed(0)}${m === fit.m ? "*" : ""} `;
      w.write(`  round ${rnd + 1}: |Q|=${k} cliffGap=${r.cliffGap.toFixed(3)} | x* ${wS} (×1/|Q|; ≈1 flat) within max/min=${(xs[0] / xs[k - 1]).toFixed(1)} | boundary drop=${bDrop}\n`);
      w.write(`           recall(m=1..): ${curve.trim()}   m*=${fit.m} rec=${(100 * fit.recall).toFixed(0)}% prec=${(100 * fit.precision).toFixed(0)}%\n`);
      // reweight for next round (mirror boost)
      if (fit.valid) { let Wn = 0; const al = 0.5 * Math.log((1 - Math.min(Math.max(1 - fit.recall, 1e-10), 1 - 1e-10)) / Math.min(Math.max(1 - fit.recall, 1e-10), 1 - 1e-10)); for (let i = 0; i < nA; i++) { const h = cA[i] >= fit.m ? 1 : 0; if (h) covered[i] = true; ww[i] *= Math.exp(-al * h); Wn += ww[i]; } for (let i = 0; i < nA; i++) ww[i] /= Wn; }
    }
    return;
  }

  // ── TRIMCV: full discovered set vs early-stopped, crossed with θ-tune. Isolates trimming's effect. ──
  if (process.env.TRIMCV === "1") {
    const K = Number(process.env.K || 5), RHO = Number(process.env.RHO || 80);
    const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length, sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length); };
    const cfgs = [
      ["full,θ=½Σα", { earlyStop: false, tuneTheta: false }],
      ["full,θtune", { earlyStop: false, tuneTheta: true }],
      ["trim,θ=½Σα", { earlyStop: true, tuneTheta: false }],
      ["trim,θtune", { earlyStop: true, tuneTheta: true }],
    ];
    const agg = new Map(cfgs.map(([n]) => [n, { rec: [], fp: [], k: [] }]));
    for (let f = 0; f < K; f++) {
      const test = A.filter((_, i) => i % K === f), dev = A.filter((_, i) => i % K !== f);
      for (const [name, o] of cfgs) {
        const fit = fitClass(dev, negPool, Mglob, { rho: RHO, delta: DBAND, ...o });
        agg.get(name).rec.push(rate((y) => fit.head.fires(y), test)); agg.get(name).fp.push(rate((y) => fit.head.fires(y), fit.neg.Neg)); agg.get(name).k.push(fit.rStar);
      }
    }
    w.write(`# TRIM-CV | TARGET=${TARGET} |A|=${A.length} (ρ=${RHO}, δ=${DBAND})\n   config        K    test-rec        test-confFP\n`);
    for (const [name] of cfgs) { const a = agg.get(name); w.write(`   ${name.padEnd(12)} ${mean(a.k).toFixed(0).padStart(2)}   ${(100 * mean(a.rec)).toFixed(1)}%±${(100 * sd(a.rec)).toFixed(1)}   ${(100 * mean(a.fp)).toFixed(1)}%±${(100 * sd(a.fp)).toFixed(1)}\n`); }
    return;
  }

  // ── TFCV: raise the recall floor above 0.5 so weak-part α doesn't collapse to 0. Effect on α + FP. ──
  if (process.env.TFCV === "1") {
    const K = Number(process.env.K || 5), RHO = Number(process.env.RHO || 80);
    const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length, sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length); };
    const floors = (process.env.FLOORS || "0.50,0.55,0.60,0.65").split(",").map(Number);
    w.write(`# TAU-FLOOR SWEEP | TARGET=${TARGET} |A|=${A.length} (recallTau, ρ=${RHO})\n  floor  K   α-mean  α-cv   held-rec  confFP\n`);
    for (const tf of floors) {
      const rec = [], fp = [], al = [], ks = [];
      for (let f = 0; f < K; f++) { const test = A.filter((_, i) => i % K === f), dev = A.filter((_, i) => i % K !== f); const fit = fitClass(dev, negPool, Mglob, { rho: RHO, delta: DBAND, recallTau: true, tauFloor: tf }); rec.push(rate((y) => fit.head.fires(y), test)); fp.push(rate((y) => fit.head.fires(y), fit.neg.Neg)); ks.push(fit.G.length); for (const g of fit.G) al.push(g.alpha); }
      const am = mean(al), acv = sd(al) / (am || 1e-9);
      w.write(`  ${tf.toFixed(2)}   ${mean(ks).toFixed(0).padStart(2)}  ${am.toFixed(3)}   ${acv.toFixed(2)}   ${(100 * mean(rec)).toFixed(1)}%    ${(100 * mean(fp)).toFixed(1)}%\n`);
    }
    return;
  }

  // ── DIST: shape of the AdaBoost head-weight (α) and part-size (|Q|) distributions — flat or skewed? ──
  if (process.env.DIST === "1") {
    const K = Number(process.env.K || 5), RHO = Number(process.env.RHO || 80);
    const cfgs = [["generic ", { recallTau: false }], ["recallTau", { recallTau: true }]];
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length, sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length); };
    const pct = (a, q) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };
    const flat = (cv) => (cv < 0.3 ? "FLAT" : cv < 0.7 ? "moderate" : "skewed");
    w.write(`# DISTRIBUTIONS | TARGET=${TARGET} |A|=${A.length} (${K}-fold pooled, ρ=${RHO})\n`);
    for (const [name, o] of cfgs) {
      const al = [], qz = [];
      for (let f = 0; f < K; f++) { const dev = A.filter((_, i) => i % K !== f); const fit = fitClass(dev, negPool, Mglob, { rho: RHO, delta: DBAND, ...o }); for (const g of fit.G) { al.push(g.alpha); qz.push(g.Qbits.length); } }
      const acv = sd(al) / (mean(al) || 1e-9), qcv = sd(qz) / (mean(qz) || 1e-9);
      w.write(`  ${name} | α: n=${al.length} mean=${mean(al).toFixed(3)} cv=${acv.toFixed(2)} p10/50/90=${pct(al, .1).toFixed(2)}/${pct(al, .5).toFixed(2)}/${pct(al, .9).toFixed(2)} [${Math.min(...al).toFixed(2)}–${Math.max(...al).toFixed(2)}] → ${flat(acv)}\n`);
      w.write(`  ${" ".repeat(name.length)} | |Q|: mean=${mean(qz).toFixed(0)} cv=${qcv.toFixed(2)} p10/50/90=${pct(qz, .1)}/${pct(qz, .5)}/${pct(qz, .9)} [${Math.min(...qz)}–${Math.max(...qz)}] → ${flat(qcv)}\n`);
    }
    return;
  }

  // ── COMPARE50: reach 50% recall two ways — TRIM (min-cover, coverage-greedy) vs AUGMENT (weight-first grow
  // of the tight core). Which is more precise (lower FP)? ──
  if (process.env.COMPARE50 === "1") {
    const RHO = Number(process.env.RHO || 80);
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const nA = A.length;
    const rate = (b) => { if (!b.length) return 0; const s = [...b].sort((x, y) => x - y); return A.filter((y) => andCount(y, s) >= 1).length / nA; };
    const fpr = (b) => { if (!b.length) return 0; const s = [...b].sort((x, y) => x - y); return neg.Neg.filter((y) => andCount(y, s) >= 1).length / neg.Neg.length; };
    const ww = new Float64Array(nA).fill(1 / nA);
    const { u, edges } = buildAffinity(A, ww, neg, { commonSet: neg.DtauSel, commonCoef: 1 });
    const r = replicate(u, edges, { solver: "exp", rho: RHO });
    const byW = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i]]).sort((a, b) => b[1] - a[1]).map((e) => e[0]);
    const core = [...byW];
    // AUGMENT: weight-first prefix until recall ≥ 0.5
    let aug = []; for (const b of byW) { aug.push(b); if (rate(aug) >= 0.55) break; }
    // TRIM: greedy max-coverage to 0.5
    const ps = new Set(core), cv = new Map();
    for (let i = 0; i < nA; i++) for (const b of A[i]) if (ps.has(b)) { let a = cv.get(b); if (!a) { a = []; cv.set(b, a); } a.push(i); }
    const done = new Uint8Array(nA); let c = 0; const trim = [];
    while (c / nA < 0.55) { let bb = -1, bg = 0; for (const [b, idx] of cv) { let g = 0; for (const i of idx) if (!done[i]) g++; if (g > bg) { bg = g; bb = b; } } if (bb < 0 || !bg) break; trim.push(bb); for (const i of cv.get(bb)) if (!done[i]) { done[i] = 1; c++; } }
    w.write(`# REACH-50 | TARGET=${TARGET} |A|=${nA}  (both ≈55% recall, buffer)\n`);
    w.write(`  AUGMENT (weight-first): ${aug.length}b  rec=${(100 * rate(aug)).toFixed(0)}%  fp=${(100 * fpr(aug)).toFixed(0)}%\n`);
    w.write(`  TRIM (min-cover):       ${trim.length}b  rec=${(100 * rate(trim)).toFixed(0)}%  fp=${(100 * fpr(trim)).toFixed(0)}%\n`);
    return;
  }

  // ── ADAPTCORE (full): extract dominant → WEIGHT-FIRST tight core at equal recall (waste removed, low FP).
  // If ≥STRONG ship single; else augment weight-first to VIABLE(55%), peel (disjoint), reweight, iterate.
  // Multi-core: report union-of-cores vs the dominant core, and inter-core overlap. ──
  if (process.env.ADAPTCORE === "1") {
    const RHO = Number(process.env.RHO || 80), STRONG = Number(process.env.STRONG || 0.7), TR = Number(process.env.TR || 0.8), VIABLE = Number(process.env.VIABLE || 0.55);
    // dynamic FPMAX = predicted ensemble FP from concentration (recall-50 prefix size): FPMAX ≈ slope·r50, capped.
    const FPCAP = Number(process.env.FPCAP || 0.25), FPSLOPE = Number(process.env.FPSLOPE || 0.015);
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const nA = A.length;
    const rate = (b) => { if (!b.length) return 0; const s = [...b].sort((x, y) => x - y); return A.filter((y) => andCount(y, s) >= 1).length / nA; };
    const fpr = (b) => { if (!b.length) return 0; const s = [...b].sort((x, y) => x - y); return neg.Neg.filter((y) => andCount(y, s) >= 1).length / neg.Neg.length; };
    const jac = (a, b) => { const B = new Set(b); let it = 0; for (const x of a) if (B.has(x)) it++; const un = a.length + b.length - it; return un ? it / un : 0; };
    const memb = (y, set) => { for (const b of y) if (set.has(b)) return true; return false; };
    const peeled = new Set(), covered = new Uint8Array(nA); const cores = []; let stoppedStrong = false, domCore = null, cov = 0, dynFP = 0, r50 = 0;
    for (let round = 0; round < 12; round++) {
      let W = 0; const wt = new Float64Array(nA); for (let i = 0; i < nA; i++) { wt[i] = covered[i] ? 0.02 : 1; W += wt[i]; } for (let i = 0; i < nA; i++) wt[i] /= W;
      const { u, edges } = buildAffinity(A, wt, neg, { commonSet: neg.DtauSel, commonCoef: 1, exclude: peeled, excludeCoef: 0 });
      const r = replicate(u, edges, { solver: "exp", rho: RHO });
      const byW = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i]]).sort((a, b) => b[1] - a[1]).map((e) => e[0]);
      if (!byW.length) break;
      if (round === 0) domCore = byW.slice();                                     // full dominant core (for comparison)
      // bit → covered contexts (this round's dominant set)
      const dset = new Set(byW), cmap = new Map();
      for (let i = 0; i < nA; i++) for (const b of A[i]) if (dset.has(b)) { let a = cmap.get(b); if (!a) { a = []; cmap.set(b, a); } a.push(i); }
      if (round === 0) {                                                          // dynamic FPMAX from concentration
        const p = []; for (const b of byW) { p.push(b); if (rate(p) >= 0.5) break; } r50 = p.length;
        dynFP = Math.min(FPCAP, FPSLOPE * r50);                                    // predicted ensemble FP ≈ FPMAX
        const Rdom = rate(byW);
        if (Rdom >= STRONG) { const t = []; for (const b of byW) { t.push(b); if (rate(t) >= Rdom) break; } if (fpr(t) <= dynFP) { cores.push(t); stoppedStrong = true; break; } }
      }
      // weak: augment weight-first until WEIGHTED recall ≥ VIABLE (buffer), on current focus
      const wc = new Uint8Array(nA); let wcov = 0; const weak = [];
      for (const b of byW) { weak.push(b); for (const i of (cmap.get(b) || [])) if (!wc[i]) { wc[i] = 1; wcov += wt[i]; } if (wcov >= VIABLE) break; }
      cores.push(weak);
      for (const b of weak) peeled.add(b);
      const ws = new Set(weak); for (let i = 0; i < nA; i++) if (!covered[i] && memb(A[i], ws)) { covered[i] = 1; cov++; }
      if (cov / nA >= TR) break;
    }
    const all = cores.flatMap((c) => c);
    w.write(`# ADAPTIVE CORE | TARGET=${TARGET} |A|=${nA} (STRONG=${STRONG}, VIABLE=${VIABLE}, TR=${TR} | r50=${r50} → dynFPMAX=${(100 * dynFP).toFixed(0)}%)\n`);
    if (stoppedStrong) {
      w.write(`  STRONG single core: ${cores[0].length}b  rec=${(100 * rate(cores[0])).toFixed(0)}%  fp=${(100 * fpr(cores[0])).toFixed(0)}%\n`);
    } else {
      let js = 0, np = 0; for (let i = 0; i < cores.length; i++) for (let j = i + 1; j < cores.length; j++) { js += jac(cores[i], cores[j]); np++; }
      w.write(`  WEAK ENSEMBLE K=${cores.length}  per-core rec=[${cores.map((c) => (100 * rate(c)).toFixed(0)).join(",")}]%  per-core bits=[${cores.map((c) => c.length).join(",")}]\n`);
      w.write(`   union of cores: ${all.length}b  rec=${(100 * rate(all)).toFixed(0)}%  fp=${(100 * fpr(all)).toFixed(0)}%\n`);
      w.write(`   dominant core : ${domCore.length}b  rec=${(100 * rate(domCore)).toFixed(0)}%  fp=${(100 * fpr(domCore)).toFixed(0)}%\n`);
      w.write(`   → inter-core overlap (mean pairwise Jaccard) = ${(np ? js / np : 0).toFixed(2)}\n`);
    }
    return;
  }

  // ── DISJTIGHT: extract a tight core (x*·|Q|≥T); if its union recall clears TR, stop; else peel it (disjoint)
  // and extract another over the up-weighted uncovered positives. Accumulate disjoint tight cores until TR. ──
  if (process.env.DISJTIGHT === "1") {
    const RHO = Number(process.env.RHO || 80), T = Number(process.env.CORET || 2), TR = Number(process.env.TR || 0.5);
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const nA = A.length, peeled = new Set(), covered = new Uint8Array(nA);
    const fpr = (b) => { if (!b.length) return 0; const s = [...b].sort((x, y) => x - y); return neg.Neg.filter((y) => andCount(y, s) >= 1).length / neg.Neg.length; };
    const memb = (y, set) => { for (const b of y) if (set.has(b)) return true; return false; };
    const cores = []; let cov = 0;
    for (let round = 0; round < 20; round++) {
      let W = 0; const w = new Float64Array(nA); for (let i = 0; i < nA; i++) { w[i] = covered[i] ? 0.02 : 1; W += w[i]; } for (let i = 0; i < nA; i++) w[i] /= W;
      const { u, edges } = buildAffinity(A, w, neg, { commonSet: neg.DtauSel, commonCoef: 1, exclude: peeled, excludeCoef: 0 });
      const r = replicate(u, edges, { solver: "exp", rho: RHO }), k = r.Q.length;
      const tight = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i] * k]).filter(([, x]) => x >= T).map((e) => e[0]);
      if (!tight.length) break;
      const ts = new Set(tight);
      let selfCov = 0; for (let i = 0; i < nA; i++) if (memb(A[i], ts)) selfCov++;
      cores.push({ bits: tight, rec: selfCov / nA });
      for (const b of tight) peeled.add(b);
      for (let i = 0; i < nA; i++) if (!covered[i] && memb(A[i], ts)) { covered[i] = 1; cov++; }
      if (cov / nA >= TR) break;
    }
    const unionBits = cores.reduce((s, c) => s + c.bits.length, 0);
    const allBits = cores.flatMap((c) => c.bits);
    w.write(`# DISJOINT TIGHT CORES | TARGET=${TARGET} |A|=${nA} (T=${T}, target union rec=${TR})\n`);
    w.write(`  K=${cores.length}  union-rec=${(100 * cov / nA).toFixed(0)}%  union-bits=${unionBits}  union-fp=${(100 * fpr(allBits)).toFixed(0)}%  per-core rec=[${cores.map((c) => (100 * c.rec).toFixed(0)).join(",")}]%  per-core bits=[${cores.map((c) => c.bits.length).join(",")}]\n`);
    return;
  }

  // ── COREWASTE: greedy minimal cover (fewest bits reproducing a core's recall) → "wasted" bits = full−min.
  // Applied to the dominant core AND each peel core. Also: is min-dominant still > union-of-peel after tightening? ──
  if (process.env.COREWASTE === "1") {
    const RHO = Number(process.env.RHO || 80);
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const nA = A.length;
    const rate = (b) => { if (!b.length) return 0; const s = [...b].sort((x, y) => x - y); return A.filter((y) => andCount(y, s) >= 1).length / nA; };
    const fpr = (b) => { if (!b.length) return 0; const s = [...b].sort((x, y) => x - y); return neg.Neg.filter((y) => andCount(y, s) >= 1).length / neg.Neg.length; };
    // greedy min cover of `pool` to reach `target` recall (max-coverage each step)
    const minCover = (pool, target) => {
      const ps = new Set(pool), cov = new Map();
      for (let i = 0; i < nA; i++) for (const b of A[i]) if (ps.has(b)) { let a = cov.get(b); if (!a) { a = []; cov.set(b, a); } a.push(i); }
      const done = new Uint8Array(nA); let c = 0; const pick = [];
      while (c / nA < target) { let bb = -1, bg = 0; for (const [b, idx] of cov) { let g = 0; for (const i of idx) if (!done[i]) g++; if (g > bg) { bg = g; bb = b; } } if (bb < 0 || !bg) break; pick.push(bb); for (const i of cov.get(bb)) if (!done[i]) { done[i] = 1; c++; } }
      return pick;
    };
    // dominant core
    const ww = new Float64Array(nA).fill(1 / nA);
    const { u, edges } = buildAffinity(A, ww, neg, { commonSet: neg.DtauSel, commonCoef: 1 });
    const dom = replicate(u, edges, { solver: "exp", rho: RHO }).Q.map((p) => neg.pPlus[p]);
    const domR = rate(dom), domMin = minCover(dom, domR);
    w.write(`# CORE WASTE | TARGET=${TARGET} |A|=${nA}\n`);
    w.write(`  dominant core: ${dom.length}b rec=${(100 * domR).toFixed(0)}% → min-cover ${domMin.length}b (${(100 * (1 - domMin.length / dom.length)).toFixed(0)}% wasted)  min: rec=${(100 * rate(domMin)).toFixed(0)}% fp=${(100 * fpr(domMin)).toFixed(0)}%\n`);
    // peel cores
    const run = boost(A, neg, { rho: RHO, solver: "exp", coreT: 0, peel: true });
    let uFull = 0, uMin = 0;
    run.G.forEach((g, i) => { const cr = rate(g.Qbits), mc = minCover(g.Qbits, cr); uFull += g.Qbits.length; uMin += mc.length; w.write(`  peel core ${i + 1}: ${g.Qbits.length}b rec=${(100 * cr).toFixed(0)}% → min ${mc.length}b (${(100 * (1 - mc.length / g.Qbits.length)).toFixed(0)}% wasted)\n`); });
    w.write(`  → K=${run.G.length} peel cores: union full=${uFull}b, union min=${uMin}b  |  dominant min=${domMin.length}b\n`);
    return;
  }

  // ── TAIL: dominant core = tight (x*·|Q|≥T) ∪ tail (rest). What do the tail bits bring — recall & FP? ──
  if (process.env.TAIL === "1") {
    const RHO = Number(process.env.RHO || 80), T = Number(process.env.CORET || 2);
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const ww = new Float64Array(A.length).fill(1 / A.length);
    const { u, edges } = buildAffinity(A, ww, neg, { commonSet: neg.DtauSel, commonCoef: 1 });
    const r = replicate(u, edges, { solver: "exp", rho: RHO }), k = r.Q.length;
    const wp = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i] * k]);
    const tight = wp.filter(([, x]) => x >= T).map((e) => e[0]);
    const tail = wp.filter(([, x]) => x < T).map((e) => e[0]);
    const full = wp.map((e) => e[0]);
    const rate = (b) => { if (!b.length) return 0; const s = [...b].sort((x, y) => x - y); return A.filter((y) => andCount(y, s) >= 1).length / A.length; };
    const fpr = (b) => { if (!b.length) return 0; const s = [...b].sort((x, y) => x - y); return neg.Neg.filter((y) => andCount(y, s) >= 1).length / neg.Neg.length; };
    const rt = rate(tight), ft = fpr(tight), rf = rate(full), ff = fpr(full);
    w.write(`# TAIL | TARGET=${TARGET} |A|=${A.length} core=${full.length}b tight=${tight.length}b tail=${tail.length}b (T=${T})\n`);
    w.write(`  tight (${String(tight.length).padStart(3)}b): rec=${(100 * rt).toFixed(0).padStart(3)}%  fp=${(100 * ft).toFixed(0).padStart(3)}%\n`);
    w.write(`  tail  (${String(tail.length).padStart(3)}b): rec=${(100 * rate(tail)).toFixed(0).padStart(3)}%  fp=${(100 * fpr(tail)).toFixed(0).padStart(3)}%  (alone)\n`);
    w.write(`  full  (${String(full.length).padStart(3)}b): rec=${(100 * rf).toFixed(0).padStart(3)}%  fp=${(100 * ff).toFixed(0).padStart(3)}%\n`);
    w.write(`  → tail's MARGINAL: Δrec=+${(100 * (rf - rt)).toFixed(0)}pt  Δfp=+${(100 * (ff - ft)).toFixed(0)}pt  (per 100 tail bits: +${(100 * (rf - rt) / (tail.length / 100 || 1)).toFixed(1)}rec/+${(100 * (ff - ft) / (tail.length / 100 || 1)).toFixed(1)}fp)\n`);
    return;
  }

  // ── COREDECOMP: partition the DOMINANT core into K disjoint band-balanced pieces (union = core, no overlap).
  // How many pieces still each clear 50% recall? = the core's coverage redundancy. ──
  if (process.env.COREDECOMP === "1") {
    const RHO = Number(process.env.RHO || 80);
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const ww = new Float64Array(A.length).fill(1 / A.length);
    const { u, edges } = buildAffinity(A, ww, neg, { commonSet: neg.DtauSel, commonCoef: 1 });
    const r = replicate(u, edges, { solver: "exp", rho: RHO });
    // core sorted by weight desc (so round-robin gives each piece a weight-spread), keep band info
    const coreW = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i]]).sort((a, b) => b[1] - a[1]).map((e) => e[0]);
    const bOf = (b) => Math.floor(b / F);
    const rate = (bits) => { const s = [...bits].sort((a, b) => a - b); return A.filter((y) => andCount(y, s) >= 1).length / A.length; };
    const fpr = (bits) => { const s = [...bits].sort((a, b) => a - b); return neg.Neg.filter((y) => andCount(y, s) >= 1).length / neg.Neg.length; };
    w.write(`# CORE DECOMP | TARGET=${TARGET} |A|=${A.length} core=${coreW.length}b (union stays = core, m=1 rec=${(100 * rate(coreW)).toFixed(0)}% fp=${(100 * fpr(coreW)).toFixed(0)}%)\n  K  piece-bits  per-piece-rec[min-max]  all≥50%?\n`);
    let maxK = 1;
    for (const K of [1, 2, 3, 4, 5, 6]) {
      // band-balanced AND weight-spread: within each band, round-robin the weight-sorted bits
      const subs = Array.from({ length: K }, () => []);
      for (const bd of [0, 1, 2]) coreW.filter((b) => bOf(b) === bd).forEach((b, i) => subs[i % K].push(b));
      const G = subs.filter((s) => s.length); const recs = G.map((s) => rate(s));
      const minR = Math.min(...recs), maxR = Math.max(...recs), ok = minR >= 0.5;
      if (ok) maxK = K;
      w.write(`  ${K}  ${Math.round(coreW.length / K).toString().padStart(3)}       ${(100 * minR).toFixed(0)}-${(100 * maxR).toFixed(0)}%          ${ok ? "YES" : "no"}\n`);
    }
    w.write(`  → max disjoint pieces each ≥50%: ${maxK}\n`);
    return;
  }

  // ── SOFTPEEL: recall-target cores (≥tauFloor) + SOFT peel (scale claimed bits' u,M ×peelCoef, a nudge not
  // a ban). Sweep peelCoef: 1=no peel, →0=hard. Check cores don't overlap too much (meanJac) yet keep recall. ──
  if (process.env.SOFTPEEL === "1") {
    const RHO = Number(process.env.RHO || 80), TF = Number(process.env.TAUFLOOR || 0.5);
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const jac = (a, b) => { const B = new Set(b); let it = 0; for (const x of a) if (B.has(x)) it++; const un = a.length + b.length - it; return un ? it / un : 0; };
    const mn = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0);
    w.write(`# SOFT-PEEL | TARGET=${TARGET} |A|=${A.length} tauFloor=${TF} ρ=${RHO}\n  peelCoef  K  union%  meanJac  |Q|~  recall[min-max]  α[min-max]  stop\n`);
    for (const pc of (process.env.PEELCOEFS || "1,0.75,0.5,0.25,0").split(",").map(Number)) {
      const run = boost(A, neg, { rho: RHO, solver: "exp", recallTau: true, tauFloor: TF, peel: pc < 1, peelCoef: pc });
      const G = run.G.map((g) => g.Qbits); let js = 0, np = 0;
      for (let i = 0; i < G.length; i++) for (let j = i + 1; j < G.length; j++) { js += jac(G[i], G[j]); np++; }
      const recs = run.G.map((g) => g.recall), als = run.G.map((g) => g.alpha), qs = run.G.map((g) => g.Qbits.length);
      w.write(`  ${pc.toFixed(2).padEnd(7)}  ${String(G.length).padStart(2)} ${(100 * run.unionRecall).toFixed(0).padStart(4)}   ${(np ? js / np : 0).toFixed(2)}     ${mn(qs).toFixed(0).padStart(3)}   ${G.length ? (100 * Math.min(...recs)).toFixed(0) + "-" + (100 * Math.max(...recs)).toFixed(0) : "-"}%     ${G.length ? Math.min(...als).toFixed(2) + "-" + Math.max(...als).toFixed(2) : "-"}   ${run.stop}\n`);
    }
    return;
  }

  // ── NEFF: is the inverse participation ratio n_eff=1/Σx*² a direct FORMULA for the right tightness (no
  // sweep)? Report n_eff-core (top-n_eff bits) recall/fp vs the smallest prefix clearing 50% recall. ──
  if (process.env.NEFF === "1") {
    const RHO = Number(process.env.RHO || 80);
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const ww = new Float64Array(A.length).fill(1 / A.length);
    const { u, edges } = buildAffinity(A, ww, neg, { commonSet: neg.DtauSel, commonCoef: 1 });
    const r = replicate(u, edges, { solver: "exp", rho: RHO });
    const S = r.wPart.reduce((s, v) => s + v, 0), xn = r.wPart.map((v) => v / S);
    const neff = 1 / xn.reduce((s, v) => s + v * v, 0);
    const order = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i]]).sort((a, b) => b[1] - a[1]);
    const rate = (bits) => { const s = [...bits].sort((a, b) => a - b); return A.filter((y) => andCount(y, s) >= 1).length / A.length; };
    const fpr = (bits) => { const s = [...bits].sort((a, b) => a - b); return neg.Neg.filter((y) => andCount(y, s) >= 1).length / neg.Neg.length; };
    const kEff = Math.max(1, Math.round(neff)), coreEff = order.slice(0, kEff).map((e) => e[0]);
    // smallest top-weight prefix clearing 50% recall (m=1)
    let cov = new Uint8Array(A.length), c = 0, pref = [];
    for (const [b] of order) { pref.push(b); for (let i = 0; i < A.length; i++) if (!cov[i]) { const y = A[i]; for (let t = 0; t < y.length; t++) if (y[t] === b) { cov[i] = 1; c++; break; } } if (c / A.length >= 0.5) break; }
    w.write(`# NEFF | TARGET=${TARGET} |A|=${A.length} |Q|=${r.Q.length}\n`);
    w.write(`  n_eff=1/Σx*² = ${neff.toFixed(0)}  → top-${kEff} core: rec=${(100 * rate(coreEff)).toFixed(0)}% fp=${(100 * fpr(coreEff)).toFixed(0)}%\n`);
    w.write(`  smallest prefix ≥50% recall: ${pref.length} bits  rec=${(100 * rate(pref)).toFixed(0)}% fp=${(100 * fpr(pref)).toFixed(0)}%\n`);
    w.write(`  → n_eff ${kEff >= pref.length ? "≥" : "<"} recall-50 size (${kEff} vs ${pref.length})\n`);
    return;
  }

  // ── TIGHTBOOST: extract the TIGHT core (x*·|Q|≥T) each AdaBoost round. Do successive tight cores DIFFER
  // (multiple distinct cores) or repeat (same core re-emerges)? Report inter-core Jaccard, per-core recall
  // (>50%?), α, union, stop. peel=1 to also force disjointness. ──
  if (process.env.TIGHTBOOST === "1") {
    const RHO = Number(process.env.RHO || 80), pk = process.env.PEEL === "1";
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const jac = (a, b) => { const B = new Set(b); let it = 0; for (const x of a) if (B.has(x)) it++; const un = a.length + b.length - it; return un ? it / un : 0; };
    w.write(`# TIGHT-CORE ADABOOST | TARGET=${TARGET} |A|=${A.length} ρ=${RHO}${pk ? " +peel" : ""}\n  coreT  K  stop            union%  meanJac  |Q|~  recall[min-max]  α[min-max]\n`);
    for (const T of (process.env.CORETS || "0,1,2,4").split(",").map(Number)) {
      const run = boost(A, neg, { rho: RHO, solver: "exp", coreT: T, peel: pk });
      const G = run.G.map((g) => g.Qbits); let js = 0, np = 0;
      for (let i = 0; i < G.length; i++) for (let j = i + 1; j < G.length; j++) { js += jac(G[i], G[j]); np++; }
      const recs = run.G.map((g) => g.recall), als = run.G.map((g) => g.alpha), qs = run.G.map((g) => g.Qbits.length);
      const mn = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0);
      w.write(`  ${String(T).padEnd(5)}  ${String(G.length).padStart(2)} ${run.stop.padEnd(15)} ${(100 * run.unionRecall).toFixed(0).padStart(4)}   ${(np ? js / np : 0).toFixed(2)}     ${mn(qs).toFixed(0).padStart(3)}   ${G.length ? (100 * Math.min(...recs)).toFixed(0) + "-" + (100 * Math.max(...recs)).toFixed(0) : "-"}%       ${G.length ? Math.min(...als).toFixed(2) + "-" + Math.max(...als).toFixed(2) : "-"}\n`);
    }
    return;
  }

  // ── TIGHTCORE: sweep the x* weight threshold T (core tightness). Core = bits with x*·|Q|≥T, band-balanced
  // partitioned into ~4-bit sub-parts, θ-tuned head. Shows core-size/recall/FP vs tightness — the sweep. ──
  if (process.env.TIGHTCORE === "1") {
    const KF = Number(process.env.K || 5), RHO = Number(process.env.RHO || 80);
    const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
    const bOf = (b) => Math.floor(b / F);
    const Ts = (process.env.CORETS || "0,1,2,4,8").split(",").map(Number);
    w.write(`# TIGHT-CORE PARTITION SWEEP | TARGET=${TARGET} |A|=${A.length} (${KF}-fold, ρ=${RHO})\n  weightT  core-bits  K-sub  held-rec(%)  confFP(%)\n`);
    for (const T of Ts) {
      const cb = [], ks = [], rec = [], fp = [];
      for (let f = 0; f < KF; f++) {
        const test = A.filter((_, i) => i % KF === f), dev = A.filter((_, i) => i % KF !== f);
        const nTr = Math.floor(dev.length * 0.75), tr = dev.slice(0, nTr), val = dev.slice(nTr);
        const neg = buildNegSet(tr, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
        const ww = new Float64Array(tr.length).fill(1 / tr.length);
        const { u, edges } = buildAffinity(tr, ww, neg, { commonSet: neg.DtauSel, commonCoef: 1 });
        const r = replicate(u, edges, { solver: "exp", rho: RHO }), k = r.Q.length;
        const core = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i]]).filter(([, wt]) => wt * k >= T).map((e) => e[0]);
        if (core.length < 2) { cb.push(core.length); ks.push(0); rec.push(0); fp.push(0); continue; }
        const Kp = Math.max(2, Math.round(core.length / 4)), subs = Array.from({ length: Kp }, () => []);
        for (const bd of [0, 1, 2]) core.filter((b) => bOf(b) === bd).forEach((b, i) => subs[i % Kp].push(b));
        const Gp = subs.filter((s) => s.length).map((s) => ({ Qbits: s.sort((a, b) => a - b), m: 1, alpha: 1 }));
        let head = makeHead(Gp);
        const Sv = val.map((y) => head.S(y)), Sn = neg.Neg.map((y) => head.S(y));
        let bTh = head.theta, bObj = -Infinity;
        for (const c of [...new Set([0, ...Sv, ...Sn])]) { const vr = Sv.filter((s) => s > c).length / (Sv.length || 1), vf = Sn.filter((s) => s > c).length / (Sn.length || 1); if (vr - vf > bObj) { bObj = vr - vf; bTh = c; } }
        head = makeHead(Gp, bTh);
        cb.push(core.length); ks.push(Gp.length); rec.push(rate((y) => head.fires(y), test)); fp.push(rate((y) => head.fires(y), neg.Neg));
      }
      w.write(`  ${String(T).padEnd(6)}   ${mean(cb).toFixed(0).padStart(4)}      ${mean(ks).toFixed(0).padStart(3)}      ${(100 * mean(rec)).toFixed(1).padStart(5)}      ${(100 * mean(fp)).toFixed(1).padStart(4)}\n`);
    }
    return;
  }

  // ── BANDMASK: band-aware node mask — prune the noisy F0-far band (and optionally generic F2-adjacent)
  // from the affinity nodes, keeping discriminative F1-near. Test recall/FP/bits/diversity vs keep-all. ──
  if (process.env.BANDMASK === "1") {
    const KF = Number(process.env.K || 5), RHO = Number(process.env.RHO || 80);
    const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
    const bOf = (b) => Math.floor(b / F);
    const bandSet = (bands) => { const s = new Set(); for (const b of bitCount.keys()) if (bands.includes(bOf(b))) s.add(b); return s; };
    const freqSet = (dd) => { const s = new Set(); for (const [b, c] of bitCount) if (c / Ntot > dd) s.add(b); return s; };
    const uni = (...ss) => { const s = new Set(); for (const st of ss) for (const x of st) s.add(x); return s; };
    const cfgs = [
      ["keep-all", new Set()],
      ["prune-F0(far)", bandSet([0])],
      ["prune-common@0.01", freqSet(0.01)],
      ["F0 + common@0.01", uni(bandSet([0]), freqSet(0.01))],
      ["F0 + common@0.003", uni(bandSet([0]), freqSet(0.003))],
    ];
    w.write(`# BAND-AWARE NODE MASK | TARGET=${TARGET} |A|=${A.length} (${KF}-fold, ρ=${RHO})\n  config                 |P⁺nodes|  bits-used  intra-Jac  held-rec(%)  confFP(%)\n`);
    for (const [name, mnodes] of cfgs) {
      const nn = [], bits = [], jc = [], rec = [], fp = [];
      for (let f = 0; f < KF; f++) {
        const test = A.filter((_, i) => i % KF === f), dev = A.filter((_, i) => i % KF !== f);
        const fit = fitClass(dev, negPool, Mglob, { rho: RHO, delta: DBAND, maskNodes: mnodes });
        nn.push(fit.neg.pPlus.length); bits.push(new Set(fit.G.flatMap((g) => g.Qbits)).size);
        rec.push(rate((y) => fit.head.fires(y), test)); fp.push(rate((y) => fit.head.fires(y), fit.neg.Neg));
        let js = 0, np = 0; for (let i = 0; i < fit.G.length; i++) { const Bi = new Set(fit.G[i].Qbits); for (let j = i + 1; j < fit.G.length; j++) { let it = 0; for (const b of fit.G[j].Qbits) if (Bi.has(b)) it++; js += it / (fit.G[i].Qbits.length + fit.G[j].Qbits.length - it || 1); np++; } } if (np) jc.push(js / np);
      }
      w.write(`  ${name.padEnd(22)} ${mean(nn).toFixed(0).padStart(6)}    ${mean(bits).toFixed(0).padStart(5)}     ${(jc.length ? mean(jc) : 0).toFixed(2)}      ${(100 * mean(rec)).toFixed(1).padStart(5)}      ${(100 * mean(fp)).toFixed(1).padStart(4)}\n`);
    }
    return;
  }

  // ── COREPART: efficiency test — MULTI-DS AdaBoost (union spans ~5× the core) vs ONE core partitioned into
  // K disjoint band-balanced sub-parts (uniform α head, θ-tuned). Same held-out result with fewer bits ⇒ core wins. ──
  if (process.env.COREPART === "1") {
    const KF = Number(process.env.K || 5), RHO = Number(process.env.RHO || 80);
    const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
    const bOf = (b) => Math.floor(b / F);
    const M = { rec: [], fp: [], bits: [], k: [] }, C = { rec: [], fp: [], bits: [], k: [] };
    for (let f = 0; f < KF; f++) {
      const test = A.filter((_, i) => i % KF === f), dev = A.filter((_, i) => i % KF !== f);
      const fit = fitClass(dev, negPool, Mglob, { rho: RHO, delta: DBAND });        // multi-DS (deployed)
      M.rec.push(rate((y) => fit.head.fires(y), test)); M.fp.push(rate((y) => fit.head.fires(y), fit.neg.Neg));
      M.bits.push(new Set(fit.G.flatMap((g) => g.Qbits)).size); M.k.push(fit.G.length);
      const Kp = Math.max(1, fit.G.length);
      // core-partition: strong core on tr, split band-balanced into Kp, uniform α, θ-tune on val
      const nTr = Math.floor(dev.length * 0.75), tr = dev.slice(0, nTr), val = dev.slice(nTr);
      const neg = buildNegSet(tr, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
      const ww = new Float64Array(tr.length).fill(1 / tr.length);
      const { u, edges } = buildAffinity(tr, ww, neg, { commonSet: neg.DtauSel, commonCoef: 1 });
      const core = replicate(u, edges, { solver: "exp", rho: RHO }).Q.map((p) => neg.pPlus[p]);
      const subs = Array.from({ length: Kp }, () => []);
      for (const bd of [0, 1, 2]) core.filter((b) => bOf(b) === bd).forEach((b, i) => subs[i % Kp].push(b));
      const Gp = subs.filter((s) => s.length).map((s) => ({ Qbits: s.sort((a, b) => a - b), m: 1, alpha: 1 }));
      let head = makeHead(Gp);
      const Sv = val.map((y) => head.S(y)), Sn = neg.Neg.map((y) => head.S(y));
      let bTh = head.theta, bObj = -Infinity;
      for (const c of [...new Set([0, ...Sv, ...Sn])]) { const vr = Sv.filter((s) => s > c).length / (Sv.length || 1), vf = Sn.filter((s) => s > c).length / (Sn.length || 1); if (vr - vf > bObj) { bObj = vr - vf; bTh = c; } }
      head = makeHead(Gp, bTh);
      C.rec.push(rate((y) => head.fires(y), test)); C.fp.push(rate((y) => head.fires(y), neg.Neg)); C.bits.push(core.length); C.k.push(Gp.length);
    }
    w.write(`# CORE-PARTITION vs MULTI-DS | TARGET=${TARGET} |A|=${A.length} (${KF}-fold, ρ=${RHO})\n  config          K(count)  bits-used  held-rec(%)  confFP(%)\n`);
    w.write(`  multi-DS(ada)   ${mean(M.k).toFixed(0).padStart(3)}     ${mean(M.bits).toFixed(0).padStart(4)}       ${(100 * mean(M.rec)).toFixed(1).padStart(5)}       ${(100 * mean(M.fp)).toFixed(1).padStart(4)}\n`);
    w.write(`  core-partition  ${mean(C.k).toFixed(0).padStart(3)}     ${mean(C.bits).toFixed(0).padStart(4)}       ${(100 * mean(C.rec)).toFixed(1).padStart(5)}       ${(100 * mean(C.fp)).toFixed(1).padStart(4)}\n`);
    w.write(`  → bits ratio multi/core = ${(mean(M.bits) / mean(C.bits)).toFixed(1)}×\n`);
    return;
  }

  // ── BANDS: (a) per-band bit distribution [F2-adj|F1-near|F0-far], (b) does ∪(AdaBoost parts)=the core
  // (1 strong dominant set)?, (c) multiple dominant sets vs the core partitioned into K band-BALANCED
  // sub-parts — compare diversity, coverage, FP. ──
  if (process.env.BANDS === "1") {
    const RHO = Number(process.env.RHO || 80);
    const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND, maskNodes: new Set() });
    const ww = new Float64Array(A.length).fill(1 / A.length);
    const { u, edges } = buildAffinity(A, ww, neg, { commonSet: neg.DtauSel, commonCoef: 1 });
    const core = replicate(u, edges, { solver: "exp", rho: RHO }).Q.map((p) => neg.pPlus[p]).sort((a, b) => a - b);
    const fit = fitClass(A, negPool, Mglob, { rho: RHO, delta: DBAND });
    const parts = fit.G.map((g) => g.Qbits), K = parts.length;
    const unionAda = [...new Set(parts.flat())].sort((a, b) => a - b);
    const bOf = (b) => Math.floor(b / F);                                        // 0=far 1=near 2=adj
    const pb = (bits) => { const c = [0, 0, 0]; for (const b of bits) c[bOf(b)]++; const t = bits.length || 1; return `${(100 * c[2] / t).toFixed(0)}/${(100 * c[1] / t).toFixed(0)}/${(100 * c[0] / t).toFixed(0)}`; };
    const jac = (a, b) => { const B = new Set(b); let it = 0; for (const x of a) if (B.has(x)) it++; const un = a.length + b.length - it; return un ? it / un : 0; };
    const mJac = (P) => { let s = 0, n = 0; for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) { s += jac(P[i], P[j]); n++; } return n ? s / n : 0; };
    const rate = (bits, m, pool) => (pool.length ? pool.filter((y) => andCount(y, bits) >= m).length / pool.length : 0);
    const orRate = (P, pool) => (pool.length ? pool.filter((y) => P.some((s) => andCount(y, s) >= 1)).length / pool.length : 0);
    // band-BALANCED partition of the core into K sub-parts (round-robin within each band)
    const subs = Array.from({ length: Math.max(1, K) }, () => []);
    for (const bd of [0, 1, 2]) core.filter((b) => bOf(b) === bd).forEach((b, i) => subs[i % subs.length].push(b));
    const coreSet = new Set(core), uInC = unionAda.filter((b) => coreSet.has(b)).length;
    w.write(`# BANDS & CORE | TARGET=${TARGET} |A|=${A.length} F=${F} K=${K}\n`);
    w.write(`  per-band bit dist [F2-adj/F1-near/F0-far] %:\n`);
    w.write(`    ⋁A (OR p⁺): ${pb(neg.pPlus)}  (n=${neg.pPlus.length})\n`);
    w.write(`    core (1 DS): ${pb(core)}  (n=${core.length})\n`);
    w.write(`    ada ∪parts: ${pb(unionAda)}  (n=${unionAda.length})\n`);
    w.write(`  ∪(ada parts) vs core:  covers |∪∩core|/|core|=${(100 * uInC / core.length).toFixed(0)}%   in-core |∪∩core|/|∪|=${(100 * uInC / (unionAda.length || 1)).toFixed(0)}%\n`);
    w.write(`  MULTI-DS (AdaBoost, K=${K}):  intra-Jac=${mJac(parts).toFixed(2)}  OR-rec(A)=${(100 * orRate(parts, A)).toFixed(0)}%  OR-fp(Neg)=${(100 * orRate(parts, neg.Neg)).toFixed(0)}%\n`);
    w.write(`  BALANCED-PARTITION of core (K=${K}, band-balanced):  intra-Jac=${mJac(subs).toFixed(2)}  per-sub bands=${subs.map((s) => pb(s)).slice(0, 3).join(" ")}…  OR-rec(A)=${(100 * orRate(subs, A)).toFixed(0)}%  OR-fp(Neg)=${(100 * orRate(subs, neg.Neg)).toFixed(0)}%  (core m=1 rec=${(100 * rate(core, 1, A)).toFixed(0)}%)\n`);
    return;
  }

  // ── E2E: run the deployed fitClass (incl. AdaBoost) k-fold; report part structure + train/held behavior. ──
  if (process.env.E2E === "1") {
    const K = Number(process.env.K || 5), RHO = Number(process.env.RHO || 80);
    const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length, sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length); };
    const trRec = [], teRec = [], fpv = [], disc = [], Ks = [], qmin = [], qmed = [], qmax = [], mHist = new Map();
    const jacc = [], cont = [], uniqF = [], pP = [], unionQ = []; let repr = null;
    const jac = (a, b) => { const B = new Set(b); let inter = 0; for (const x of a) if (B.has(x)) inter++; return inter / (a.length + b.length - inter || 1); };
    const contain = (a, b) => { const B = new Set(b); let inter = 0; for (const x of a) if (B.has(x)) inter++; return inter / (Math.min(a.length, b.length) || 1); };
    for (let f = 0; f < K; f++) {
      const test = A.filter((_, i) => i % K === f), devFull = A.filter((_, i) => i % K !== f);
      const ACAP = Number(process.env.ACAP || 0);                                 // subsample training A to a budget
      const dev = (ACAP && devFull.length > ACAP) ? devFull.filter((_, i) => i % Math.ceil(devFull.length / ACAP) === 0) : devFull;
      const t0 = process.hrtime.bigint();
      const fit = fitClass(dev, negPool, Mglob, { rho: RHO, delta: DBAND, peel: process.env.PEEL === "1" });
      if (f === 0 && ACAP) w.write(`  [ACAP=${ACAP}: dev ${devFull.length}→${dev.length}, fit ${(Number(process.hrtime.bigint() - t0) / 1e6).toFixed(0)}ms]\n`);
      const tr = dev.slice(0, Math.floor(dev.length * 0.75));                     // the slice fitClass trained on
      trRec.push(rate((y) => fit.head.fires(y), tr)); teRec.push(rate((y) => fit.head.fires(y), test)); fpv.push(rate((y) => fit.head.fires(y), fit.neg.Neg));
      disc.push(fit.roundsTotal);
      const sz = fit.G.map((g) => g.Qbits.length).sort((a, b) => a - b);
      Ks.push(fit.G.length); qmin.push(sz[0]); qmed.push(sz[sz.length >> 1]); qmax.push(sz[sz.length - 1]);
      for (const g of fit.G) mHist.set(g.m, (mHist.get(g.m) || 0) + 1);
      // pairwise overlap among kept parts + union coverage
      const G = fit.G; let js = 0, cs = 0, np = 0; const uni = new Set(); let sum = 0;
      for (let i = 0; i < G.length; i++) { sum += G[i].Qbits.length; for (const b of G[i].Qbits) uni.add(b); for (let j = i + 1; j < G.length; j++) { js += jac(G[i].Qbits, G[j].Qbits); cs += contain(G[i].Qbits, G[j].Qbits); np++; } }
      if (np) { jacc.push(js / np); cont.push(cs / np); } uniqF.push(uni.size / (sum || 1));
      pP.push(fit.neg.pPlus.length); unionQ.push(uni.size);                       // |P⁺|=|⋁A| and union of all part bits
      if (!repr) repr = fit;
    }
    const aStat = repr.G.map((g) => g.alpha);
    w.write(`# E2E ${K}-fold | TARGET=${TARGET} |A|=${A.length} (ρ=${RHO}, δ=${DBAND}) — deployed fitClass\n`);
    w.write(`  vs ⋁A:    |P⁺|=|⋁A|=${mean(pP).toFixed(0)}  |Q|/|⋁A| med=${(mean(qmed) / mean(pP) * 100).toFixed(0)}%  union-of-parts/|⋁A|=${(mean(unionQ) / mean(pP) * 100).toFixed(0)}%\n`);
    w.write(`  PARTS:    discovered(boost)=${mean(disc).toFixed(1)}  kept(early-stop)=${mean(Ks).toFixed(1)}  |Q| min/med/max=${mean(qmin).toFixed(0)}/${mean(qmed).toFixed(0)}/${mean(qmax).toFixed(0)}  m-hist=${[...mHist.entries()].sort((a, b) => a[0] - b[0]).map(([m, c]) => `m${m}:${c}`).join(" ")}  α∈[${Math.min(...aStat).toFixed(2)},${Math.max(...aStat).toFixed(2)}]\n`);
    w.write(`  OVERLAP:  mean pairwise Jaccard=${(jacc.length ? mean(jacc) : 0).toFixed(2)}  containment(∩/min)=${(cont.length ? mean(cont) : 0).toFixed(2)}  union/Σ|Q|=${mean(uniqF).toFixed(2)} (1=disjoint, →0=redundant)\n`);
    w.write(`  CLASSIFY: train-rec ${(100 * mean(trRec)).toFixed(1)}%±${(100 * sd(trRec)).toFixed(1)}   held-rec ${(100 * mean(teRec)).toFixed(1)}%±${(100 * sd(teRec)).toFixed(1)}   confFP ${(100 * mean(fpv)).toFixed(1)}%±${(100 * sd(fpv)).toFixed(1)}   gap ${(100 * (mean(trRec) - mean(teRec))).toFixed(1)}pt\n`);
    return;
  }

  const neg = buildNegSet(A, negPool, Mglob, { delta: DBAND });
  w.write(`# phase1 — ${path.basename(corpusArg)} | TARGET=${TARGET} | |A|=${A.length} negPool=${negPool.length} negTot=${negTot}\n`);
  w.write(`# mask: |M_glob|=${Mglob.size} |D_τ|=${neg.Dtau.size} |P⁺|=${neg.pPlus.length}\n`);
  w.write(`# Neg (masked score, δ=${DBAND}): |Neg|=${neg.negN} skipped=${neg.skipped} | score_cut pre-mask=${neg.scoreCutPre.toFixed(3)} post-mask=${neg.scoreCutPost.toFixed(3)}\n`);

  // ── T1: M density (uniform w = round 1) ──
  const wUnif = new Float64Array(A.length).fill(1);
  const t0 = process.hrtime.bigint();
  const { u, edges, n } = buildAffinity(A, wUnif, neg, {});
  const secs = Number(process.hrtime.bigint() - t0) / 1e9;
  const nnz = edges.m.length, maxEdges = (n * (n - 1)) / 2, density = nnz / maxEdges;
  let pos = 0, negE = 0; for (const v of edges.m) (v > 0 ? pos++ : negE++);
  // degree histogram
  const deg = new Int32Array(n); for (let e = 0; e < nnz; e++) { deg[edges.i[e]]++; deg[edges.j[e]]++; }
  const degSorted = [...deg].sort((a, b) => a - b); const pct = (q) => degSorted[Math.floor(q * (n - 1))];
  let umin = Infinity, umax = -Infinity, uZero = 0; for (const v of u) { if (v < umin) umin = v; if (v > umax) umax = v; if (Math.abs(v) < 1e-9) uZero++; }

  w.write(`\n# T1 — M density (uniform w, round 1) | built in ${secs.toFixed(2)}s\n`);
  w.write(`#   |P⁺|=${n}  nnz(M)=${nnz}  density=${(100 * density).toFixed(2)}%  (of ${maxEdges} possible pairs)\n`);
  w.write(`#   edges: ${pos} attractive (M>0), ${negE} repulsive (M<0)\n`);
  w.write(`#   degree: median ${pct(0.5)}  p90 ${pct(0.9)}  max ${degSorted[n - 1]}  | isolated bits (deg 0): ${degSorted.filter((d) => d === 0).length}\n`);
  w.write(`#   unary u: min ${umin.toFixed(2)} max ${umax.toFixed(2)} | ~0 (equal-rate/mask): ${uZero}\n`);
  w.write(`#   → RULE: density ≲5–10% ⇒ sparse edge map; ≳30% ⇒ dense. Here: ${density < 0.1 ? "SPARSE edge map" : density > 0.3 ? "DENSE matrix" : "borderline (sparse ok)"}\n`);

  // ── T7: ρ → part size (hybrid solver, uniform w) ──
  const tag = (p) => (neg.pPlus[p] < F ? "C" : neg.pPlus[p] < 2 * F ? "L1" : "L");   // band of bit at position p
  w.write(`\n# T7 — ρ → part size (hybrid, round 1):\n`);
  for (const rho of [0, 1, 5, 20, 100]) { const r = replicate(u, edges, { solver: "hybrid", rho }); w.write(`   ρ=${String(rho).padStart(3)}  |Q|=${String(r.Q.length).padStart(4)}  iters=${String(r.iters).padStart(3)}  f=${r.f.toFixed(3)}\n`); }

  // ── T2: solver comparison at one ρ ──
  const rho0 = Number(process.env.RHO || 5);
  w.write(`\n# T2 — solver comparison (ρ=${rho0}):\n`);
  const ref = replicate(u, edges, { solver: "std", rho: rho0 }); const refSet = new Set(ref.Q);
  for (const solver of ["std", "exp", "hybrid"]) {
    const r = replicate(u, edges, { solver, rho: rho0 });
    let inter = 0; for (const q of r.Q) if (refSet.has(q)) inter++; const jac = inter / (r.Q.length + refSet.size - inter || 1);
    w.write(`   ${solver.padEnd(7)} |Q|=${String(r.Q.length).padStart(4)} iters=${String(r.iters).padStart(3)} matvecs=${String(r.matvecs).padStart(3)} f=${r.f.toFixed(4)} jac-vs-std=${jac.toFixed(3)}\n`);
  }

  // ── T6/T8: boost on a train/held split, ρ sweep, α-sum vs max-pool head ──
  const half = A.length >> 1, Atr = A.slice(0, half), Ahe = A.slice(half);
  const negTr = buildNegSet(Atr, negPool, Mglob, { delta: DBAND, twoPass: false });
  const broadNeg = negPool.filter((_, i) => i % 40 === 0);
  const pMinTr = Atr.length / (Atr.length + negTr.negN);
  const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
  w.write(`\n# T6/T8 — boost on train |Atr|=${Atr.length} |Neg|=${negTr.negN} p_min=${pMinTr.toFixed(3)} (ρ sweep, solver=exp):\n`);
  w.write(`   ρ    K  unionRec stop        | α-sum: tr-rec  hd-rec  confFP%  broadFP% | maxpool: hd-rec confFP%\n`);
  let bestRun = null;
  for (const rho of (process.env.RHOS || "10,20,40,80").split(",").map(Number)) {
    const run = boost(Atr, negTr, { rho, solver: "exp" });
    const head = makeHead(run.G);
    const trRec = rate((y) => head.fires(y), Atr), hdRec = rate((y) => head.fires(y), Ahe);
    const confFP = rate((y) => head.fires(y), negTr.Neg), broadFP = rate((y) => head.fires(y), broadNeg);
    const mpRec = rate((y) => head.maxpoolFires(y), Ahe), mpFP = rate((y) => head.maxpoolFires(y), negTr.Neg);
    w.write(`   ${String(rho).padStart(3)}  ${String(run.rounds).padStart(2)}   ${(100 * run.unionRecall).toFixed(1).padStart(5)}%  ${run.stop.padEnd(11)} |  ${(100 * trRec).toFixed(1).padStart(5)}%  ${(100 * hdRec).toFixed(1).padStart(5)}%  ${(100 * confFP).toFixed(2).padStart(5)}%  ${(100 * broadFP).toFixed(3).padStart(6)}% |  ${(100 * mpRec).toFixed(1).padStart(5)}%  ${(100 * mpFP).toFixed(2).padStart(5)}%\n`);
    if (!bestRun || run.unionRecall > bestRun.run.unionRecall) bestRun = { rho, run };
  }
  if (bestRun && bestRun.run.curve.length) {
    w.write(`\n# boost curve (ρ=${bestRun.rho}):  round  α      rec    prec   m   |Q|  cliff recAt1 unionRec\n`);
    for (const c of bestRun.run.curve) w.write(`   ${String(c.round).padStart(11)}  ${c.alpha.toFixed(2).padStart(5)}  ${(100 * c.recall).toFixed(1).padStart(5)}% ${(100 * c.precision).toFixed(1).padStart(5)}%  ${String(c.m).padStart(2)}  ${String(c.size).padStart(4)}  ${c.cliffGap.toFixed(2)}  ${(100 * c.recallAt1).toFixed(0).padStart(3)}%  ${(100 * c.unionRecall).toFixed(1).padStart(5)}%\n`);
    w.write(`# D3: stop=${bestRun.run.stop}  rounds=${bestRun.run.rounds}  union=${(100 * bestRun.run.unionRecall).toFixed(1)}%  no_valid_m=${bestRun.run.noValidM}\n`);
  }

  // ── MASK SWEEP: vary common-bit threshold DD (low=aggressive → high=mild → unmasked). Find the level
  // that de-saturates the boundary score (score_cut<1) AND keeps the pairwise signal (discovery works). ──
  const RHO = Number(process.env.RHO || 80);
  w.write(`\n# MASK SWEEP (ρ=${RHO}, δ=${DBAND}, masked boundary score kept):\n   DD       |D_τ|  |P⁺|  score_cut  K  unionRec  stop          hd-rec  confFP%\n`);
  for (const dd of (process.env.DDS || "0.0005,0.002,0.01,0.05,0.2,1").split(",").map(Number)) {
    const Mg = new Set(); for (const [b, c] of bitCount) if (c / Ntot > dd) Mg.add(b);
    const nt = buildNegSet(Atr, negPool, Mg, { delta: DBAND, twoPass: false });
    const run = boost(Atr, nt, { rho: RHO, solver: "exp" });
    const head = makeHead(run.G);
    const hd = rate((y) => head.fires(y), Ahe), fp = rate((y) => head.fires(y), nt.Neg);
    w.write(`   ${String(dd).padEnd(7)} ${String(nt.Dtau.size).padStart(6)} ${String(nt.pPlus.length).padStart(5)}  ${nt.scoreCutPost.toFixed(3)}     ${String(run.rounds).padStart(2)}  ${(100 * run.unionRecall).toFixed(1).padStart(5)}%   ${run.stop.padEnd(12)} ${(100 * hd).toFixed(1).padStart(5)}%  ${(100 * fp).toFixed(1).padStart(5)}%\n`);
  }

  // ── LOCAL-MASK SWEEP: fix a mild global mask (DDL), then re-run the SAME mask mechanism locally over
  // {pos ∪ Pass-1 neg}, sweeping θ_loc. Row θ=∞ (single-pass, no local mask) is the baseline. ──
  const DDL = Number(process.env.DDL || 0.05);
  const MgL = new Set(); for (const [b, c] of bitCount) if (c / Ntot > DDL) MgL.add(b);
  w.write(`\n# LOCAL-MASK SWEEP (global DD=${DDL}, ρ=${RHO}, δ=${DBAND}):\n   θ_loc   D_τ^loc  |P⁺|  score_cut  K  unionRec  stop          hd-rec  confFP%\n`);
  const thList = process.env.THLIST ? [Infinity, ...process.env.THLIST.split(",").map(Number)] : [Infinity, 0.9, 0.7, 0.5, 0.3];
  for (const th of thList) {
    const two = Number.isFinite(th);
    const nt = buildNegSet(Atr, negPool, MgL, { delta: DBAND, twoPass: two, localTheta: th });
    const run = boost(Atr, nt, { rho: RHO, solver: "exp" });
    const head = makeHead(run.G);
    const hd = rate((y) => head.fires(y), Ahe), fp = rate((y) => head.fires(y), nt.Neg);
    w.write(`   ${(two ? th.toFixed(2) : "∞(1p)").padEnd(7)} ${String(nt.DtauLoc).padStart(7)} ${String(nt.pPlus.length).padStart(5)}  ${nt.scoreCutPost.toFixed(3)}     ${String(run.rounds).padStart(2)}  ${(100 * run.unionRecall).toFixed(1).padStart(5)}%   ${run.stop.padEnd(12)} ${(100 * hd).toFixed(1).padStart(5)}%  ${(100 * fp).toFixed(1).padStart(5)}%\n`);
  }

  // ── NODE-MASK TEST: SELECT mask fixed mild (DDL). ban = nodes masked (common bits removed);
  // keep = nodes unmasked (common bits kept as affinity nodes). If kept common bits are (almost)
  // never selected into a part, self-elimination holds and the ban is redundant for discovery. ──
  w.write(`\n# NODE-MASK TEST (select DD=${DDL} mild, ρ=${RHO}, δ=${DBAND}):\n`);
  w.write(`   config          |P⁺nodes| commonKept  K  unionRec  stop         hd-rec confFP%  commonSel  %part-bits-common\n`);
  for (const [name, mnodes] of [["ban(masked)", MgL], ["keep(unmasked)", new Set()]]) {
    const nt = buildNegSet(Atr, negPool, MgL, { delta: DBAND, maskNodes: mnodes });
    const run = boost(Atr, nt, { rho: RHO, solver: "exp" });
    const head = makeHead(run.G);
    const hd = rate((y) => head.fires(y), Ahe), fp = rate((y) => head.fires(y), nt.Neg);
    const commonKept = nt.pPlus.filter((b) => MgL.has(b)).length;                 // common bits present as nodes
    const sel = new Set(); let totBits = 0, commonPartBits = 0;
    for (const g of run.G) for (const b of g.Qbits) { sel.add(b); totBits++; if (MgL.has(b)) commonPartBits++; }
    let commonSel = 0; for (const b of sel) if (MgL.has(b)) commonSel++;          // distinct common bits selected into ANY part
    const pct = totBits ? (100 * commonPartBits) / totBits : 0;
    w.write(`   ${name.padEnd(15)} ${String(nt.pPlus.length).padStart(6)} ${String(commonKept).padStart(9)}  ${String(run.rounds).padStart(2)}  ${(100 * run.unionRecall).toFixed(1).padStart(5)}%   ${run.stop.padEnd(12)} ${(100 * hd).toFixed(1).padStart(5)}% ${(100 * fp).toFixed(1).padStart(5)}%  ${String(commonSel).padStart(6)}/${commonKept}    ${pct.toFixed(2)}%\n`);
  }

  // ── PRECONDITION SWEEP: nodes KEPT (maskNodes=∅); soft-suppress common bits by scaling u,M ×c.
  // c=1 keep, c=0 ≡ soft-ban. Looking for a c that keeps state/time's hd-rec gain WITHOUT bank's drop. ──
  const ntK = buildNegSet(Atr, negPool, MgL, { delta: DBAND, maskNodes: new Set() });
  w.write(`\n# PRECONDITION SWEEP (nodes kept, common u,M ×c; ρ=${RHO}, δ=${DBAND}):\n   c       K  unionRec  stop         hd-rec confFP%  commonSel\n`);
  for (const c of [1, 0.75, 0.5, 0.25, 0.1, 0]) {
    const run = boost(Atr, ntK, { rho: RHO, solver: "exp", commonCoef: c });
    const head = makeHead(run.G);
    const hd = rate((y) => head.fires(y), Ahe), fp = rate((y) => head.fires(y), ntK.Neg);
    const sel = new Set(); for (const g of run.G) for (const b of g.Qbits) if (MgL.has(b)) sel.add(b);
    w.write(`   ${c.toFixed(2).padEnd(6)} ${String(run.rounds).padStart(2)}  ${(100 * run.unionRecall).toFixed(1).padStart(5)}%   ${run.stop.padEnd(12)} ${(100 * hd).toFixed(1).padStart(5)}% ${(100 * fp).toFixed(1).padStart(5)}%   ${String(sel.size).padStart(3)}/12\n`);
  }

  // ── SCAFFOLD TEST: does the value of keeping common bits come from bridging (M) or from the feature (Q)?
  // ban = nodes removed. keep = nodes in M AND selectable. scaffold = nodes in M but stripped from Q.
  // If bridging is the mechanism, scaffold ≈ keep for state AND ≈ ban for bank (can't overfit → not selectable). ──
  w.write(`\n# SCAFFOLD TEST (ρ=${RHO}, δ=${DBAND}):\n   config     K  unionRec  stop         hd-rec confFP%  commonSel\n`);
  const cfgs = [
    ["ban", { maskNodes: MgL }, {}],
    ["keep", { maskNodes: new Set() }, { commonCoef: 1 }],
    ["scaffold", { maskNodes: new Set() }, { commonCoef: 1, scaffoldOnly: true }],
  ];
  for (const [name, negOpts, boostOpts] of cfgs) {
    const nt = buildNegSet(Atr, negPool, MgL, { delta: DBAND, ...negOpts });
    const run = boost(Atr, nt, { rho: RHO, solver: "exp", ...boostOpts });
    const head = makeHead(run.G);
    const hd = rate((y) => head.fires(y), Ahe), fp = rate((y) => head.fires(y), nt.Neg);
    const sel = new Set(); for (const g of run.G) for (const b of g.Qbits) if (MgL.has(b)) sel.add(b);
    w.write(`   ${name.padEnd(10)} ${String(run.rounds).padStart(2)}  ${(100 * run.unionRecall).toFixed(1).padStart(5)}%   ${run.stop.padEnd(12)} ${(100 * hd).toFixed(1).padStart(5)}% ${(100 * fp).toFixed(1).padStart(5)}%   ${String(sel.size).padStart(3)}/12\n`);
  }

  // ── EARLY-STOP UNIFIED: ONE rule for all targets. Common bits kept available (maskNodes=∅); overfit
  // controlled not by a per-target ban flag but by VALIDATION prefix-selection — keep the part-prefix that
  // maximizes val (recall−confFP). Bank should self-limit to ≈ban; state self-run to ≈keep. 3-way split. ──
  const nT = Math.floor(A.length * 0.6), nV = Math.floor(A.length * 0.8);
  const Atr3 = A.slice(0, nT), Aval3 = A.slice(nT, nV), Ahe3 = A.slice(nV);
  w.write(`\n# EARLY-STOP UNIFIED (3-way |tr|=${Atr3.length} |val|=${Aval3.length} |test|=${Ahe3.length}, ρ=${RHO}, δ=${DBAND}):\n   config           K  r*  test-rec test-confFP\n`);
  for (const [name, negOpts, early] of [["ban", { maskNodes: MgL }, false], ["keep(full)", { maskNodes: new Set() }, false], ["keep+earlystop", { maskNodes: new Set() }, true]]) {
    const nt = buildNegSet(Atr3, negPool, MgL, { delta: DBAND, ...negOpts });
    const G = boost(Atr3, nt, { rho: RHO, solver: "exp" }).G;
    let rStar = G.length;
    if (early) { let best = -Infinity; for (let r = 1; r <= G.length; r++) { const h = makeHead(G.slice(0, r)); const vr = rate((y) => h.fires(y), Aval3), vfp = rate((y) => h.fires(y), nt.Neg); if (vr - vfp > best) { best = vr - vfp; rStar = r; } } }
    const head = makeHead(G.slice(0, rStar));
    const trec = rate((y) => head.fires(y), Ahe3), tfp = rate((y) => head.fires(y), nt.Neg);
    w.write(`   ${name.padEnd(15)} ${String(G.length).padStart(2)} ${String(rStar).padStart(3)}  ${(100 * trec).toFixed(1).padStart(6)}%  ${(100 * tfp).toFixed(1).padStart(6)}%\n`);
  }
};

main();
