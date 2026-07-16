"use strict";

// ============================================================================
// benchmark/bandConcat.js — does CONCATENATING [L|C] (separate namespaces) beat
// UNIONING them (one namespace), for the causal next-word task? Predict word[t]==TARGET
// from the causal context at t-1: L = OR(C[t-1-r .. t-2]) in [0,F); C = C[t-1].
//   union  LC = L ∪ C                    (F dims)   — current buildSignature.js:214
//   concat LC = L ∪ {a+F : a∈C}          (2F dims)  — the spec's positional typing
// Metrics per representation: |p|, density, coverage pos/neg, and RECONSTRUCTION
// (fraction of p's atoms present in OR of ALL negatives). Union → ~100% (blob); the
// question is whether concat leaves bank-exclusive bits.
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/bandConcat.js english.txt <corpus>
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { encodeWord } from "../src/core/signatures/wordEncoder.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGET = process.env.TARGET || "bank";
const R = Number(process.env.R || 5);
const NEG_N = Number(process.env.NEG || 3000);

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size();
  const cache = new Map();
  const atomsOf = (w) => { let a = cache.get(w); if (!a) { a = Int32Array.from(new Set(encodeWord(dict, w))); cache.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const toks = tokenizeStream(line.slice(seg.start, seg.end));
    const ws = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws.map((x) => ({ w: x, atoms: atomsOf(x) })));
  }

  const stamp = new Int32Array(2 * F).fill(-1); let pid = 0;
  // LC feature for predicting word at position t (context = t-1 back to t-1-r).
  // returns {u:[ids in 0..F), c:[ids in F..2F)} where u = L∪C atoms, cc = C atoms offset
  const feat = (s, t) => {
    pid++; const u = [], cc = [];
    for (let q = Math.max(0, t - 1 - R); q <= t - 1; q++) for (const a of s[q].atoms) if (stamp[a] !== pid) { stamp[a] = pid; u.push(a); }   // union band [0,F)
    const Cw = s[t - 1].atoms; for (const a of Cw) { const o = a + F; if (stamp[o] !== pid) { stamp[o] = pid; cc.push(o); } }                  // C band offset
    return { u, cc };
  };

  // collect POS (word[t]==TARGET) and strided NEG
  let posTot = 0, negTot = 0;
  for (const s of sents) for (let t = 1; t < s.length; t++) { if (s[t].w === TARGET) posTot++; else negTot++; }
  const step = Math.max(1, Math.floor(negTot / NEG_N));
  const POS = [], NEG = []; let ni = 0;
  for (const s of sents) for (let t = 1; t < s.length; t++) { const f = feat(s, t); if (s[t].w === TARGET) POS.push(f); else { if (ni % step === 0) NEG.push(f); ni++; } }

  // build p for union (dim F) and concat (dim 2F)
  const pU = new Uint8Array(F), pC = new Uint8Array(2 * F); let pUn = 0, pCn = 0;
  for (const f of POS) { for (const a of f.u) if (!pU[a]) { pU[a] = 1; pUn++; } for (const a of f.u) if (!pC[a]) { pC[a] = 1; pCn++; } for (const a of f.cc) if (!pC[a]) { pC[a] = 1; pCn++; } }

  // per-sample coverage over a given SET
  const cov = (P, get, SET) => { let s = 0, s2 = 0, n = 0; for (const f of SET) { const y = get(f); let k = 0; for (const a of y) if (P[a]) k++; const c = k / (y.length || 1); s += c; s2 += c * c; n++; } const m = s / n; return { m, sd: Math.sqrt(Math.max(0, s2 / n - m * m)) }; };
  const uGet = (f) => f.u, cGet = (f) => [...f.u, ...f.cc];
  const meanLen = (get) => POS.reduce((a, f) => a + get(f).length, 0) / POS.length;
  const covUnP = cov(pU, uGet, POS), covUnN = cov(pU, uGet, NEG);
  const covCnP = cov(pC, cGet, POS), covCnN = cov(pC, cGet, NEG);

  // reconstruction: OR of ALL negatives' atoms, ∩ p
  const orNU = new Uint8Array(F), orNC = new Uint8Array(2 * F);
  for (const s of sents) for (let t = 1; t < s.length; t++) { if (s[t].w === TARGET) continue; const f = feat(s, t); for (const a of f.u) { orNU[a] = 1; orNC[a] = 1; } for (const a of f.cc) orNC[a] = 1; }
  let recU = 0; for (let a = 0; a < F; a++) if (pU[a] && orNU[a]) recU++;
  let recC = 0; for (let a = 0; a < 2 * F; a++) if (pC[a] && orNC[a]) recC++;

  // ---- DETECTOR at perfect coverage (=1.0) + global doc-freq M[a] over the WHOLE corpus ----
  const covOf = (P, y) => { let k = 0; for (const a of y) if (P[a]) k++; return k / (y.length || 1); };
  const M = new Float64Array(2 * F);   // # positions whose [L|C] feature contains atom a (both bands)
  let Ntot = 0, firedU = 0, tpU = 0, firedC = 0, tpC = 0;
  for (const s of sents) for (let t = 1; t < s.length; t++) {
    const f = feat(s, t); Ntot++; const isPos = s[t].w === TARGET;
    for (const a of f.u) M[a]++; for (const a of f.cc) M[a]++;
    if (covOf(pU, f.u) >= 1) { firedU++; if (isPos) tpU++; }
    if (covOf(pC, [...f.u, ...f.cc]) >= 1) { firedC++; if (isPos) tpC++; }
  }

  const w = process.stdout;
  w.write(`# bandConcat — ${path.basename(corpusArg)} | TARGET=${TARGET} r=${R} sparse | F=${F}\n`);
  w.write(`# POS n=${POS.length}  NEG n=${NEG.length} (stride ${step})\n\n`);
  const block = (name, dim, pn, mlen, cvP, cvN, rec) => {
    w.write(`── ${name}  (dim=${dim}) ──\n`);
    w.write(`   |p| = ${pn}  (${(100 * pn / dim).toFixed(2)}% of dim)   mean|y| pos = ${mlen.toFixed(1)}\n`);
    w.write(`   coverage p vs pos = ${cvP.m.toFixed(4)} ± ${cvP.sd.toFixed(4)}   (in-sample: pos built p)\n`);
    w.write(`   coverage p vs neg = ${cvN.m.toFixed(4)} ± ${cvN.sd.toFixed(4)}\n`);
    w.write(`   → coverage ratio pos/neg = ${(cvP.m / cvN.m).toFixed(3)}\n`);
    w.write(`   RECONSTRUCTION: OR(all negatives) ∩ p = ${rec}/${pn} = ${(100 * rec / pn).toFixed(1)}%   bank-exclusive bits = ${pn - rec}\n\n`);
  };
  block("UNION  [L∪C]", F, pUn, meanLen(uGet), covUnP, covUnN, recU);
  block("CONCAT [L|C]", 2 * F, pCn, meanLen(cGet), covCnP, covCnN, recC);

  w.write(`── DETECTOR: coverage ≥ 1.0 over whole corpus (N=${Ntot}) ──\n`);
  w.write(`  UNION  [L∪C]:  fired=${firedU} (${(100 * firedU / Ntot).toFixed(1)}%)  TP=${tpU}/${POS.length}  FALSE POSITIVES=${firedU - tpU}  precision=${(100 * tpU / firedU).toFixed(4)}%\n`);
  w.write(`  CONCAT [L|C]:  fired=${firedC} (${(100 * firedC / Ntot).toFixed(1)}%)  TP=${tpC}/${POS.length}  FALSE POSITIVES=${firedC - tpC}  precision=${(100 * tpC / firedC).toFixed(4)}%\n`);

  // ---- COMMON bits as a NEGATIVE MASK: mask BOTH p and candidate, normalize by |masked candidate| ----
  //   coverage = |p' ∩ y'| / |y'|  where ' = drop atoms with global doc-freq > δ (the common bits).
  //   candidates with no rare bits (|y'|=0) are undefined and excluded (the short all-common contexts).
  const covMasked = (SET, d, P) => {
    let s = 0, s2 = 0, n = 0, empty = 0;
    for (const f of SET) { const y = cGet(f); let yr = 0, k = 0; for (const a of y) if (M[a] / Ntot <= d) { yr++; if (P[a]) k++; } if (yr === 0) { empty++; continue; } const c = k / yr; s += c; s2 += c * c; n++; }
    const m = n ? s / n : 0; return { m, sd: n ? Math.sqrt(Math.max(0, s2 / n - m * m)) : 0, empty, tot: n + empty };
  };
  w.write(`\n── CONCAT [L|C], COMMON bits as negative mask — coverage = |p'∩y'|/|y'| (rare subspace only) ──\n`);
  w.write(`   δ        |p'|    cov pos (rare)      %pos∅   cov neg (rare)      %neg∅   ratio\n`);
  const DELS = [Infinity, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001];
  for (const d of DELS) {
    let pn = 0; for (let a = 0; a < 2 * F; a++) if (pC[a] && M[a] / Ntot <= d) pn++;
    const cp = covMasked(POS, d, pC), cn = covMasked(NEG, d, pC);
    w.write(`   ${(d === Infinity ? "∞" : d).toString().padEnd(6)} ${String(pn).padStart(6)}   ${cp.m.toFixed(4)} ± ${cp.sd.toFixed(4)}   ${(100 * cp.empty / cp.tot).toFixed(0).padStart(3)}%   ${cn.m.toFixed(4)} ± ${cn.sd.toFixed(4)}   ${(100 * cn.empty / cn.tot).toFixed(0).padStart(3)}%   ${(cp.m / (cn.m || 1e-9)).toFixed(2)}\n`);
  }

  // ---- HELD-OUT: build p from half the positives, measure on the other half (removes the pos=1.0 tautology) ----
  const pHO = new Uint8Array(2 * F);
  const POSbuild = POS.filter((_, i) => i % 2 === 0), POStest = POS.filter((_, i) => i % 2 === 1);
  for (const f of POSbuild) { for (const a of f.u) pHO[a] = 1; for (const a of f.cc) pHO[a] = 1; }
  w.write(`\n── HELD-OUT (p built from ${POSbuild.length} pos, tested on ${POStest.length} held-out pos + neg) ──\n`);
  w.write(`   δ       cov pos-HO (rare)   %pos∅   cov neg (rare)      %neg∅   ratio\n`);
  for (const d of DELS) {
    const cp = covMasked(POStest, d, pHO), cn = covMasked(NEG, d, pHO);
    w.write(`   ${(d === Infinity ? "∞" : d).toString().padEnd(6)}  ${cp.m.toFixed(4)} ± ${cp.sd.toFixed(4)}   ${(100 * cp.empty / cp.tot).toFixed(0).padStart(3)}%   ${cn.m.toFixed(4)} ± ${cn.sd.toFixed(4)}   ${(100 * cn.empty / cn.tot).toFixed(0).padStart(3)}%   ${(cp.m / (cn.m || 1e-9)).toFixed(2)}\n`);
  }

  // ---- MASKED perfect-coverage detector at δ=0.001: fire iff ALL rare bits of candidate are in p ----
  const DD = 0.001;
  const parts = dict.allParts();
  let mFired = 0, mTP = 0, mUndef = 0, mN = 0; const fpEx = [];
  for (const s of sents) for (let t = 1; t < s.length; t++) {
    const f = feat(s, t); mN++; const y = cGet(f);
    let yr = 0, allIn = true;
    for (const a of y) if (M[a] / Ntot <= DD) { yr++; if (!pC[a]) { allIn = false; break; } }
    if (yr === 0) { mUndef++; continue; }
    if (allIn) {
      mFired++;
      if (s[t].w === TARGET) mTP++;
      else if (fpEx.length < 20 && t % 7 === 0) {   // sample false positives (next word ≠ bank)
        const matched = []; for (const a of y) if (M[a] / Ntot <= DD && pC[a]) { const inL = a < F; matched.push(`${parts[inL ? a : a - F].value}@${inL ? "L" : "C"}`); }
        const ctx = []; for (let q = Math.max(0, t - 1 - R); q <= t - 1; q++) ctx.push(s[q].w);
        fpEx.push({ ctx: ctx.join(" "), tgt: s[t].w, matched, yr, total: y.length });
      }
    }
  }
  w.write(`\n── MASKED detector: perfect coverage (all rare bits ∈ p), δ=${DD}, in-sample p, over corpus (N=${mN}) ──\n`);
  w.write(`  fired=${mFired} (${(100 * mFired / mN).toFixed(3)}%)  TP=${mTP}/${POS.length} (recall ${(100 * mTP / POS.length).toFixed(1)}%)  FALSE POSITIVES=${mFired - mTP}  precision=${(100 * mTP / mFired).toFixed(3)}%\n`);
  w.write(`  (undefined: ${mUndef} positions with no rare bits = ${(100 * mUndef / mN).toFixed(1)}% of corpus, not classifiable)\n`);
  w.write(`\n── sample FALSE POSITIVES (fired as 'bank', actual next word ≠ bank) ──\n`);
  for (const e of fpEx) w.write(`  {${e.ctx}} → [${e.tgt}]   (${e.total} atoms total → ${e.total - e.yr} masked as common, ${e.yr} rare: ${e.matched.join(" ")})\n`);

  // ---- band split of p and the δ=0.001 common-mask (L-union band [0,F) vs C band [F,2F)) ----
  let pL = 0, pCb = 0, comL = 0, comC = 0, rareL = 0, rareC = 0;
  for (let a = 0; a < 2 * F; a++) if (pC[a]) { const inL = a < F; if (inL) pL++; else pCb++; if (M[a] / Ntot > DD) { if (inL) comL++; else comC++; } else { if (inL) rareL++; else rareC++; } }
  // ---- probe: sparse peel of specific words + each atom's global L-band doc-freq ----
  w.write(`\n── word peel probe (atom value : global freq : rare?≤${DD}) ──\n`);
  for (const word of ["built", "masonry", "stone", "however", "governor", "bank"]) {
    const at = [...new Set(encodeWord(dict, word))];
    const cells = at.map((a) => { const fr = M[a] / Ntot; return `${parts[a].value}:${fr.toExponential(1)}:${fr <= DD ? "RARE" : "common"}`; });
    w.write(`  ${word.padEnd(9)} ${at.length} atoms → ${cells.join("  ")}\n`);
  }

  w.write(`\n── band split of concat p (δ=${DD}) ──\n`);
  w.write(`   L-union band [0,F):  total=${pL}   common(masked)=${comL}   rare(kept)=${rareL}\n`);
  w.write(`   C band     [F,2F):   total=${pCb}   common(masked)=${comC}   rare(kept)=${rareC}\n`);
  w.write(`   → of ${comL + comC} common bits removed: ${comL} (${(100 * comL / (comL + comC)).toFixed(0)}%) in L-union, ${comC} (${(100 * comC / (comL + comC)).toFixed(0)}%) in C\n`);
  w.write(`   → of ${rareL + rareC} rare bits kept:    ${rareL} (${(100 * rareL / (rareL + rareC)).toFixed(0)}%) in L-union, ${rareC} (${(100 * rareC / (rareL + rareC)).toFixed(0)}%) in C\n`);
};

main();
