"use strict";

// ============================================================================
// benchmark/adaptMask.js — ADAPTED mask = global mask ∩ positive support.
// Global mask removes ALL globally-common bits (over-masking: it also removes common bits that
// never occur in bank → discriminative negatives lost). Adapted mask removes a common bit ONLY if
// it also appears in some positive signature. So:
//   keep(b) = (global-rare)  OR  (b ∉ positive support P)
// A candidate is rejected if it has ANY kept bit ∉ p⁺ — i.e. a rare bit outside p⁺ (gate-1) OR a
// COMMON bit absent from every positive (the recovered negative). Positives' common bits are all in
// P ⇒ still masked ⇒ recall-safe on train. Compares gate-1 vs gate-1+adapted, held-out.
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/adaptMask.js english.txt <corpus>
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { encodeWord } from "../src/core/signatures/wordEncoder.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";
import { kindToString } from "../src/core/parts/kind.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGET = process.env.TARGET || "bank";
const R = Number(process.env.R || 5), D = R + 1, DD = Number(process.env.DELTA || 0.001);

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F, parts = dict.allParts();
  const pc = new Map(); const partsOf = (w) => { let a = pc.get(w); if (!a) { a = [...new Set(encodeWord(dict, w))]; pc.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const ws = tokenizeStream(line.slice(seg.start, seg.end)).filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws);
  }
  const band = (d) => (d === 1 ? 2 : d <= 3 ? 1 : 0);
  const stamp = new Int32Array(DIM).fill(-1); let sp = 0;
  const feat = (words, t) => { sp++; const y = []; for (let d = 1; d <= D; d++) { const q = t - d; if (q < 0) break; const off = band(d) * F; for (const a of partsOf(words[q])) { const b = a + off; if (stamp[b] !== sp) { stamp[b] = sp; y.push(b); } } } return y; };   // FULL signature (all bits)

  const POS = [];
  for (const words of sents) for (let t = 1; t < words.length; t++) if (words[t] === TARGET) POS.push(feat(words, t));
  const Pb = POS.filter((_, i) => i % 2 === 0), Pt = POS.filter((_, i) => i % 2 === 1);

  // global doc-freq M per bit
  const M = new Float64Array(DIM); let Ntot = 0;
  for (const words of sents) for (let t = 1; t < words.length; t++) { const f = feat(words, t); Ntot++; for (const b of f) M[b]++; }
  const rareBit = (b) => M[b] / Ntot <= DD;

  // positive support P and p⁺ (rare positive bits) from build half
  const P = new Uint8Array(DIM), pPlus = new Uint8Array(DIM); let ppn = 0;
  for (const cand of Pb) for (const b of cand) { P[b] = 1; if (rareBit(b) && !pPlus[b]) { pPlus[b] = 1; ppn++; } }

  // gates
  const gate1 = (cand) => { let any = false; for (const b of cand) if (rareBit(b)) { any = true; if (!pPlus[b]) return false; } return any; };
  const gateA = (cand) => { let any = false; for (const b of cand) { if (rareBit(b)) { any = true; if (!pPlus[b]) return false; } else if (!P[b]) return false; } return any; };   // + common bit must be in P

  // bit-count accounting over OCCURRING bits (M>0)
  let occ = 0, gRare = 0, gCommon = 0, adaptRem = 0, commonNotP = 0;
  for (let b = 0; b < DIM; b++) if (M[b] > 0) { occ++; if (rareBit(b)) gRare++; else { gCommon++; if (P[b]) adaptRem++; else commonNotP++; } }

  const rec = (SET, g) => { let k = 0; for (const c of SET) if (g(c)) k++; return k / SET.length; };
  const r1 = rec(Pt, gate1), rA = rec(Pt, gateA);

  let N = 0, fp1 = 0, fpA = 0; const rej = []; const activeNeg = new Set();
  for (const words of sents) for (let t = 1; t < words.length; t++) {
    if (words[t] === TARGET) continue; N++;
    const cand = feat(words, t);
    if (!gate1(cand)) continue; fp1++;
    if (gateA(cand)) fpA++;
    else { for (const b of cand) if (!rareBit(b) && !P[b]) activeNeg.add(b); if (rej.length < 10 && t % 5 === 0) { const bad = cand.filter((b) => !rareBit(b) && !P[b]).map((b) => parts[b % F].value + ":" + kindToString(parts[b % F].kind)[0]); rej.push({ ctx: words.slice(Math.max(0, t - D), t).join(" "), tgt: words[t], bad }); } }
  }

  const w = process.stdout;
  w.write(`# adaptMask [L|L1∪L2|C] — ${path.basename(corpusArg)} | TARGET=${TARGET} δ=${DD} | p⁺=${ppn} rare bits, |P|=${P.reduce((a, b) => a + b, 0)} support bits\n\n`);
  w.write(`                          held-out recall (Pt=${Pt.length})   FALSE POSITIVES (of ${N})   precision*\n`);
  w.write(`  gate-1 (global mask)     ${(100 * r1).toFixed(1)}%                        ${fp1}                ${(100 * (r1 * Pt.length) / (r1 * Pt.length + fp1)).toFixed(3)}%\n`);
  w.write(`  gate-1 + adapted mask    ${(100 * rA).toFixed(1)}%                        ${fpA}                ${(100 * (rA * Pt.length) / (rA * Pt.length + fpA)).toFixed(3)}%\n`);
  w.write(`\n  Δ: false positives ${fp1}→${fpA} (${(100 * (fp1 - fpA) / fp1).toFixed(1)}% cut)   recall ${(100 * r1).toFixed(1)}%→${(100 * rA).toFixed(1)}% (Δ ${(100 * (r1 - rA)).toFixed(1)} pts)\n`);
  w.write(`\n── BIT COUNTS (over ${occ} occurring bits, DIM=${DIM}) ──\n`);
  w.write(`  global mask   : keeps ${gRare} rare, removes ${gCommon} common\n`);
  w.write(`  adapted mask  : removes ${adaptRem} (common ∩ P), keeps ${gRare + commonNotP} = ${gRare} rare + ${commonNotP} common-∉-P\n`);
  w.write(`  → adaptation UN-masks ${commonNotP} common bits (${(100 * commonNotP / gCommon).toFixed(1)}% of the ${gCommon} the global mask removed)\n`);
  w.write(`\n  positive parts p⁺ (rare ∩ P)          = ${ppn}\n`);
  w.write(`  positive support |P| (all pos bits)   = ${P.reduce((a, b) => a + b, 0)}  (${ppn} rare + ${P.reduce((a, b) => a + b, 0) - ppn} common)\n`);
  w.write(`  ACTIVE negative parts (common ∉ P that rejected ≥1 FP) = ${activeNeg.size}\n`);
  // ---- SIGNED DIVISIVE SCORE: (|p⁺ ∧ M_τ ∧ x| − |p⁻ ∧ x|) / |M_τ ∧ x|, sweep threshold t ----
  //   p⁺ = P (full positive support, rare+common) — MASKED by M_τ (rare) in the numerator
  //   p⁻ = common ∉ P — applied UN-masked (disjoint from M_τ and from p⁺∧M_τ)
  //   M_τ = rare bits (the keep-set). Hard adapted gate ≡ score ≥ 1.
  const scoreOf = (cand) => {
    let mtx = 0, posM = 0, neg = 0;                    // |M_τ∧x|, |p⁺∧M_τ∧x|, |p⁻∧x|
    for (const b of cand) {
      if (rareBit(b)) { mtx++; if (P[b]) posM++; }     // p⁺=P masked by M_τ  ⇒  rare AND in support
      else if (!P[b]) neg++;                            // p⁻ = common bit ∉ P (un-masked)
    }
    return mtx > 0 ? (posM - neg) / mtx : null;
  };
  const TS = [1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0, -2.0];
  const ptS = Pt.map(scoreOf);
  const fpAt = new Array(TS.length).fill(0);
  for (const words of sents) for (let t = 1; t < words.length; t++) {
    if (words[t] === TARGET) continue; const s = scoreOf(feat(words, t)); if (s === null) continue;
    for (let j = 0; j < TS.length; j++) if (s >= TS[j]) fpAt[j]++;
  }
  w.write(`\n── DENOMINATOR = |M_τ∧x| (candidate length)   t=1.0 ≡ hard gate ──\n   t        held-out recall   false positives   precision*\n`);
  for (let j = 0; j < TS.length; j++) {
    const rc = ptS.filter((s) => s !== null && s >= TS[j]).length / Pt.length, fp = fpAt[j];
    w.write(`   ${TS[j].toFixed(2).padStart(5)}    ${(100 * rc).toFixed(1).padStart(6)}%           ${String(fp).padStart(9)}       ${(100 * (rc * Pt.length) / (rc * Pt.length + fp)).toFixed(3)}%\n`);
  }

  // ALTERNATIVE denominator = |p⁺∧M_τ∧x| + |p⁻∧x| (total evidence examined) — score ∈ [−1,1]
  const scoreAltOf = (cand) => { let posM = 0, neg = 0; for (const b of cand) { if (rareBit(b)) { if (P[b]) posM++; } else if (!P[b]) neg++; } return (posM + neg > 0) ? (posM - neg) / (posM + neg) : null; };
  const TA = [1.0, 0.75, 0.5, 0.34, 0.2, 0.0, -0.34, -1.0];
  const ptA = Pt.map(scoreAltOf); const faAt = new Array(TA.length).fill(0);
  for (const words of sents) for (let t = 1; t < words.length; t++) { if (words[t] === TARGET) continue; const s = scoreAltOf(feat(words, t)); if (s === null) continue; for (let j = 0; j < TA.length; j++) if (s >= TA[j]) faAt[j]++; }
  w.write(`\n── DENOMINATOR = |p⁺∧x|+|p⁻∧x| (total evidence)   score ∈ [−1,1] ──\n   t        held-out recall   false positives   precision*\n`);
  for (let j = 0; j < TA.length; j++) {
    const rc = ptA.filter((s) => s !== null && s >= TA[j]).length / Pt.length, fp = faAt[j];
    w.write(`   ${TA[j].toFixed(2).padStart(5)}    ${(100 * rc).toFixed(1).padStart(6)}%           ${String(fp).padStart(9)}       ${(100 * (rc * Pt.length) / (rc * Pt.length + fp)).toFixed(3)}%\n`);
  }

  // REPARAMETERIZED gate: |p⁻∧x| < (1−t)|M_τ∧x|, M_τ=rare∩P, p⁻=¬P (rare∉P + common∉P) — score = 1 − neg/posM
  const scoreRe = (cand) => { let posM = 0, neg = 0; for (const b of cand) { if (P[b]) { if (rareBit(b)) posM++; } else neg++; } return posM > 0 ? 1 - neg / posM : null; };
  const TR = [1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0, -2.0];
  const ptR = Pt.map(scoreRe); const frAt = new Array(TR.length).fill(0);
  for (const words of sents) for (let t = 1; t < words.length; t++) { if (words[t] === TARGET) continue; const s = scoreRe(feat(words, t)); if (s === null) continue; for (let j = 0; j < TR.length; j++) if (s >= TR[j]) frAt[j]++; }
  w.write(`\n── REPARAM |p⁻∧x| < (1−t)|M_τ∧x|  (M_τ=rare∩P, p⁻=¬P)   t=1.0 ≡ hard gate ──\n   t        held-out recall   false positives   precision*\n`);
  for (let j = 0; j < TR.length; j++) {
    const rc = ptR.filter((s) => s !== null && s >= TR[j]).length / Pt.length, fp = frAt[j];
    w.write(`   ${TR[j].toFixed(2).padStart(5)}    ${(100 * rc).toFixed(1).padStart(6)}%           ${String(fp).padStart(9)}       ${(100 * (rc * Pt.length) / (rc * Pt.length + fp)).toFixed(3)}%\n`);
  }

  // CORRECT: M_τ=¬(common∧P). p⁺∧M_τ = rare∩P (posM); p⁻∧M_τ = common∉P (negC, the 745);
  // rare∉P is kept (dilutes denominator) but NOT subtracted.  |M_τ∧x| = posM + rare∉P + negC.
  const scoreC = (cand) => { let posM = 0, rnp = 0, negC = 0; for (const b of cand) { if (P[b]) { if (rareBit(b)) posM++; } else if (rareBit(b)) rnp++; else negC++; } const kept = posM + rnp + negC; return kept > 0 ? (posM - negC) / kept : null; };
  const TC = [1.0, 0.9, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0];
  const ptC = Pt.map(scoreC); const fcAt = new Array(TC.length).fill(0);
  // corr(|M_τ∧x|, |x|) over corpus non-bank positions
  let cn = 0, sk = 0, st = 0, skk = 0, stt = 0, skt = 0;
  for (const words of sents) for (let t = 1; t < words.length; t++) {
    if (words[t] === TARGET) continue; const cand = feat(words, t);
    let posM = 0, rnp = 0, negC = 0, cip = 0; for (const b of cand) { if (P[b]) { if (rareBit(b)) posM++; else cip++; } else if (rareBit(b)) rnp++; else negC++; }
    const kept = posM + rnp + negC, tot = kept + cip;
    cn++; sk += kept; st += tot; skk += kept * kept; stt += tot * tot; skt += kept * tot;
    const s = kept > 0 ? (posM - negC) / kept : null; if (s === null) continue; for (let j = 0; j < TC.length; j++) if (s >= TC[j]) fcAt[j]++;
  }
  const corr = (cn * skt - sk * st) / (Math.sqrt(cn * skk - sk * sk) * Math.sqrt(cn * stt - st * st));
  w.write(`\n── CORRECT M_τ=¬(common∧P): (|p⁺∧M_τ∧x|−|p⁻∧M_τ∧x|)/|M_τ∧x|, p⁻=common∉P   t=1.0 ≡ hard gate ──\n   t        held-out recall   false positives   precision*\n`);
  for (let j = 0; j < TC.length; j++) {
    const rc = ptC.filter((s) => s !== null && s >= TC[j]).length / Pt.length, fp = fcAt[j];
    w.write(`   ${TC[j].toFixed(2).padStart(5)}    ${(100 * rc).toFixed(1).padStart(6)}%           ${String(fp).padStart(9)}       ${(100 * (rc * Pt.length) / (rc * Pt.length + fp)).toFixed(3)}%\n`);
  }
  w.write(`\n  corr(|M_τ∧x|, |x|) = ${corr.toFixed(4)}   (mean |M_τ∧x|=${(sk / cn).toFixed(1)}, mean |x|=${(st / cn).toFixed(1)}; M_τ removes only ${adaptRem} of ${occ} bits)\n`);

  w.write(`\n── sample adapted-mask rejections (common bit absent from ALL positives) ──\n`);
  for (const e of rej) w.write(`  {${e.ctx}} → [${e.tgt}]   killer common bits (∉ P): ${e.bad.slice(0, 6).join(" ")}\n`);
};

main();
