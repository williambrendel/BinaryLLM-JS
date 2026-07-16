"use strict";

// benchmark/variants.js — single p⁺, three selection rules, same mask rule M_τ = ¬(OR ∖ p⁺):
//   OR      : p⁺ = {count_A ≥ 1}                 (baseline; M_τ = keep-all)
//   UNIQUE  : p⁺ = {count_A == 1}                (strictly unique; drop bits seen 2+ times)
//   CONTRAST: p⁺ = top-k bits by freq_A/freq_Ā   (enriched in positives vs negatives)
// coverage(x) = |p⁺ ∧ M_τ ∧ x| / |M_τ ∧ x|,  M_τ keeps b iff (b∈p⁺) or (b∉OR).
// Reports in-sample coverage pos/neg + ratio, corner (cov=1) FP-rate, and held-out recall (250/250).
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/variants.js english.txt <corpus>

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
const R = Number(process.env.R || 5), D = R + 1;

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F;
  const pc = new Map(); const partsOf = (w) => { let a = pc.get(w); if (!a) { a = [...new Set(encodeWord(dict, w))]; pc.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const ws = tokenizeStream(line.slice(seg.start, seg.end)).filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws);
  }
  const band = (d) => (d === 1 ? 2 : d <= 3 ? 1 : 0);
  const stmp = new Int32Array(DIM).fill(-1); let sp = 0;
  const feat = (s, t) => { sp++; const y = []; for (let d = 1; d <= D; d++) { const q = t - d; if (q < 0) break; const off = band(d) * F; for (const a of partsOf(s[q])) { const b = a + off; if (stmp[b] !== sp) { stmp[b] = sp; y.push(b); } } } return y; };

  const M = new Float64Array(DIM); let Ntot = 0; const POS = []; const neg = []; let ni = 0;
  let negTot = 0; for (const s of sents) for (let t = 1; t < s.length; t++) if (s[t] !== TARGET) negTot++;
  const stride = Math.max(1, Math.floor(negTot / 120000));
  for (const s of sents) for (let t = 1; t < s.length; t++) { const f = feat(s, t); Ntot++; for (const b of f) M[b]++; if (s[t] === TARGET) POS.push(f); else if (ni++ % stride === 0) neg.push(f); }
  const half = POS.length >> 1, train = POS.slice(0, half), held = POS.slice(half);
  const cntA = new Float64Array(DIM), OR = new Uint8Array(DIM); for (const f of train) for (const b of f) { cntA[b]++; }
  for (let b = 0; b < DIM; b++) if (cntA[b] > 0) OR[b] = 1;
  const A = train.length, Abar = Ntot - POS.length;
  const contrast = (b) => ((cntA[b] + 0.5) / (A + 1)) / ((M[b] - cntA[b] + 0.5) / (Abar + 1));

  // build a p⁺ (Uint8Array) for each rule
  const mkOR = () => { const p = OR.slice(); return { p, n: p.reduce((a, x) => a + x, 0) }; };
  const mkUniq = () => { const p = new Uint8Array(DIM); let n = 0; for (let b = 0; b < DIM; b++) if (cntA[b] === 1) { p[b] = 1; n++; } return { p, n }; };
  const mkContrast = (k) => { const bits = []; for (let b = 0; b < DIM; b++) if (OR[b]) bits.push(b); bits.sort((a, b) => contrast(b) - contrast(a)); const p = new Uint8Array(DIM); for (let i = 0; i < Math.min(k, bits.length); i++) p[bits[i]] = 1; return { p, n: Math.min(k, bits.length) }; };

  const measure = (name, pj) => {
    const p = pj.p; const keep = (b) => p[b] === 1 || OR[b] === 0;   // M_τ = ¬(OR ∖ p⁺)
    const cov = (SET) => { let s = 0, n = 0, empty = 0; for (const f of SET) { let pm = 0, kept = 0; for (const b of f) { if (!keep(b)) continue; kept++; if (p[b]) pm++; } if (kept === 0) { empty++; continue; } s += pm / kept; n++; } return { m: n ? s / n : 0, empty: empty / SET.length }; };
    const covTrain = cov(train), covHeld = cov(held), covNeg = cov(neg);
    // corner FP-rate: coverage == 1  (all kept bits ∈ p⁺)
    const fires = (f) => { let pm = 0, kept = 0; for (const b of f) { if (!keep(b)) continue; kept++; if (p[b]) pm++; } return kept > 0 && pm === kept; };
    let fp = 0; for (const f of neg) if (fires(f)) fp++;
    let rTr = 0; for (const f of train) if (fires(f)) rTr++;
    let rHe = 0; for (const f of held) if (fires(f)) rHe++;
    return { name, n: pj.n, covTrain: covTrain.m, covNeg: covNeg.m, ratio: covTrain.m / (covNeg.m || 1e-9), fpRate: fp / neg.length, recTr: rTr / train.length, recHe: rHe / held.length };
  };

  const w = process.stdout;
  w.write(`# variants — ${path.basename(corpusArg)} | TARGET=${TARGET} | |A_train|=${train.length} held=${held.length} negSample=${neg.length}  OR=${OR.reduce((a, x) => a + x, 0)} bits\n\n`);
  w.write(`  rule          |p⁺|    cov(pos)  cov(neg)  ratio    corner FP-rate   train-rec  held-rec\n`);
  const rows = [measure("OR", mkOR()), measure("UNIQUE", mkUniq()), measure("CONTRAST k=100", mkContrast(100)), measure("CONTRAST k=300", mkContrast(300)), measure("CONTRAST k=800", mkContrast(800))];
  for (const r of rows) w.write(`  ${r.name.padEnd(14)} ${String(r.n).padStart(5)}   ${r.covTrain.toFixed(3)}     ${r.covNeg.toFixed(3)}     ${r.ratio.toFixed(2).padStart(5)}    ${(100 * r.fpRate).toFixed(3).padStart(6)}%         ${(100 * r.recTr).toFixed(0)}%       ${(100 * r.recHe).toFixed(1)}%\n`);
};

main();
