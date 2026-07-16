"use strict";

// ============================================================================
// benchmark/contrastGate.js — does the CONTRAST criterion scale where hard-exclusion decays?
// Exact 3F [L|L1∪L2|C] signature. Two selection-rule changes (gate unchanged):
//   p⁺ = { b : count_A(b) ≥ ν  ∧  freq_A(b) > freq_Ā(b) }           ν ∈ {1,2,3}
//   p⁻ : (hard)  peel from {b∉p⁺} by FP-coverage (≥K)   vs
//        (ratio) { b : freq_Ā(b)/freq_A(b) > ρ }                     ρ ∈ {10,50}   (disjoint from p⁺ by sign)
//   M_τ = ¬(M_glob ∧ p⁺), computed against the new p⁺.  score,sweep unchanged.
// Run at |A|=250 AND |A|=1000; report |p⁺|,|p⁻|, FP-rate, recall for each criterion at both |A|.
// Usage: TARGET=state R=5 K=30 node --max-old-space-size=8192 benchmark/contrastGate.js english.txt <corpus>
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

const TARGET = process.env.TARGET || "state";
const R = Number(process.env.R || 5), D = R + 1, DD = Number(process.env.DELTA || 0.001), K = Number(process.env.K || 30);

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
  const st = new Int32Array(DIM).fill(-1); let sp = 0;
  const feat = (words, t) => { sp++; const y = []; for (let d = 1; d <= D; d++) { const q = t - d; if (q < 0) break; const off = band(d) * F; for (const a of partsOf(words[q])) { const b = a + off; if (st[b] !== sp) { st[b] = sp; y.push(b); } } } return y; };

  // one pass: M_all, N, all target feats, strided negative sample
  const M = new Float64Array(DIM); let N = 0; const tgt = [], neg = []; let ni = 0; const NEGN = 150000;
  // first count negatives for stride
  let negTot = 0; for (const words of sents) for (let t = 1; t < words.length; t++) if (words[t] !== TARGET) negTot++;
  const stride = Math.max(1, Math.floor(negTot / NEGN));
  for (const words of sents) for (let t = 1; t < words.length; t++) { const f = feat(words, t); N++; for (const b of f) M[b]++; if (words[t] === TARGET) tgt.push(f); else if (ni++ % stride === 0) neg.push(f); }
  const rareBit = (b) => M[b] / N <= DD;

  // full target doc-count + negative rate (fixed)
  const cTgt = new Float64Array(DIM); for (const f of tgt) for (const b of f) cTgt[b]++;
  const Abar = N - tgt.length; const fAbar = (b) => (M[b] - cTgt[b] + 0.5) / (Abar + 1);

  const w = process.stdout;
  w.write(`# contrastGate 3F [L|L1∪L2|C] — ${path.basename(corpusArg)} | TARGET=${TARGET} δ=${DD} K=${K} | |A_full|=${tgt.length}  negSample=${neg.length}\n`);

  for (const A of [250, 1000]) {
    if (A > tgt.length) continue;
    const sample = tgt.slice(0, A);
    const cA = new Float64Array(DIM); for (const f of sample) for (const b of f) cA[b]++;
    const fA = (b) => (cA[b] + 0.5) / (A + 1);
    w.write(`\n════ |A|=${A} ════\n   ν  criterion     |p⁺|    |p⁻|    FP-rate(t1/t.5)     recall(in)\n`);

    for (const nu of [1, 2, 3]) {
      // p⁺ = count_A≥ν ∧ freq_A>freq_Ā
      const pPlus = new Uint8Array(DIM); let ppn = 0;
      for (let b = 0; b < DIM; b++) if (cA[b] >= nu && fA(b) > fAbar(b)) { pPlus[b] = 1; ppn++; }
      const keep = (b) => rareBit(b) || !pPlus[b];   // M_τ = ¬(common ∧ p⁺)
      // score with a given p⁻ predicate: (posM − neg)/kept ; posM=rare∩p⁺, neg=p⁻ (kept)
      const evalSet = (isNeg) => {
        const score = (f) => { let posM = 0, neg2 = 0, kept = 0; for (const b of f) { if (!keep(b)) continue; kept++; if (pPlus[b]) { if (rareBit(b)) posM++; } else if (isNeg(b)) neg2++; } return kept > 0 ? { s1: (posM - neg2) / kept, posM, neg2, kept } : null; };
        let f1 = 0, fh = 0, r1 = 0; for (const f of neg) { const r = score(f); if (r && r.s1 >= 1) f1++; if (r && r.s1 >= 0.5) fh++; }
        for (const f of sample) { const r = score(f); if (r && r.s1 >= 1) r1++; }
        return { fp1: f1 / neg.length, fph: fh / neg.length, rec: r1 / sample.length };
      };
      // hard-exclusion peel: peel common∉p⁺ from gate-1 FPs (≥K), gate-1 = all rare bits ∈ p⁺
      const cnp = new Map(); for (const f of neg) { let ok = true; for (const b of f) if (rareBit(b) && !pPlus[b]) { ok = false; break; } if (!ok) continue; for (const b of f) if (!rareBit(b) && !pPlus[b]) cnp.set(b, (cnp.get(b) || 0) + 1); }
      const peel = new Uint8Array(DIM); let peN = 0; for (const [b, c] of cnp) if (c >= K) { peel[b] = 1; peN++; }
      // report rows
      const row = (name, pm, isNeg) => { const e = evalSet(isNeg); w.write(`   ${nu}  ${name.padEnd(12)} ${String(ppn).padStart(6)}  ${String(pm).padStart(6)}   ${(100 * e.fp1).toFixed(3)}% / ${(100 * e.fph).toFixed(2)}%     ${(100 * e.rec).toFixed(1)}%\n`); };
      row("none", 0, () => false);
      row(`hard(≥${K})`, peN, (b) => peel[b] === 1);
      for (const rho of [10, 50]) { let pmn = 0; for (let b = 0; b < DIM; b++) if (!pPlus[b] && fAbar(b) / fA(b) > rho) pmn++; row(`ratio ρ=${rho}`, pmn, (b) => !pPlus[b] && fAbar(b) / fA(b) > rho); }
    }
  }
};

main();
