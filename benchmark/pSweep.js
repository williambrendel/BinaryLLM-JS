"use strict";

// ============================================================================
// benchmark/pSweep.js — is the negative channel LEARNED or a NOVELTY artifact?
// Sweep build size B (→ |P|). Two negative gates on top of gate-1 (rare bits ∈ p⁺):
//   NOVELTY : reject if candidate has ANY common∉P bit          (p⁻ = ¬P, unlearned)
//   PEELED  : reject if candidate has a common∉P bit seen in ≥K gate-1 FP contexts (learned, floored)
// If NOVELTY's FP-cut collapses as |P| grows (|¬P|→0) while PEELED holds, the peel is real.
// Track held-out recall too: novelty over-rejects unseen positives; peeled shouldn't.
// Usage: TARGET=bank R=5 K=30 node --max-old-space-size=8192 benchmark/pSweep.js english.txt <corpus>
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
const R = Number(process.env.R || 5), D = R + 1, DD = Number(process.env.DELTA || 0.001), K = Number(process.env.K || 30);
const HOLD = Number(process.env.HOLD || 100);

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F;
  const pc = new Map(); const partsOf = (w) => { let a = pc.get(w); if (!a) { a = [...new Set(encodeWord(dict, w))]; pc.set(w, a); } return a; };
  const sents = []; const cnt = new Map(); let Ntok = 0;
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const ws = tokenizeStream(line.slice(seg.start, seg.end)).filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) { sents.push(ws); for (const x of ws) { cnt.set(x, (cnt.get(x) || 0) + 1); Ntok++; } }
  }
  const rareBitFreq = new Float64Array(DIM);   // global doc-freq per bit
  const band = (d) => (d === 1 ? 2 : d <= 3 ? 1 : 0);
  const st = new Int32Array(DIM).fill(-1); let sp = 0;
  const feat = (words, t) => { sp++; const y = []; for (let d = 1; d <= D; d++) { const q = t - d; if (q < 0) break; const off = band(d) * F; for (const a of partsOf(words[q])) { const b = a + off; if (st[b] !== sp) { st[b] = sp; y.push(b); } } } return y; };
  let Ntot = 0; for (const words of sents) for (let t = 1; t < words.length; t++) { const f = feat(words, t); Ntot++; for (const b of f) rareBitFreq[b]++; }
  const rare = (b) => rareBitFreq[b] / Ntot <= DD;

  const POS = [];
  for (const words of sents) for (let t = 1; t < words.length; t++) if (words[t] === TARGET) POS.push(feat(words, t));
  const held = POS.slice(POS.length - HOLD), pool = POS.slice(0, POS.length - HOLD);

  const w = process.stdout;
  w.write(`# pSweep [L|L1∪L2|C] — ${path.basename(corpusArg)} | TARGET=${TARGET} K=${K} | pool=${pool.length} held=${held.length}\n`);
  w.write(`   B     |P|     gate1 FP    NOVELTY(¬P): FP / recall     PEELED(≥${K}): FP / recall    |p⁻peel|\n`);

  for (const B of [25, 50, 100, 200, pool.length]) {
    const P = new Uint8Array(DIM), pPlus = new Uint8Array(DIM);
    for (let i = 0; i < B; i++) for (const b of pool[i]) { P[b] = 1; if (rare(b)) pPlus[b] = 1; }
    let Psz = 0; for (let b = 0; b < DIM; b++) if (P[b]) Psz++;
    const gate1 = (cand) => { let any = false; for (const b of cand) if (rare(b)) { any = true; if (!pPlus[b]) return false; } return any; };

    // pass 1: gate-1 FPs, tally common∉P bits
    const cnpCount = new Map(); const fpCnp = [];
    for (const words of sents) for (let t = 1; t < words.length; t++) {
      if (words[t] === TARGET) continue; const cand = feat(words, t);
      if (!gate1(cand)) continue;
      const cnp = []; for (const b of cand) if (!rare(b) && !P[b]) { cnp.push(b); cnpCount.set(b, (cnpCount.get(b) || 0) + 1); }
      fpCnp.push(cnp);
    }
    const learned = new Set(); for (const [b, c] of cnpCount) if (c >= K) learned.add(b);
    // FP: novelty passes iff no common∉P; peeled passes iff no learned bit
    let novFP = 0, peelFP = 0; for (const cnp of fpCnp) { if (cnp.length === 0) { novFP++; peelFP++; } else if (!cnp.some((b) => learned.has(b))) peelFP++; }
    // IN-SAMPLE recall on the build positives (mask-classifiable; negatives recall-safe here) + HELD-OUT for reference
    const recOf = (SET) => { let nov = 0, peel = 0; for (const cand of SET) { if (!gate1(cand)) continue; const cnp = []; for (const b of cand) if (!rare(b) && !P[b]) cnp.push(b); if (cnp.length === 0) nov++; if (!cnp.some((b) => learned.has(b))) peel++; } return { nov: nov / SET.length, peel: peel / SET.length }; };
    const inS = recOf(pool.slice(0, B)), ho = recOf(held);
    const pc = (x) => (100 * x).toFixed(0) + "%";
    w.write(`  ${String(B).padStart(3)}   ${String(Psz).padStart(5)}   ${String(fpCnp.length).padStart(7)}   nov ${String(novFP).padStart(7)} (R ${pc(inS.nov)} in / ${pc(ho.nov)} ho)   peel ${String(peelFP).padStart(7)} (R ${pc(inS.peel)} in / ${pc(ho.peel)} ho)  |p⁻|=${learned.size}\n`);
  }
};

main();
