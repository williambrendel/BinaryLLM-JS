"use strict";

// ============================================================================
// benchmark/baseline.js — THE reference config, end-to-end, one pass.
// Signature: 3F [L | L1∪L2 | C], sparse peel.  Mask: M_τ = ¬(M_glob ∧ p⁺).
// Score: (|p⁺∧M_τ∧x| − |p⁻∧M_τ∧x|)/|M_τ∧x| > t.   p⁺,P from ALL positives (recall = mask ~90%).
// p⁻ options compared: ∅ (positive-only), full common∉P, and PEELED (common∉P seen in ≥K gate-1 FPs, ⊊ ¬P).
// Reports recall (in-sample, classifiable) + corpus FP + precision across t, for each p⁻.
// Usage: TARGET=bank R=5 K=30 node --max-old-space-size=8192 benchmark/baseline.js english.txt <corpus>
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

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F;
  const pcache = new Map(); const partsOf = (w) => { let a = pcache.get(w); if (!a) { a = [...new Set(encodeWord(dict, w))]; pcache.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const ws = tokenizeStream(line.slice(seg.start, seg.end)).filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws);
  }
  const band = (d) => (d === 1 ? 2 : d <= 3 ? 1 : 0);
  const st = new Int32Array(DIM).fill(-1); let sp = 0;
  const feat = (words, t) => { sp++; const y = []; for (let d = 1; d <= D; d++) { const q = t - d; if (q < 0) break; const off = band(d) * F; for (const a of partsOf(words[q])) { const b = a + off; if (st[b] !== sp) { st[b] = sp; y.push(b); } } } return y; };

  // global doc-freq M
  const M = new Float64Array(DIM); let Ntot = 0;
  for (const words of sents) for (let t = 1; t < words.length; t++) { const f = feat(words, t); Ntot++; for (const b of f) M[b]++; }
  const rare = (b) => M[b] / Ntot <= DD;

  // p⁺ / P from the first NPOS positives (default: all)
  const POSall = []; for (const words of sents) for (let t = 1; t < words.length; t++) if (words[t] === TARGET) POSall.push(feat(words, t));
  const NPOS = Math.min(Number(process.env.NPOS || POSall.length), POSall.length);
  const POS = POSall.slice(0, NPOS);
  const P = new Uint8Array(DIM), pPlus = new Uint8Array(DIM); let ppn = 0;
  for (const cand of POS) for (const b of cand) { P[b] = 1; if (rare(b) && !pPlus[b]) { pPlus[b] = 1; ppn++; } }
  const gate1 = (cand) => { let any = false; for (const b of cand) if (rare(b)) { any = true; if (!pPlus[b]) return false; } return any; };

  // PASS A: peel p⁻ from gate-1 FP common∉P bits, floor K
  const cnp = new Map();
  for (const words of sents) for (let t = 1; t < words.length; t++) { if (words[t] === TARGET) continue; const cand = feat(words, t); if (!gate1(cand)) continue; for (const b of cand) if (!rare(b) && !P[b]) cnp.set(b, (cnp.get(b) || 0) + 1); }
  const peel = new Uint8Array(DIM); let peelN = 0, fullN = 0;
  for (const [b, c] of cnp) { fullN++; if (c >= K) { peel[b] = 1; peelN++; } }

  // per-candidate stats: posM, kept(|M_τ∧x|), negFull(common∉P), negPeel
  const stat = (cand) => { let posM = 0, kept = 0, negFull = 0, negPeel = 0; for (const b of cand) { if (P[b]) { if (rare(b)) { posM++; kept++; } } else { kept++; if (!rare(b)) { negFull++; if (peel[b]) negPeel++; } } } return { posM, kept, negFull, negPeel }; };
  const sc = (s, mode) => s.kept > 0 ? (s.posM - (mode === "full" ? s.negFull : mode === "peel" ? s.negPeel : 0)) / s.kept : null;

  const TS = [1.0, 0.75, 0.5, 0.25, 0.0];
  const modes = ["none", "full", "peel"];
  const fp = { none: TS.map(() => 0), full: TS.map(() => 0), peel: TS.map(() => 0) };
  let Nneg = 0;
  for (const words of sents) for (let t = 1; t < words.length; t++) { if (words[t] === TARGET) continue; Nneg++; const s = stat(feat(words, t)); for (const m of modes) { const v = sc(s, m); if (v === null) continue; for (let j = 0; j < TS.length; j++) if (v >= TS[j]) fp[m][j]++; } }
  // recall (in-sample POS) — same for all modes (positives have no ∉P bits)
  const rec = TS.map(() => 0); let clf = 0;
  for (const cand of POS) { const s = stat(cand); const v = sc(s, "peel"); if (v === null) continue; clf++; for (let j = 0; j < TS.length; j++) if (v >= TS[j]) rec[j]++; }

  const w = process.stdout;
  w.write(`# baseline 3F [L|L1∪L2|C] — ${path.basename(corpusArg)} | TARGET=${TARGET} δ=${DD} K=${K}\n`);
  w.write(`# |POS|=${POS.length} (classifiable ${clf}, recall ceiling ${(100 * clf / POS.length).toFixed(1)}%)  |P|=${P.reduce((a, b) => a + b, 0)}  p⁺=${ppn}  Nneg=${Nneg}\n`);
  w.write(`# p⁻: full common∉P = ${fullN} bits;  PEELED (≥${K} FPs) = ${peelN} bits (⊊ ¬P)\n\n`);
  const prec = (tp, f) => (100 * tp / (tp + f)).toFixed(2);
  for (const m of modes) {
    w.write(`── p⁻ = ${m === "none" ? "∅ (positive-only)" : m === "full" ? "full common∉P (" + fullN + ")" : "PEELED (" + peelN + ")"} ──\n   t       recall     FP        precision\n`);
    for (let j = 0; j < TS.length; j++) { const tp = rec[j], f = fp[m][j]; w.write(`   ${TS[j].toFixed(2).padStart(4)}    ${(100 * tp / POS.length).toFixed(1).padStart(5)}%   ${String(f).padStart(8)}   ${prec(tp, f).padStart(6)}%\n`); }
    w.write(`\n`);
  }
};

main();
