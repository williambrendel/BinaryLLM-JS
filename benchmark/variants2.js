"use strict";

// benchmark/variants2.js — 4 metrics (train/held precision+recall) for p⁺ selection rules, single gate, 50/50 split.
//   ADAPTED  : global common mask — p⁺=rare∩OR, M_τ=¬(common∧OR)   (today's baseline)
//   UNIQUE   : p⁺={count_A==1},                 M_τ=¬(OR∖p⁺)
//   CONTRAST : p⁺=top-k by freq_A/freq_Ā,       M_τ=¬(OR∖p⁺),  k swept up to |OR|
// Corner gate (coverage=1): fires(x) ⟺ x⊆OR ∧ x∩p⁺≠∅ (for the variant mask) / x has a rare∩OR bit (adapted).
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/variants2.js english.txt <corpus>

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
const R = Number(process.env.R || 5), D = R + 1, DD = Number(process.env.DELTA || 0.001);

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

  const half = sents.length >> 1; const M = new Float64Array(DIM); let Ntot = 0;
  const set = { train: { pos: [], negN: 0, negS: [] }, held: { pos: [], negN: 0, negS: [] } }; const nstr = 50;
  for (let si = 0; si < sents.length; si++) { const s = sents[si], part = si < half ? "train" : "held"; for (let t = 1; t < s.length; t++) { const f = feat(s, t); Ntot++; for (const b of f) M[b]++; if (s[t] === TARGET) set[part].pos.push(f); else { set[part].negN++; if (set[part].negN % nstr === 0) set[part].negS.push(f); } } }
  const rare = (b) => M[b] / Ntot <= DD;
  const cntA = new Float64Array(DIM), OR = new Uint8Array(DIM); for (const f of set.train.pos) for (const b of f) cntA[b]++;
  let orN = 0; for (let b = 0; b < DIM; b++) if (cntA[b] > 0) { OR[b] = 1; orN++; }
  const A = set.train.pos.length, Abar = Ntot - (A + set.held.pos.length);
  const contrast = (b) => ((cntA[b] + 0.5) / (A + 1)) / ((M[b] - cntA[b] + 0.5) / (Abar + 1));
  const orBits = []; for (let b = 0; b < DIM; b++) if (OR[b]) orBits.push(b); orBits.sort((a, b) => contrast(b) - contrast(a));

  const metrics = (fires) => {
    const rc = (arr) => { let k = 0; for (const f of arr) if (fires(f)) k++; return k; };
    const out = {}; for (const part of ["train", "held"]) { const P = set[part], tp = rc(P.pos), fr = rc(P.negS) / P.negS.length, FP = fr * P.negN; out[part] = { rec: tp / P.pos.length, prec: tp + FP > 0 ? tp / (tp + FP) : 0, fp: fr }; }
    return out;
  };
  const cornerVariant = (pset) => (f) => { let hasP = false; for (const b of f) { if (!OR[b]) return false; if (pset[b]) hasP = true; } return hasP; };   // x⊆OR ∧ x∩p⁺≠∅
  const cornerAdapted = (f) => { let hasR = false; for (const b of f) { if (!OR[b]) return false; if (rare(b)) hasR = true; } return hasR; };   // x⊆OR ∧ rare bit
  const topk = (k) => { const p = new Uint8Array(DIM); for (let i = 0; i < Math.min(k, orBits.length); i++) p[orBits[i]] = 1; return p; };
  const uniq = () => { const p = new Uint8Array(DIM); let n = 0; for (let b = 0; b < DIM; b++) if (cntA[b] === 1) { p[b] = 1; n++; } return { p, n }; };

  const w = process.stdout;
  w.write(`# variants2 — ${path.basename(corpusArg)} | TARGET=${TARGET} | train pos=${A} neg=${set.train.negN} | held pos=${set.held.pos.length} neg=${set.held.negN} | |OR|=${orN}\n\n`);
  w.write(`  rule            |p⁺|   train-rec  train-prec  train-FP%   held-rec  held-prec  held-FP%\n`);
  const row = (name, n, fires) => { const m = metrics(fires); w.write(`  ${name.padEnd(15)} ${String(n).padStart(5)}   ${(100 * m.train.rec).toFixed(1).padStart(5)}%     ${(100 * m.train.prec).toFixed(2).padStart(6)}%    ${(100 * m.train.fp).toFixed(3).padStart(6)}%    ${(100 * m.held.rec).toFixed(1).padStart(5)}%    ${(100 * m.held.prec).toFixed(2).padStart(6)}%   ${(100 * m.held.fp).toFixed(3).padStart(6)}%\n`); };
  row("ADAPTED", orN, cornerAdapted);
  const u = uniq(); row("UNIQUE", u.n, cornerVariant(u.p));
  for (const k of [100, 300, 500, 700, orN]) row(`CONTRAST k=${k}`, Math.min(k, orN), cornerVariant(topk(k)));
};

main();
