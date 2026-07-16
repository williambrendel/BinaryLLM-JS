"use strict";

// benchmark/discrimMask2.js — discriminativeness mask on a TRAIN/HELD split (250/250).
// d(b), p⁺, p⁻, M_τ all built from TRAIN positives; recall measured on TRAIN and HELD; FP on corpus negatives.
// Compares the discriminativeness κ-band vs CONTRAST top-k (one-sided) vs GLOBAL freq mask.
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/discrimMask2.js english.txt <corpus>

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

  const M = new Float64Array(DIM); let Ntot = 0; const POS = []; const neg = []; let ni = 0, negTot = 0;
  for (const s of sents) for (let t = 1; t < s.length; t++) if (s[t] !== TARGET) negTot++;
  const stride = Math.max(1, Math.floor(negTot / 150000));
  for (const s of sents) for (let t = 1; t < s.length; t++) { const f = feat(s, t); Ntot++; for (const b of f) M[b]++; if (s[t] === TARGET) POS.push(f); else if (ni++ % stride === 0) neg.push(f); }
  const rare = (b) => M[b] / Ntot <= DD;
  const half = POS.length >> 1, train = POS.slice(0, half), held = POS.slice(half);
  const cntT = new Float64Array(DIM), cntAll = new Float64Array(DIM);
  for (const f of train) for (const b of f) cntT[b]++; for (const f of POS) for (const b of f) cntAll[b]++;
  const A = train.length, Abar = Ntot - POS.length, eps = 1 / Abar;
  const d = new Float64Array(DIM); for (let b = 0; b < DIM; b++) { if (M[b] === 0) { d[b] = 1; continue; } d[b] = (cntT[b] / A + eps) / ((M[b] - cntAll[b]) / Abar + eps); }
  const OR = new Uint8Array(DIM); const orBits = []; for (let b = 0; b < DIM; b++) if (cntT[b] > 0) { OR[b] = 1; orBits.push(b); } orBits.sort((a, b) => d[b] - d[a]);

  const rc = (arr, fn) => { let k = 0; for (const f of arr) if (fn(f)) k++; return k; };
  const ev = (fn) => { const fp = rc(neg, fn) / neg.length, FP = fp * negTot; const tr = rc(train, fn) / train.length, hr = rc(held, fn) / held.length; return { tr, hr, fp, tp: (tr * train.length) / (tr * train.length + FP) || 0, hp: (hr * held.length) / (hr * held.length + FP) || 0 }; };
  const w = process.stdout;
  const row = (name, fn) => { const e = ev(fn); w.write(`  ${name.padEnd(20)} ${(100 * e.tr).toFixed(1).padStart(5)}% ${(100 * e.tp).toFixed(2).padStart(6)}%   ${(100 * e.hr).toFixed(1).padStart(5)}% ${(100 * e.hp).toFixed(2).padStart(6)}%   ${(100 * e.fp).toFixed(3).padStart(6)}%\n`); };

  w.write(`# discrimMask2 — ${path.basename(corpusArg)} | TARGET=${TARGET} | train=${train.length} held=${held.length} |OR|=${orBits.length}\n\n`);
  w.write(`  rule                 train-rec train-prec  held-rec held-prec  FP%\n`);
  for (const k of [1.5, 2, 3, 5, 10]) row(`DISCRIM κ=${k}`, (f) => { let pp = 0, pm = 0; for (const b of f) { if (d[b] >= k) pp++; else if (d[b] <= 1 / k) pm++; } return pp - pm > 0; });
  const cornerV = (pset) => (f) => { let h = false; for (const b of f) { if (!OR[b]) return false; if (pset[b]) h = true; } return h; };
  const topk = (k) => { const p = new Uint8Array(DIM); for (let i = 0; i < Math.min(k, orBits.length); i++) p[orBits[i]] = 1; return p; };
  w.write(`  ----\n`);
  row(`GLOBAL freq δ=${DD}`, (f) => { let h = false; for (const b of f) { if (!OR[b]) return false; if (rare(b)) h = true; } return h; });
  for (const k of [100, 300, 500, 800]) row(`CONTRAST top-${k}`, cornerV(topk(k)));
};

main();
