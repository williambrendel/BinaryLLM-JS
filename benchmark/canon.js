"use strict";

// benchmark/canon.js — CANONICAL harness. ONE d(b), ONE 50/50 split (pos AND neg), 6 metrics, deterministic.
//   d(b) = freq_A/freq_Ā, Haldane-smoothed, from TRAIN only (train positives vs train negatives — no leakage).
//   Metrics: train-rec, train-prec, held-rec, held-prec, FP%-train, FP%-held.
//   Masks/gates: GLOBAL(corner) | CONTRAST top-k(corner) | DISCRIM κ-band(soft net-positive gate).
// Deterministic: fixed-stride negative sampling, no RNG. Re-runs are identical.
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/canon.js english.txt <corpus>

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

  const half = sents.length >> 1;
  const M = new Float64Array(DIM), cntTP = new Float64Array(DIM), cntTN = new Float64Array(DIM); let Ntot = 0;
  const trainPos = [], heldPos = [], trainNegS = [], heldNegS = []; let tnN = 0, hnN = 0; const NS = 40;
  for (let si = 0; si < sents.length; si++) {
    const s = sents[si], isTrain = si < half;
    for (let t = 1; t < s.length; t++) {
      const f = feat(s, t); Ntot++; for (const b of f) M[b]++;
      if (s[t] === TARGET) { if (isTrain) { trainPos.push(f); for (const b of f) cntTP[b]++; } else heldPos.push(f); }
      else if (isTrain) { tnN++; for (const b of f) cntTN[b]++; if (tnN % NS === 0) trainNegS.push(f); }
      else { hnN++; if (hnN % NS === 0) heldNegS.push(f); }
    }
  }
  const rare = (b) => M[b] / Ntot <= DD;
  const tpN = trainPos.length;
  // additive-ε on the RATES so an unseen bit (0 in both) gets d=1 (masked), NOT (tnN/tpN)≈huge (Haldane bug)
  const d = new Float64Array(DIM); const eps = 1 / tnN;
  for (let b = 0; b < DIM; b++) { if (M[b] === 0) { d[b] = 1; continue; } d[b] = (cntTP[b] / tpN + eps) / (cntTN[b] / tnN + eps); }
  const OR = new Uint8Array(DIM); const orBits = []; for (let b = 0; b < DIM; b++) if (cntTP[b] > 0) { OR[b] = 1; orBits.push(b); } orBits.sort((a, b) => d[b] - d[a]);

  // 6 metrics for a fires() predicate
  const rc = (arr, fn) => { let k = 0; for (const f of arr) if (fn(f)) k++; return k; };
  const M6 = (fn) => {
    const trec = rc(trainPos, fn) / trainPos.length, hrec = rc(heldPos, fn) / heldPos.length;
    const fpt = rc(trainNegS, fn) / trainNegS.length, fph = rc(heldNegS, fn) / heldNegS.length;
    const tTP = trec * trainPos.length, tFP = fpt * tnN, hTP = hrec * heldPos.length, hFP = fph * hnN;
    const tacc = (tTP + (tnN - tFP)) / (trainPos.length + tnN), hacc = (hTP + (hnN - hFP)) / (heldPos.length + hnN);
    return { trec, tprec: tTP + tFP > 0 ? tTP / (tTP + tFP) : 0, hrec, hprec: hTP + hFP > 0 ? hTP / (hTP + hFP) : 0, fpt, fph, tacc, hacc, hTP: Math.round(hTP), hFP: Math.round(hFP) };
  };
  const cornerV = (pset) => (f) => { let h = false; for (const b of f) { if (!OR[b]) return false; if (pset[b]) h = true; } return h; };
  const topk = (k) => { const p = new Uint8Array(DIM); for (let i = 0; i < Math.min(k, orBits.length); i++) p[orBits[i]] = 1; return p; };
  const globalFn = (f) => { let h = false; for (const b of f) { if (!OR[b]) return false; if (rare(b)) h = true; } return h; };
  const discrim = (k) => (f) => { let pp = 0, pm = 0; for (const b of f) { if (d[b] >= k) pp++; else if (d[b] <= 1 / k) pm++; } return pp - pm > 0; };

  const countOcc = (pred) => { let n = 0; for (let b = 0; b < DIM; b++) if (M[b] > 0 && pred(b)) n++; return n; };
  const occ = countOcc(() => true);
  const w = process.stdout;
  w.write(`# canon — ${path.basename(corpusArg)} | TARGET=${TARGET} δ=${DD} | trainPos=${trainPos.length} heldPos=${heldPos.length} trainNeg=${tnN} heldNeg=${hnN} | |OR|=${orBits.length} occ-bits=${occ}\n\n`);
  w.write(`  config              |p⁺|   |M_τ|  | train-rec train-prec train-acc FP-tr% | held-rec held-prec held-acc FP-held%\n`);
  const row = (name, fn, ppN, mtN, pmN) => { const m = M6(fn); w.write(`  ${name.padEnd(19)} ${String(ppN).padStart(5)} ${String(mtN).padStart(6)} | ${(100 * m.trec).toFixed(1).padStart(5)}%   ${(100 * m.tprec).toFixed(2).padStart(6)}%   ${(100 * m.tacc).toFixed(3).padStart(6)}%  ${(100 * m.fpt).toFixed(3).padStart(6)}% | ${(100 * m.hrec).toFixed(1).padStart(5)}%   ${(100 * m.hprec).toFixed(2).padStart(6)}%   ${(100 * m.hacc).toFixed(3).padStart(6)}%  ${(100 * m.fph).toFixed(3).padStart(6)}%${pmN !== undefined ? `  |p⁻|=${pmN}` : ""}\n`); };
  row("GLOBAL (corner)", globalFn, countOcc((b) => rare(b) && OR[b]), countOcc((b) => rare(b) || !OR[b]));
  for (const k of [300, 800]) { const p = topk(k); row(`CONTRAST top-${k}`, cornerV(p), countOcc((b) => p[b] === 1), countOcc((b) => p[b] === 1 || !OR[b])); }
  for (const k of [1.5, 2, 2.5, 3, 5, 10]) row(`DISCRIM κ=${k} (soft)`, discrim(k), countOcc((b) => d[b] >= k), countOcc((b) => d[b] >= k || d[b] <= 1 / k), countOcc((b) => d[b] <= 1 / k));
};

main();
