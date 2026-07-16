"use strict";

// benchmark/overfit3.js — overfit ensemble WITH M_τ (subset firing) vs WITHOUT M_τ (exact-match firing).
//   SUBSET  (with M_τ, t=1 corner):  fire iff x ⊆ y_i   — short contexts nest inside a positive → leak
//   EXACT   (no M_τ, full signature): fire iff x == y_i  — pure template memorization
// Reports train/held precision+recall for both. Expectation: EXACT → ~perfect train precision, ~0 held recall.
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/overfit3.js english.txt <corpus>

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
  const key = (f) => f.slice().sort((a, b) => a - b).join(",");

  const M = new Float64Array(DIM); let Ntot = 0; const half = sents.length >> 1;
  const set = { train: { pos: [], negN: 0, negS: [] }, held: { pos: [], negN: 0, negS: [] } }; const negStride = 60;
  for (let si = 0; si < sents.length; si++) { const s = sents[si], part = si < half ? "train" : "held"; for (let t = 1; t < s.length; t++) { const f = feat(s, t); Ntot++; for (const b of f) M[b]++; if (s[t] === TARGET) set[part].pos.push(f); else { set[part].negN++; if (set[part].negN % negStride === 0) set[part].negS.push(f); } } }
  const rare = (b) => M[b] / Ntot <= DD;

  // gates from train positives
  const gates = set.train.pos.map((f) => new Set(f));
  const post = new Map(); gates.forEach((g, i) => { for (const b of g) if (rare(b)) { let a = post.get(b); if (!a) post.set(b, a = []); a.push(i); } });
  const firesSubset = (f) => { const rb = f.filter(rare); if (!rb.length) return false; let bb = rb[0], bl = (post.get(rb[0]) || []).length; for (const b of rb) { const l = (post.get(b) || []).length; if (l < bl) { bl = l; bb = b; } } const cand = post.get(bb); if (!cand) return false; outer: for (const i of cand) { const g = gates[i]; for (const b of f) if (!g.has(b)) continue outer; return true; } return false; };
  const keys = new Set(set.train.pos.map(key));
  const firesExact = (f) => keys.has(key(f));

  const evalPart = (part, fn) => {
    const P = set[part].pos; let tp = 0; for (const f of P) if (fn(f)) tp++;
    const recall = tp / P.length; let nf = 0; for (const f of set[part].negS) if (fn(f)) nf++;
    const FP = (nf / set[part].negS.length) * set[part].negN, TP = recall * P.length;
    return { recall, prec: TP + FP > 0 ? TP / (TP + FP) : 0, TP: TP | 0, FP: FP | 0 };
  };
  const w = process.stdout;
  w.write(`# overfit3 — ${path.basename(corpusArg)} | TARGET=${TARGET} | train pos=${set.train.pos.length} neg=${set.train.negN} | held pos=${set.held.pos.length} neg=${set.held.negN}\n\n`);
  const fmt = (e) => `${(100 * e.recall).toFixed(1).padStart(5)}% / ${(100 * e.prec).toFixed(2).padStart(6)}%  (${e.TP}/${e.FP})`;
  w.write(`                    SUBSET  x⊆y_i  (with M_τ)          EXACT  x==y_i  (no M_τ)\n`);
  w.write(`   recall/precision (TP/FP)\n`);
  w.write(`  TRAINING          ${fmt(evalPart("train", firesSubset))}        ${fmt(evalPart("train", firesExact))}\n`);
  w.write(`  HELD-OUT          ${fmt(evalPart("held", firesSubset))}        ${fmt(evalPart("held", firesExact))}\n`);
};

main();
