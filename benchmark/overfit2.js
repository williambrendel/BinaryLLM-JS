"use strict";

// ============================================================================
// benchmark/overfit2.js — the OVERFIT baseline, 4 metrics.
// Each TRAIN positive y_i is its own gate: p⁺_i=y_i, M_i=¬(M_glob∧y_i); at the corner the gate
// fires iff candidate ⊆ y_i (the adapted mask's common∉y_i bits are the implicit p⁻). Ensemble = OR:
//   x is positive iff x ⊆ some train y_i.
// Expected: ~perfect on training (each pos ⊆ itself; negatives rarely ⊆ any single example),
//   catastrophic held-out (a new positive is ⊆ no training example — near-disjoint contexts).
// Reports TRAIN precision/recall and HELD precision/recall.
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/overfit2.js english.txt <corpus>
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

  // 50/50 sentence split; global M over all
  const M = new Float64Array(DIM); let Ntot = 0;
  const half = sents.length >> 1;
  const set = { train: { pos: [], negN: 0 }, held: { pos: [], negN: 0 } };
  const negSamp = { train: [], held: [] }; const negStride = 60;
  for (let si = 0; si < sents.length; si++) { const s = sents[si], part = si < half ? "train" : "held"; for (let t = 1; t < s.length; t++) { const f = feat(s, t); Ntot++; for (const b of f) M[b]++; if (s[t] === TARGET) set[part].pos.push(f); else { set[part].negN++; if (set[part].negN % negStride === 0) negSamp[part].push(f); } } }
  const rare = (b) => M[b] / Ntot <= DD;

  // gates = train positives; posting on rare bits for the ⊆ narrowing
  const gates = set.train.pos.map((f) => new Set(f));
  const rareOf = (f) => f.filter(rare);
  const post = new Map(); gates.forEach((g, i) => { for (const b of g) if (rare(b)) { let a = post.get(b); if (!a) post.set(b, a = []); a.push(i); } });
  const fires = (f) => {                    // ⊆ some train example
    const rb = rareOf(f); if (!rb.length) return false;
    let bestBit = rb[0], bestLen = (post.get(rb[0]) || []).length; for (const b of rb) { const l = (post.get(b) || []).length; if (l < bestLen) { bestLen = l; bestBit = b; } }
    const cand = post.get(bestBit); if (!cand) return false;
    outer: for (const i of cand) { const g = gates[i]; for (const b of f) if (!g.has(b)) continue outer; return true; }
    return false;
  };

  const rateCount = (arr) => { let k = 0; for (const f of arr) if (fires(f)) k++; return k; };
  const w = process.stdout;
  const evalPart = (part) => {
    const P = set[part].pos, tp = rateCount(P), recall = tp / P.length;
    const nf = rateCount(negSamp[part]), fireRate = nf / negSamp[part].length, FP = fireRate * set[part].negN;
    const TP = recall * P.length;
    return { recall, prec: TP / (TP + FP), fp: FP, np: P.length, nn: set[part].negN };
  };
  w.write(`# overfit2 (OR ensemble, ${gates.length} per-example gates) — ${path.basename(corpusArg)} | TARGET=${TARGET}\n`);
  w.write(`# train pos=${set.train.pos.length} neg=${set.train.negN} | held pos=${set.held.pos.length} neg=${set.held.negN}\n\n`);
  const tr = evalPart("train"), he = evalPart("held");
  w.write(`             recall     precision      (TP / FP)\n`);
  w.write(`  TRAINING   ${(100 * tr.recall).toFixed(1).padStart(5)}%     ${(100 * tr.prec).toFixed(2).padStart(5)}%       ${(tr.recall * tr.np) | 0} / ${tr.fp | 0}\n`);
  w.write(`  HELD-OUT   ${(100 * he.recall).toFixed(1).padStart(5)}%     ${(100 * he.prec).toFixed(2).padStart(5)}%       ${(he.recall * he.np) | 0} / ${he.fp | 0}\n`);
  // diagnostic: are the firing FPs short? (subset ≠ equality)
  let nAll = 0, bAll = 0, nF = 0, bF = 0, rF = 0; for (const f of negSamp.train) { nAll++; bAll += f.length; if (fires(f)) { nF++; bF += f.length; rF += rareOf(f).length; } }
  w.write(`\n  FP diagnostic (train neg sample): all negs mean |x|=${(bAll / nAll).toFixed(1)}  |  FIRING negs mean |x|=${(bF / (nF || 1)).toFixed(1)}, mean rare bits=${(rF / (nF || 1)).toFixed(1)}\n`);
  w.write(`  → they fire because a short signature is ⊆ some positive, not because it equals one.\n`);
};

main();
