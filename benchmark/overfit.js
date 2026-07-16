"use strict";

// ============================================================================
// benchmark/overfit.js — the memorization reference + the honest ensemble spectrum.
// Three classifiers, all on rare-bit containment (3F [L|L1∪L2|C]), 250 train / 250 held:
//   SINGLE  : fire iff cand's rare bits ⊆ ⋁_train (rare∩P_union) — today's single gate
//   OVERFIT : fire iff cand's rare bits ⊆ SOME single train example y_i — 1 gate per positive
//   (multiGate's 27% train recall is the bug: it drops the residual instead of covering it)
// Report train recall, held recall, FP-rate for each — the memorize↔generalize spectrum.
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/overfit.js english.txt <corpus>
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
  const st = new Int32Array(DIM).fill(-1); let sp = 0;
  const feat = (words, t) => { sp++; const y = []; for (let d = 1; d <= D; d++) { const q = t - d; if (q < 0) break; const off = band(d) * F; for (const a of partsOf(words[q])) { const b = a + off; if (st[b] !== sp) { st[b] = sp; y.push(b); } } } return y; };

  const M = new Float64Array(DIM); let N = 0; const POS = []; const neg = []; let ni = 0;
  let negTot = 0; for (const words of sents) for (let t = 1; t < words.length; t++) if (words[t] !== TARGET) negTot++;
  const stride = Math.max(1, Math.floor(negTot / 120000));
  for (const words of sents) for (let t = 1; t < words.length; t++) { const f = feat(words, t); N++; for (const b of f) M[b]++; if (words[t] === TARGET) POS.push(f); else if (ni++ % stride === 0) neg.push(f); }
  const rare = (b) => M[b] / N <= DD;
  const rareOf = (f) => f.filter(rare);

  const half = POS.length >> 1; const train = POS.slice(0, half), held = POS.slice(half);
  // SINGLE: rare∩P_union
  const pUnion = new Uint8Array(DIM); for (const f of train) for (const b of f) if (rare(b)) pUnion[b] = 1;
  const single = (f) => { const rb = rareOf(f); if (!rb.length) return false; for (const b of rb) if (!pUnion[b]) return false; return true; };
  // OVERFIT: cand rare bits ⊆ some train example's rare bits (posting-list membership count)
  const trainRare = train.map(rareOf).map((r) => new Set(r));
  const post = new Map(); trainRare.forEach((s, i) => { for (const b of s) { let a = post.get(b); if (!a) post.set(b, a = []); a.push(i); } });
  const cnt = new Int32Array(train.length); const stamp2 = new Int32Array(train.length).fill(-1); let pid = 0;
  const overfit = (f) => { const rb = rareOf(f); if (!rb.length) return false; pid++; let best = 0; for (const b of rb) { const lst = post.get(b); if (!lst) continue; for (const i of lst) { if (stamp2[i] !== pid) { stamp2[i] = pid; cnt[i] = 0; } if (++cnt[i] === rb.length) return true; } } return false; };

  const rate = (SET, fn) => { let k = 0; for (const f of SET) if (fn(f)) k++; return 100 * k / SET.length; };
  const w = process.stdout;
  w.write(`# overfit reference — ${path.basename(corpusArg)} | TARGET=${TARGET} | train=${train.length} held=${held.length} neg=${neg.length}\n\n`);
  w.write(`  classifier   train-recall   held-recall   FP-rate\n`);
  for (const [nm, fn] of [["SINGLE ", single], ["OVERFIT", overfit]]) {
    w.write(`  ${nm}      ${rate(train, fn).toFixed(1).padStart(5)}%        ${rate(held, fn).toFixed(1).padStart(5)}%       ${rate(neg, fn).toFixed(3)}%\n`);
  }
};

main();
