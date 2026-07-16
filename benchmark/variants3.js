"use strict";

// benchmark/variants3.js — ONE consistent setup (position-split, |OR|=970 like variants.js) with the 4 metrics
// + FP%, k swept to the full OR — to verify k=970 reproduces the OR corner (~2.068%) and k=800 (~0.376%).
// train = first-half bank contexts, held = second-half; negatives = one global sample.
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/variants3.js english.txt <corpus>

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
  const NPOS = Math.min(Number(process.env.NPOS || POS.length), POS.length);
  const train = POS.slice(0, NPOS), held = POS.slice(NPOS);
  const cntA = new Float64Array(DIM), OR = new Uint8Array(DIM); for (const f of train) for (const b of f) cntA[b]++;
  let orN = 0; for (let b = 0; b < DIM; b++) if (cntA[b] > 0) { OR[b] = 1; orN++; }
  const A = train.length, Abar = Ntot - POS.length;
  const contrast = (b) => ((cntA[b] + 0.5) / (A + 1)) / ((M[b] - cntA[b] + 0.5) / (Abar + 1));
  const orBits = []; for (let b = 0; b < DIM; b++) if (OR[b]) orBits.push(b); orBits.sort((a, b) => contrast(b) - contrast(a));

  const rc = (arr, fires) => { let k = 0; for (const f of arr) if (fires(f)) k++; return k; };
  const metrics = (fires) => { const fp = rc(neg, fires) / neg.length, FP = fp * negTot; const trec = rc(train, fires) / train.length, hrec = held.length ? rc(held, fires) / held.length : NaN; const tTP = trec * train.length, hTP = hrec * held.length; return { trec, tprec: tTP + FP > 0 ? tTP / (tTP + FP) : 0, hrec, hprec: hTP + FP > 0 ? hTP / (hTP + FP) : 0, fp }; };
  const cornerVariant = (pset) => (f) => { let hasP = false; for (const b of f) { if (!OR[b]) return false; if (pset[b]) hasP = true; } return hasP; };
  const cornerAdapted = (f) => { let hasR = false; for (const b of f) { if (!OR[b]) return false; if (rare(b)) hasR = true; } return hasR; };
  const topk = (k) => { const p = new Uint8Array(DIM); for (let i = 0; i < Math.min(k, orBits.length); i++) p[orBits[i]] = 1; return p; };

  const w = process.stdout;
  w.write(`# variants3 (position-split) — ${path.basename(corpusArg)} | TARGET=${TARGET} | train=${A} held=${held.length} neg(all)=${negTot} sample=${neg.length} | |OR|=${orN}\n\n`);
  w.write(`  rule            |p⁺|   train-rec  train-prec   held-rec  held-prec   corner-FP%\n`);
  const row = (name, n, fires) => { const m = metrics(fires); w.write(`  ${name.padEnd(15)} ${String(n).padStart(5)}   ${(100 * m.trec).toFixed(1).padStart(5)}%     ${(100 * m.tprec).toFixed(2).padStart(6)}%     ${(100 * m.hrec).toFixed(1).padStart(5)}%    ${(100 * m.hprec).toFixed(2).padStart(6)}%     ${(100 * m.fp).toFixed(3).padStart(6)}%\n`); };
  row("ADAPTED", orN, cornerAdapted);
  for (const k of [100, 300, 500, 800, orN]) row(`CONTRAST k=${k}`, Math.min(k, orN), cornerVariant(topk(k)));
};

main();
