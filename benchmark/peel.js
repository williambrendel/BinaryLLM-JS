"use strict";

// ============================================================================
// benchmark/peel.js — greedy set-cover construction of the negative gate g⁻.
// Gate-1 = perfect coverage of rare-word PARTS in p⁺ (from build positives). Collect its false
// positives. Then PEEL: repeatedly pick the most common rare WORD among surviving FPs that is NOT
// in the positive vocab V⁺, add it to g⁻, remove (peel) every FP containing it, repeat. Each bit is
// ∉ V⁺ ⇒ recall-safe on train; held-out recall cost measured as g⁻ grows.
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/peel.js english.txt <corpus>
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
  const sents = []; const cnt = new Map(); let Ntok = 0;
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const ws = tokenizeStream(line.slice(seg.start, seg.end)).filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) { sents.push(ws); for (const x of ws) { cnt.set(x, (cnt.get(x) || 0) + 1); Ntok++; } }
  }
  const wf = (w) => (cnt.get(w) || 0) / Ntok;
  const band = (d) => (d === 1 ? 2 : d <= 3 ? 1 : 0);
  const feat = (words, t) => { const e = []; for (let d = 1; d <= D; d++) { const q = t - d; if (q < 0) break; const w = words[q], off = band(d) * F; e.push({ wf: wf(w), ids: partsOf(w).map((a) => a + off), w }); } return e; };
  const rareWords = (cand) => { const s = new Set(); for (const e of cand) if (e.wf <= DD) s.add(e.w); return s; };

  const POS = [];
  for (const words of sents) for (let t = 1; t < words.length; t++) if (words[t] === TARGET) POS.push(feat(words, t));
  const Pb = POS.filter((_, i) => i % 2 === 0), Pt = POS.filter((_, i) => i % 2 === 1);

  const pPlus = new Uint8Array(DIM); const Vplus = new Set();
  for (const cand of Pb) for (const e of cand) if (e.wf <= DD) { Vplus.add(e.w); for (const id of e.ids) pPlus[id] = 1; }
  const gate1 = (cand) => { let any = false; for (const e of cand) if (e.wf <= DD) { any = true; for (const id of e.ids) if (!pPlus[id]) return false; } return any; };

  // collect gate-1 false positives; store each FP's rare words ∉ V⁺
  let fp1 = 0, unpeelable = 0; const fpNeg = []; const unpEx = [];
  for (const words of sents) for (let t = 1; t < words.length; t++) {
    if (words[t] === TARGET) continue;
    const cand = feat(words, t);
    if (!gate1(cand)) continue; fp1++;
    const neg = []; for (const w of rareWords(cand)) if (!Vplus.has(w)) neg.push(w);
    if (neg.length === 0) { unpeelable++; if (unpEx.length < 8 && t % 5 === 0) unpEx.push({ ctx: words.slice(Math.max(0, t - D), t).join(" "), tgt: words[t], rw: [...rareWords(cand)] }); }
    else fpNeg.push(new Set(neg));
  }
  process.stderr.write(`\n── sample UNPEELABLE FPs (all rare words ∈ V⁺, i.e. each DOES precede bank somewhere) ──\n`);
  for (const e of unpEx) process.stderr.write(`  {${e.ctx}} → [${e.tgt}]   all-in-V⁺ rare words: ${e.rw.join(" ")}\n`);

  // Pt held-out gate-1 passers + their rare words (for recall-cost tracking)
  const ptPass = []; for (const c of Pt) if (gate1(c)) ptPass.push(rareWords(c));
  const recallAt = (gset) => { let k = 0; for (const rw of ptPass) { let hit = false; for (const w of rw) if (gset.has(w)) { hit = true; break; } if (!hit) k++; } return k / Pt.length; };

  // GREEDY PEEL
  const w = process.stdout;
  w.write(`# peel g⁻ [L|L1∪L2|C] — ${path.basename(corpusArg)} | TARGET=${TARGET} δ=${DD} | V⁺=${Vplus.size} words\n`);
  w.write(`# gate-1 false positives=${fp1}  peelable=${fpNeg.length}  UNPEELABLE (all words ∈ V⁺)=${unpeelable} (${(100 * unpeelable / fp1).toFixed(1)}%)\n`);
  w.write(`# gate-1 held-out recall (Pt=${Pt.length}) = ${(100 * recallAt(new Set())).toFixed(1)}%\n\n`);

  const alive = fpNeg.map(() => true); const gminus = new Set(); const order = [];
  let peeled = 0; const checkpoints = [1, 3, 5, 10, 25, 50, 100, 200, 400, 800];
  w.write(`   g⁻ size   FP peeled (cum)   FP remaining   held-out recall   last word (coverage)\n`);
  for (let iter = 1; iter <= 1000; iter++) {
    const count = new Map();
    for (let i = 0; i < fpNeg.length; i++) if (alive[i]) for (const wd of fpNeg[i]) count.set(wd, (count.get(wd) || 0) + 1);
    if (count.size === 0) break;
    let best = null, bc = 0; for (const [wd, c] of count) if (c > bc) { bc = c; best = wd; }
    gminus.add(best); order.push([best, bc]);
    for (let i = 0; i < fpNeg.length; i++) if (alive[i] && fpNeg[i].has(best)) { alive[i] = false; peeled++; }
    if (checkpoints.includes(iter)) {
      const remain = fp1 - peeled;
      w.write(`   ${String(iter).padStart(7)}   ${String(peeled).padStart(13)}   ${String(remain).padStart(12)}   ${(100 * recallAt(gminus)).toFixed(1).padStart(13)}%   ${best} (${bc})\n`);
    }
    if (bc <= 1) break;   // stop when the best remaining bit covers ≤1 FP (diminishing)
  }
  const remain = fp1 - peeled;
  w.write(`\n  final: g⁻=${gminus.size} words peel ${peeled}/${fpNeg.length} peelable FPs; ${remain} FP remain (${unpeelable} unpeelable + ${remain - unpeelable} low-coverage)\n`);
  w.write(`  held-out recall ${(100 * recallAt(new Set())).toFixed(1)}% → ${(100 * recallAt(gminus)).toFixed(1)}%   FP ${fp1} → ${remain} (${(100 * (fp1 - remain) / fp1).toFixed(1)}% cut)\n`);
  w.write(`\n  first 20 negative words (greedy order, coverage): ${order.slice(0, 20).map(([x, c]) => `${x}(${c})`).join(" ")}\n`);
};

main();
