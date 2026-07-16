"use strict";

// ============================================================================
// benchmark/gate2.js — the NEGATIVE gate (spec's g⁻).
// Gate-1 (positive): perfect coverage of the candidate's rare-word PARTS in p⁺ (generalizes via
//   morphemes → high recall, but lets words whose parts are in p⁺ but that never precede bank slip in).
// Gate-2 (negative): reject a gate-1 passer if it contains a rare WORD not in the positive vocab V⁺.
//   V⁺ is built from positives only ⇒ recall-safe on train; held-out recall cost measured explicitly.
// Reports FP + recall for gate-1 alone vs gate-1 ∧ gate-2, all held-out (p⁺,V⁺ from build half).
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/gate2.js english.txt <corpus>
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
  const dict = loadDict(dictArg); const F = dict.size(), NB = 3, DIM = NB * F;
  const pc = new Map(); const partsOf = (w) => { let a = pc.get(w); if (!a) { a = [...new Set(encodeWord(dict, w))]; pc.set(w, a); } return a; };
  const sents = []; const cnt = new Map(); let Ntok = 0;
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const ws = tokenizeStream(line.slice(seg.start, seg.end)).filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) { sents.push(ws); for (const x of ws) { cnt.set(x, (cnt.get(x) || 0) + 1); Ntok++; } }
  }
  const wf = (w) => (cnt.get(w) || 0) / Ntok;
  const band = (d) => (d === 1 ? 2 : d <= 3 ? 1 : 0);   // [L | L1∪L2 | C]
  const feat = (words, t) => { const e = []; for (let d = 1; d <= D; d++) { const q = t - d; if (q < 0) break; const w = words[q], off = band(d) * F; e.push({ wf: wf(w), ids: partsOf(w).map((a) => a + off), w }); } return e; };

  const POS = [];
  for (const words of sents) for (let t = 1; t < words.length; t++) if (words[t] === TARGET) POS.push(feat(words, t));
  const Pb = POS.filter((_, i) => i % 2 === 0), Pt = POS.filter((_, i) => i % 2 === 1);

  // build p⁺ (parts) and V⁺ (rare words) from the BUILD half only
  const pPlus = new Uint8Array(DIM); let ppn = 0; const Vplus = new Set();
  for (const cand of Pb) for (const e of cand) if (e.wf <= DD) { Vplus.add(e.w); for (const id of e.ids) if (!pPlus[id]) { pPlus[id] = 1; ppn++; } }

  const stamp = new Int32Array(DIM).fill(-1); let sp = 0;
  const gate1 = (cand) => { sp++; let any = false; for (const e of cand) if (e.wf <= DD) for (const id of e.ids) { any = true; if (!pPlus[id]) return false; if (stamp[id] !== sp) stamp[id] = sp; } return any; };   // all rare parts ∈ p⁺, ≥1 rare part
  const gate2ok = (cand) => { for (const e of cand) if (e.wf <= DD && !Vplus.has(e.w)) return false; return true; };   // every rare word ∈ V⁺

  // held-out recall on Pt
  const rec = (SET, useG2) => { let k = 0; for (const c of SET) if (gate1(c) && (!useG2 || gate2ok(c))) k++; return k / SET.length; };
  const r1 = rec(Pt, false), r2 = rec(Pt, true);

  // FP over corpus (non-bank positions)
  let N = 0, fp1 = 0, fp2 = 0, rej = [];
  for (const words of sents) for (let t = 1; t < words.length; t++) {
    if (words[t] === TARGET) continue; N++;
    const cand = feat(words, t);
    if (gate1(cand)) { fp1++; if (gate2ok(cand)) fp2++; else if (rej.length < 10 && t % 5 === 0) { const nw = cand.filter((e) => e.wf <= DD && !Vplus.has(e.w)).map((e) => e.w); rej.push({ ctx: words.slice(Math.max(0, t - D), t).join(" "), tgt: words[t], nw }); } }
  }

  const w = process.stdout;
  w.write(`# gate2 [L|L1∪L2|C] — ${path.basename(corpusArg)} | TARGET=${TARGET} δ=${DD} | p⁺=${ppn} parts, V⁺=${Vplus.size} words (from ${Pb.length} build pos)\n\n`);
  w.write(`                       held-out recall (Pt=${Pt.length})   FALSE POSITIVES (of ${N})   precision*\n`);
  w.write(`  gate-1 only          ${(100 * r1).toFixed(1)}%                        ${fp1}                ${(100 * (r1 * Pt.length) / (r1 * Pt.length + fp1)).toFixed(3)}%\n`);
  w.write(`  gate-1 ∧ gate-2      ${(100 * r2).toFixed(1)}%                        ${fp2}                ${(100 * (r2 * Pt.length) / (r2 * Pt.length + fp2)).toFixed(3)}%\n`);
  w.write(`\n  Δ: false positives ${fp1}→${fp2} (${(100 * (fp1 - fp2) / fp1).toFixed(1)}% cut)   recall ${(100 * r1).toFixed(1)}%→${(100 * r2).toFixed(1)}% (Δ ${(100 * (r1 - r2)).toFixed(1)} pts)\n`);
  w.write(`\n── sample gate-2 rejections (fired gate-1, killed by a non-positive rare word) ──\n`);
  for (const e of rej) w.write(`  {${e.ctx}} → [${e.tgt}]   negative word(s): ${e.nw.join(" ")}\n`);
};

main();
