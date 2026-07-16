"use strict";

// benchmark/illustrate.js — one bank positive, its M_τ-masked signature, and the negatives that
// mask to the EXACT SAME signature. Shows why the gate fires on them (identical masked signature).
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/illustrate.js english.txt <corpus>

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { encodeWord } from "../src/core/signatures/wordEncoder.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";
import { kindToString } from "../src/core/parts/kind.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGET = process.env.TARGET || "bank";
const R = Number(process.env.R || 5), D = R + 1, DD = Number(process.env.DELTA || 0.001);

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F, parts = dict.allParts();
  const pc = new Map(); const partsOf = (w) => { let a = pc.get(w); if (!a) { a = [...new Set(encodeWord(dict, w))]; pc.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const ws = tokenizeStream(line.slice(seg.start, seg.end)).filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws);
  }
  const band = (d) => (d === 1 ? 2 : d <= 3 ? 1 : 0);
  const stmp = new Int32Array(DIM).fill(-1); let sp = 0;
  const feat = (s, t) => { sp++; const y = []; for (let d = 1; d <= D; d++) { const q = t - d; if (q < 0) break; const off = band(d) * F; for (const a of partsOf(s[q])) { const b = a + off; if (stmp[b] !== sp) { stmp[b] = sp; y.push(b); } } } return y; };
  const ctx = (s, t) => s.slice(Math.max(0, t - D), t).join(" ");
  const lab = (b) => parts[b % F].value + "@" + ["L", "L2", "C"][Math.floor(b / F)];

  const M = new Float64Array(DIM); let N = 0; const posList = [];
  for (let si = 0; si < sents.length; si++) { const s = sents[si]; for (let t = 1; t < s.length; t++) { const f = feat(s, t); N++; for (const b of f) M[b]++; if (s[t] === TARGET) posList.push({ si, t, f }); } }
  const rare = (b) => M[b] / N <= DD;

  // pick a bank positive with a FULL, mostly-common context — its subset FPs are the illustrative case
  let pick = null; for (const p of posList) { const rb = p.f.filter(rare); const wc = sents[p.si].slice(Math.max(0, p.t - D), p.t).length; if (wc >= 5 && rb.length >= 2 && rb.length <= 5) { if (!pick || wc > pick.wc) pick = { ...p, rb, wc, yset: new Set(p.f) }; } }
  const w = process.stdout;
  w.write(`# illustrate — ${path.basename(corpusArg)} | TARGET=${TARGET}\n\n`);
  w.write(`POSITIVE (target = bank):\n  {${ctx(sents[pick.si], pick.t)}} → [bank]\n`);
  w.write(`  |y_i|=${pick.f.length} bits;  M_τ = ¬(common ∧ y_i) masks the ${pick.f.length - pick.rb.length} common bits, keeps ${pick.rb.length} rare:\n`);
  w.write(`  ┌ M_τ-masked signature p⁺ = { ${pick.rb.map(lab).join(", ")} }\n\n`);

  // the gate fires at t=1 on x ⊆ y_i (score=1). Those are the FPs. Show short negatives ⊆ y_i and their masked sig.
  const firesOn = (f) => { let hasRare = false; for (const b of f) { if (!pick.yset.has(b)) return false; if (rare(b)) hasRare = true; } return hasRare; };
  const matches = [];
  for (let si = 0; si < sents.length && matches.length < 12; si++) { const s = sents[si]; for (let t = 1; t < s.length; t++) { if (s[t] === TARGET) continue; const f = feat(s, t); if (firesOn(f)) matches.push({ ctx: ctx(s, t), tgt: s[t], m: f.filter(rare).map(lab), n: f.length }); } }
  w.write(`NEGATIVES the gate FIRES ON (x ⊆ y_i ⇒ score=1) — note none EQUAL the positive; each is a short SUBSET:\n`);
  for (const m of matches) w.write(`  {${m.ctx}} → [${m.tgt}]   (${m.n} bits, masked→ {${m.m.join(", ")}})\n`);
  w.write(`\n  → the gate scores 1 whenever ALL of a context's bits lie inside y_i — a 2-word context trivially does.\n`);
  w.write(`  Its masked signature is a SUBSET of p⁺, not equal to it, but score = |p⁺∧M∧x|/|M∧x| = 1 all the same.\n`);
  w.write(`  Exact-equality FPs would be rare (→ high precision, as expected); it's the SUBSET rule that leaks.\n`);
};

main();
