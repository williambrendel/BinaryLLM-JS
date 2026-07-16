"use strict";

// ============================================================================
// benchmark/pipelineBench.js
//
// End-to-end pipeline benchmark: generation → encoding → normalization →
// signature. ANALYSIS tool (prints to stderr), not a unit test. See
// benchmark/README.md for the expected numbers.
//
// Reports:
//   1. BPE [L,C,R] decompositions for a few words.
//   2. Fuzzy typo-robustness — Jaccard(F, F') for word/typo pairs.
//   3. Normalizer — self-match rate (must be ~100%) and typo-recovery rate.
//   4. OOV-gate Jaccard distributions (typo→correct vs word→neighbor).
//   5. BPE coverage — parts/word, letter-use, characters in real parts.
//
// Usage: node benchmark/pipelineBench.js <corpus.txt> [<corpus.txt> ...]
// ============================================================================

import fs from "node:fs";
import { Kind } from "../../src/core/parts/kind.js";
import { wordFreq } from "../../src/utilities/corpus/wordFreq.js";
import { generateParts } from "../../src/core/parts/generation/index.js";
import { addBigramBackstop, fuzzyEncode, bpeEncode } from "../../src/core/parts/encoding/index.js";
import { buildInvertedIndex, nearestNeighbor } from "../../src/core/parts/normalization/index.js";
import { jaccard } from "../../src/core/math/sparse/jaccard.js";

const inPaths = process.argv.slice(2);
if (!inPaths.length) { process.stderr.write("usage: node benchmark/pipelineBench.js <corpus...>\n"); process.exit(2); }
const bufs = inPaths.map((p) => new Uint8Array(fs.readFileSync(p)));
const w = process.stderr;

// ---- build the pipeline ----------------------------------------------------
const freq = wordFreq(bufs);
const { dict, tracks } = generateParts(freq);
addBigramBackstop(dict);
const words = [...freq.keys()];
const index = buildInvertedIndex(words, words.map((x) => fuzzyEncode(dict, x)));
w.write(`\npipeline: ${words.length} words · dict F=${dict.size()} · ${index.postings.size} indexed parts\n`);

// ---- 1. [L,C,R] demo -------------------------------------------------------
const fmt = (s) => s.whole ? `WHOLE(${s.whole.value})` : `start=${s.start?.value ?? "∅"} mid=[${s.mid.map((c) => c.value).join(" ")}] end=${s.end?.value ?? "∅"}`;
w.write(`\n[1] BPE [L,C,R]:\n`);
for (const x of ["running", "reporting", "unhappiness", "1st", "the"]) w.write(`  ${x.padEnd(12)} ${fmt(bpeEncode(dict, x, tracks))}\n`);

// ---- 2. fuzzy typo robustness ----------------------------------------------
w.write(`\n[2] fuzzy Jaccard(F, F'):\n`);
for (const [a, b] of [["running", "runing"], ["reporting", "reporfing"], ["running", "jumping"]])
  w.write(`  ${a} / ${b}: ${jaccard(fuzzyEncode(dict, a), fuzzyEncode(dict, b)).toFixed(3)}\n`);

// ---- 3 & 4. normalizer + OOV gate ------------------------------------------
const sample = (minLen, n) => { const out = []; const stride = Math.max(1, Math.floor(words.length / n)); for (let i = 0; i < words.length && out.length < n; i += stride) if (words[i].length >= minLen) out.push(i); return out; };
const typo = (x) => { const i = x.length >> 1; return x.slice(0, i) + (x[i] === "x" ? "y" : "x") + x.slice(i + 1); };

let selfOk = 0, selfN = 0;
for (const wi of sample(3, 2000)) { selfN++; const r = nearestNeighbor(index.signatures[wi], index); if (r.wi === wi && r.j > 0.999) selfOk++; }

let recOk = 0, recN = 0; const recJ = [], oovJ = [];
for (const wi of sample(4, 2000)) {
  const t = typo(words[wi]); if (t === words[wi]) continue; recN++;
  const r = nearestNeighbor(fuzzyEncode(dict, t), index);
  if (r.wi === wi) { recOk++; recJ.push(r.j); }
  const o = nearestNeighbor(index.signatures[wi], index, wi); // nearest OTHER word (OOV proxy)
  if (o.wi >= 0) oovJ.push(o.j);
}
const p50 = (arr) => { const s = [...arr].sort((a, b) => a - b); return s[s.length >> 1]; };
w.write(`\n[3] normalizer:  self-match ${(100 * selfOk / selfN).toFixed(1)}%  (known words return themselves)\n`);
w.write(`                 typo-recovery ${(100 * recOk / recN).toFixed(1)}%  (typo → original)\n`);
w.write(`[4] OOV gate (Jaccard medians):  typo→correct ${p50(recJ).toFixed(2)}   word→neighbor ${p50(oovJ).toFixed(2)}   (overlap ⇒ gate = snap-if-close, not a typo detector)\n`);

// ---- 5. coverage -----------------------------------------------------------
let mass = 0, parts = 0, letterTok = 0, chars = 0, letterChars = 0;
for (const [x, f] of freq) {
  const s = bpeEncode(dict, x, tracks); mass += f; chars += x.length * f;
  parts += s.parts.length * f;
  // A "lone letter" is a length-1 fragment fill (Start/Mid/End), not a 1-char Whole.
  const nl = s.parts.filter((c) => c.value.length === 1 && c.kind !== Kind.Whole).length;
  letterChars += nl * f; if (nl > 0) letterTok += f;
}
w.write(`\n[5] coverage:  ${(parts / mass).toFixed(2)} parts/word · ${(100 * letterTok / mass).toFixed(1)}% tokens use a lone letter · ${(100 * (chars - letterChars) / chars).toFixed(1)}% of chars in real (≥2) parts\n`);
