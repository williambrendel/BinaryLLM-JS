"use strict";

// ============================================================================
// benchmark/atomChannel.js
//
// Does the encoding carry a sub-word GENERALIZATION channel bigram lacks, or is
// it a one-hot-of-whole-words with decoration? Decided by two facts per atom:
//   (1) its Kind  {Whole, Start, End, Mid}  — Whole = df 1 = lexical = bigram has it
//   (2) its df    (# distinct word TYPES that peel to it) — high df = generalizing
//
// Reports, over the corpus:
//   A. dict-level Kind composition
//   B. fraction of token-atom-OCCURRENCES that are non-Whole  (the headline)
//   C. df-distribution of non-Whole occurrences — high-df (shared, generalizing)
//      vs df=1 (word-specific, no generalization)
//   D. mean #atoms per token by Kind, split by word-frequency (function↔content proxy)
//
// Usage: node benchmark/atomChannel.js <dict> <corpus> [FUZZY=1]
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType } from "../src/core/parts/tokenize.js";
import { encodeWord, fuzzyEncodeWord } from "../src/core/signatures/wordEncoder.js";
import { Kind, kindToString } from "../src/core/parts/kind.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const [dictArg, corpusArg] = process.argv.slice(2);
const FUZZY = process.env.FUZZY === "1";
const bagOf = FUZZY ? fuzzyEncodeWord : encodeWord;
const dict = loadDict(dictArg);
const parts = dict.allParts(); // id -> {kind, value}

// --- A. dict-level Kind composition -----------------------------------------
const w = process.stdout;
w.write(`# atomChannel — dict=${path.basename(dictArg)} corpus=${path.basename(corpusArg)} | F=${dict.size()} | encoding=${FUZZY ? "FUZZY" : "sparse"}\n\n`);
w.write(`A. dictionary Kind composition\n`);
for (const k of [Kind.Whole, Kind.Start, Kind.End, Kind.Mid, Kind.Delimiter]) {
  w.write(`   ${kindToString(k).padEnd(6)} ${String(dict.countOfKind(k)).padStart(6)}\n`);
}

// --- word types + token frequency -------------------------------------------
const freq = new Map();
const text = fs.readFileSync(corpusArg, "utf8");
for (const line of text.split("\n")) {
  for (const t of tokenizeStream(line)) {
    if (t.type === StreamTokenType.Word) { const v = t.value.toLowerCase(); freq.set(v, (freq.get(v) || 0) + 1); }
  }
}
const types = [...freq.keys()];

// --- encode each type, record atom kinds + df (type frequency of each atom) --
const df = new Map(); // atom id -> # distinct word types containing it
const typeAtoms = new Map(); // word -> atom ids
for (const word of types) {
  const ids = bagOf(dict, word);
  typeAtoms.set(word, ids);
  for (const id of ids) df.set(id, (df.get(id) || 0) + 1);
}

// --- B & C: token-weighted occurrence tallies -------------------------------
const kindName = (id) => kindToString(parts[id].kind);
let totalOcc = 0;
const occByKind = { whole: 0, start: 0, end: 0, mid: 0, delim: 0 };
// df buckets for NON-whole occurrences
const dfBucket = { "1": 0, "2-9": 0, "10-99": 0, "100+": 0 };
const bucketOf = (d) => (d === 1 ? "1" : d < 10 ? "2-9" : d < 100 ? "10-99" : "100+");
for (const [word, f] of freq) {
  for (const id of typeAtoms.get(word)) {
    const kn = kindName(id);
    occByKind[kn] = (occByKind[kn] || 0) + f;
    totalOcc += f;
    if (kn !== "whole") dfBucket[bucketOf(df.get(id))] += f;
  }
}
const nonWhole = totalOcc - occByKind.whole;
w.write(`\nB. token-atom-OCCURRENCES by Kind (weighted by token frequency)\n`);
for (const k of ["whole", "start", "end", "mid", "delim"]) {
  const o = occByKind[k] || 0; w.write(`   ${k.padEnd(6)} ${String(o).padStart(9)}  ${(100 * o / totalOcc).toFixed(1)}%\n`);
}
w.write(`   ── non-Whole fraction = ${(100 * nonWhole / totalOcc).toFixed(1)}%  ← the headline\n`);

w.write(`\nC. df of NON-Whole occurrences  (df = # word types sharing the atom)\n`);
w.write(`   df=1 (word-specific, NO generalization) : ${(100 * dfBucket["1"] / nonWhole).toFixed(1)}%\n`);
w.write(`   df 2-9                                   : ${(100 * dfBucket["2-9"] / nonWhole).toFixed(1)}%\n`);
w.write(`   df 10-99                                 : ${(100 * dfBucket["10-99"] / nonWhole).toFixed(1)}%\n`);
w.write(`   df 100+ (broadly shared, generalizing)   : ${(100 * dfBucket["100+"] / nonWhole).toFixed(1)}%\n`);
w.write(`   ── generalizing (df>=10) share of non-Whole = ${(100 * (dfBucket["10-99"] + dfBucket["100+"]) / nonWhole).toFixed(1)}%\n`);

// --- D: per-token composition split by frequency (function <-> content proxy)-
// Proxy: rank by token frequency. Top band ~ function words; tail ~ content.
const ranked = [...freq.entries()].sort((a, b) => b[1] - a[1]);
const bands = [
  ["top-50 (function proxy)", ranked.slice(0, 50)],
  ["rank 51-500", ranked.slice(50, 500)],
  ["rank 501-5000", ranked.slice(500, 5000)],
  ["tail 5000+ (content proxy)", ranked.slice(5000)],
];
w.write(`\nD. mean #atoms per TOKEN by Kind, split by frequency band (POS proxy)\n`);
w.write(`   band                          tokens   whole  start   end    mid   |  %non-whole\n`);
for (const [name, entries] of bands) {
  let tok = 0; const c = { whole: 0, start: 0, end: 0, mid: 0, delim: 0 };
  for (const [word, f] of entries) { tok += f; for (const id of typeAtoms.get(word)) c[kindName(id)] = (c[kindName(id)] || 0) + f; }
  if (!tok) continue;
  const per = (k) => (c[k] / tok).toFixed(2).padStart(6);
  const nw = 100 * (tok ? ((c.start + c.end + c.mid) / (c.whole + c.start + c.end + c.mid + c.delim)) : 0);
  w.write(`   ${name.padEnd(28)} ${String(tok).padStart(7)}  ${per("whole")} ${per("start")} ${per("end")} ${per("mid")}  |  ${nw.toFixed(1)}%\n`);
}

// --- top generalizing non-whole atoms, by kind, for eyeballing ---------------
w.write(`\nE. highest-df non-Whole atoms (the generalization channel, if it exists)\n`);
const nonWholeIds = [...df.keys()].filter((id) => kindName(id) !== "whole" && kindName(id) !== "delim");
nonWholeIds.sort((a, b) => df.get(b) - df.get(a));
for (const id of nonWholeIds.slice(0, 20)) w.write(`   ${kindName(id).padEnd(5)} "${parts[id].value}"  df=${df.get(id)}\n`);
