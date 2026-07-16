"use strict";
// TEMP #2/#3: whole-atom frequency histogram, + Whole-cutoff ablation
// (parts/word on prose and word-collision rate vs number of Whole atoms kept).
//   node scripts/_wholeAblation.mjs <dict> <proseFile> <freqCorpus...>
import fs from "node:fs";
import readline from "node:readline";
import { loadDictText, PartDictionary } from "../src/core/parts/dictionary.js";
import { decomposeWord } from "../src/core/parts/decomposer.js";
import { tokenizeStream, StreamTokenType } from "../src/core/parts/tokenize.js";
import { Kind } from "../src/core/parts/kind.js";

const [dictPath, prosePath, ...freqCorpora] = process.argv.slice(2);
const CONN = new Set(["-", "'", "&", ",", ".", "$"]);
const isAugWhole = (v) => v.length === 1 &&
  ((v >= "a" && v <= "z") || (v >= "0" && v <= "9") || CONN.has(v));

const dict = loadDictText(fs.readFileSync(dictPath, "latin1"));

// trained (non-augmented) whole atoms
const wholes = [];
for (const { kind, value } of dict.allParts())
  if (kind === Kind.Whole && !isAugWhole(value)) wholes.push(value);
const wholeSet = new Set(wholes);

// --- corpus frequencies for the whole atoms (single streaming pass) ---
const freq = new Map(wholes.map((w) => [w, 0]));
for (const cp of freqCorpora) {
  const rl = readline.createInterface({
    input: fs.createReadStream(cp, { encoding: "latin1" }), crlfDelay: Infinity });
  for await (const line of rl)
    for (const t of tokenizeStream(line))
      if (t.type === StreamTokenType.Word && freq.has(t.value)) freq.set(t.value, freq.get(t.value) + 1);
}

// #2 histogram
const buckets = { "1": 0, "2-9": 0, "10-99": 0, "100-999": 0, "1k-9k": 0, "10k+": 0 };
for (const w of wholes) {
  const c = freq.get(w);
  const b = c <= 1 ? "1" : c < 10 ? "2-9" : c < 100 ? "10-99" : c < 1000 ? "100-999" : c < 10000 ? "1k-9k" : "10k+";
  buckets[b]++;
}
process.stdout.write(`\n#2  Whole atoms in ${dictPath.split("/").pop()}: ${wholes.length} (trained)\n`);
process.stdout.write(`    corpus-frequency histogram:\n`);
for (const [b, n] of Object.entries(buckets))
  process.stdout.write(`      ${b.padEnd(9)} ${String(n).padStart(5)}  (${(100 * n / wholes.length).toFixed(1)}%)\n`);

// rank wholes by frequency desc (ties by value for determinism)
const ranked = [...wholes].sort((a, b) => (freq.get(b) - freq.get(a)) || (a < b ? -1 : 1));

// prose tokens (frequency-weighted) + distinct prose vocab (for collisions)
const proseTokens = [];
const proseVocab = new Set();
{
  const rl = readline.createInterface({
    input: fs.createReadStream(prosePath, { encoding: "latin1" }), crlfDelay: Infinity });
  for await (const line of rl)
    for (const t of tokenizeStream(line))
      if (t.type === StreamTokenType.Word) { proseTokens.push(t.value); proseVocab.add(t.value); }
}
const vocabArr = [...proseVocab];

// build a dict copy keeping only the top-N trained wholes (augmented + all other kinds kept)
const filtered = (keepSet) => {
  const d = new PartDictionary();
  for (const { kind, value } of dict.allParts()) {
    if (kind === Kind.Whole && !isAugWhole(value) && !keepSet.has(value)) continue;
    d.add(kind, value);
  }
  return d;
};

const evalDict = (d) => {
  let sumParts = 0, sumLetters = 0;
  for (const w of proseTokens) {
    const ids = decomposeWord(d, w);
    sumParts += ids.length;
    for (const id of ids) if (d.at(id).kind === Kind.Letter) sumLetters++;
  }
  // collision: distinct words sharing a decomposition part-set
  const sig = new Map();
  for (const w of vocabArr) {
    const key = [...new Set(decomposeWord(d, w))].sort((a, b) => a - b).join(",");
    sig.set(key, (sig.get(key) || 0) + 1);
  }
  let collided = 0;
  for (const c of sig.values()) if (c > 1) collided += c;
  return {
    meanParts: sumParts / proseTokens.length,
    meanLetters: sumLetters / proseTokens.length,
    collPct: 100 * collided / vocabArr.length,
  };
};

process.stdout.write(
  `\n#3  Whole-cutoff ablation  (prose=${prosePath.split("/").pop()}, ${proseTokens.length} tokens, ` +
  `${vocabArr.length} distinct)\n`);
process.stdout.write(
  "  keep-top-N".padEnd(14) + "parts/word".padStart(12) + "letters/word".padStart(14) +
  "word-coll%".padStart(12) + "\n");
for (const N of [wholes.length, 500, 200, 100, 50, 0]) {
  const keep = new Set(ranked.slice(0, N));
  const r = evalDict(filtered(keep));
  const label = N === wholes.length ? `${N} (all)` : String(N);
  process.stdout.write(
    ("  " + label).padEnd(14) + r.meanParts.toFixed(3).padStart(12) +
    r.meanLetters.toFixed(3).padStart(14) + r.collPct.toFixed(2).padStart(12) + "\n");
}
