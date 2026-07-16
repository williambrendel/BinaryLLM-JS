"use strict";
// TEMP #1: average parts-per-word under two dictionaries, over a shared word set.
//   node scripts/_wordParts.mjs <wordsource.txt> <dictA.txt> <dictB.txt>
import fs from "node:fs";
import readline from "node:readline";
import { loadDictText } from "../src/core/parts/dictionary.js";
import { decomposeWord } from "../src/core/parts/decomposer.js";
import { tokenizeStream, StreamTokenType } from "../src/core/parts/tokenize.js";
import { Kind } from "../src/core/parts/kind.js";

const [srcPath, ...dictPaths] = process.argv.slice(2);

// TOKENS=1 -> token-weighted (every occurrence, real text); else distinct vocab
const tokenMode = process.env.TOKENS === "1";
const bag = tokenMode ? [] : new Set();
{
  const rl = readline.createInterface({
    input: fs.createReadStream(srcPath, { encoding: "latin1" }), crlfDelay: Infinity });
  for await (const line of rl)
    for (const t of tokenizeStream(line))
      if (t.type === StreamTokenType.Word) { tokenMode ? bag.push(t.value) : bag.add(t.value); }
}
const words = tokenMode ? bag : [...bag];
process.stdout.write(
  `word set: ${words.length} ${tokenMode ? "word tokens (frequency-weighted)" : "distinct words"} ` +
  `from ${srcPath.split("/").pop()}\n\n`);

const analyze = (path) => {
  const dict = loadDictText(fs.readFileSync(path, "latin1"));
  let sumParts = 0, sumLetters = 0, singleWhole = 0, usesAnyLetter = 0;
  const hist = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "6-8": 0, "9+": 0 };
  const perWord = new Map();
  for (const w of words) {
    const ids = decomposeWord(dict, w);
    const np = ids.length;
    let letters = 0;
    for (const id of ids) if (dict.at(id).kind === Kind.Letter) letters++;
    sumParts += np; sumLetters += letters;
    if (letters > 0) usesAnyLetter++;
    if (np === 1 && dict.at(ids[0]).kind === Kind.Whole) singleWhole++;
    const b = np <= 5 ? String(np) : np <= 8 ? "6-8" : "9+";
    hist[b]++;
    perWord.set(w, np);
  }
  const n = words.length;
  return {
    name: path.split("/").pop(), dict, perWord,
    meanParts: sumParts / n, meanLetters: sumLetters / n,
    singleWholePct: (100 * singleWhole) / n, usesLetterPct: (100 * usesAnyLetter) / n, hist, n,
  };
};

const R = dictPaths.map(analyze);
const pad = (s, w) => String(s).padStart(w);
process.stdout.write("metric".padEnd(34) + R.map((r) => pad(r.name, 20)).join("") + "\n");
const row = (label, f) => process.stdout.write(label.padEnd(34) + R.map((r) => pad(f(r), 20)).join("") + "\n");
row("mean parts / word", (r) => r.meanParts.toFixed(3));
row("mean letter-singletons / word", (r) => r.meanLetters.toFixed(3));
row("single-Whole coverage %", (r) => r.singleWholePct.toFixed(1));
row("words using >=1 letter-singleton %", (r) => r.usesLetterPct.toFixed(1));
for (const b of ["1", "2", "3", "4", "5", "6-8", "9+"])
  row(`  words with ${b} part(s) %`, (r) => (100 * r.hist[b] / r.n).toFixed(1));

// head-to-head per-word delta (A vs B), same word set
if (R.length === 2 && !tokenMode) {
  const [A, B] = R;
  let better = 0, same = 0, worse = 0, dsum = 0;
  for (const w of words) {
    const d = A.perWord.get(w) - B.perWord.get(w); dsum += d;
    if (d < 0) better++; else if (d === 0) same++; else worse++;
  }
  process.stdout.write(
    `\nhead-to-head (${A.name} vs ${B.name}), per word:\n` +
    `  mean delta (A-B): ${(dsum / words.length).toFixed(3)} parts` +
    `  |  A fewer: ${(100 * better / words.length).toFixed(1)}%` +
    `  same: ${(100 * same / words.length).toFixed(1)}%` +
    `  A more: ${(100 * worse / words.length).toFixed(1)}%\n`);
}
