"use strict";

// ============================================================================
// benchmark/letterFilling.js
//
// Single-letter "filler" analysis: how often decomposition falls back to a
// length-1 positional Letter, which words, and how the bigram backstop changes
// it. A lone Letter is the CORRECT representation of an atomic character (the
// "1" in "1st"), not a failure — but its rate is a useful coverage signal.
//
// Reports (per corpus), using the OPTIMAL min-letter tiling (a DP, strict
// positions — not the greedy peel, which overstates letters):
//   - filler rate by word LENGTH (the U-shape: short len-3 + long tail worst);
//   - overall by-token (freq-weighted) vs by-type (distinct word);
//   - fraction of CHARACTERS carried by real (>=2) parts;
//   - the effect of the 676-bigram backstop (with vs without).
//
// Usage: node benchmark/letterFilling.js
// ============================================================================

import fs from "node:fs";
import { Kind, kMaxPartLength, kInvalidPartId } from "../../src/core/parts/kind.js";
import { wordFreq } from "../../src/utilities/corpus/wordFreq.js";
import { generateParts } from "../../src/core/parts/generation/index.js";
import { addBigramBackstop } from "../../src/core/parts/encoding/index.js";

const CORPORA = [
  ["english", ["data/corpora/alice_en.txt", "data/corpora/pride_en.txt"]],
  ["wiki", ["data/corpora/wiki.test.txt"]],
];

// Optimal min-letter tiling (DP): Start/End proper affixes (L<n), Mid interior,
// whole fast-path. Returns the minimum number of length-1 positional letters.
const minLetters = (dict, word) => {
  const n = word.length; if (n === 0) return 0;
  if (dict.hasWhole(word)) return 0;
  const has = (k, s) => dict.lookup(k, s) !== kInvalidPartId;
  const cost = new Array(n + 1).fill(Infinity); cost[0] = 0;
  for (let i = 1; i <= n; i++) {
    if (cost[i - 1] + 1 < cost[i]) cost[i] = cost[i - 1] + 1;
    for (let L = 2; L <= kMaxPartLength && L <= i; L++) {
      const s = i - L, sub = word.slice(s, i);
      let k = -1;
      if (s === 0 && i < n) k = Kind.Start; else if (i === n && s > 0) k = Kind.End; else if (s > 0 && i < n) k = Kind.Mid; else continue;
      if (has(k, sub) && cost[s] < cost[i]) cost[i] = cost[s];
    }
  }
  return cost[n];
};

const w = process.stdout;

for (const [label, paths] of CORPORA) {
  const bufs = paths.map((p) => new Uint8Array(fs.readFileSync(p)));
  const freq = wordFreq(bufs);
  const { dict } = generateParts(freq);
  const entries = [...freq.entries()];

  const measure = () => {
    const byLen = new Map(); // len -> {types, wl}
    let types = 0, typeWL = 0, mass = 0, tokWL = 0, chars = 0, letterChars = 0;
    for (const [x, f] of entries) {
      const ml = minLetters(dict, x); const n = x.length;
      types++; mass += f; chars += n * f; letterChars += ml * f;
      if (ml > 0) { typeWL++; tokWL += f; }
      const b = byLen.get(n) || { t: 0, wl: 0 }; b.t++; if (ml > 0) b.wl++; byLen.set(n, b);
    }
    return { byLen, types, typeWL, mass, tokWL, chars, letterChars };
  };

  const before = measure();          // learned parts only (no bigram backstop)
  addBigramBackstop(dict);
  const after = measure();           // + bigram backstop

  w.write(`\n### ${label}\n\n`);
  w.write(`overall filler rate — no bigrams: by-token ${(100 * before.tokWL / before.mass).toFixed(1)}% / by-type ${(100 * before.typeWL / before.types).toFixed(1)}% · chars-as-letters ${(100 * before.letterChars / before.chars).toFixed(1)}%\n`);
  w.write(`overall filler rate — +bigrams:  by-token ${(100 * after.tokWL / after.mass).toFixed(1)}% / by-type ${(100 * after.typeWL / after.types).toFixed(1)}% · chars-as-letters ${(100 * after.letterChars / after.chars).toFixed(1)}%\n\n`);
  w.write(`by length (%need >=1 letter, OPTIMAL, +bigrams):\n`);
  w.write(`| len | ` + [3, 4, 5, 6, 7, 8, 9, 10, 12, 15].join(" | ") + " |\n|" + "---|".repeat(11) + "\n| % | ");
  for (const L of [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]) { const b = after.byLen.get(L); w.write((b ? Math.round(100 * b.wl / b.t) : "-") + " | "); }
  w.write("\n");
}
