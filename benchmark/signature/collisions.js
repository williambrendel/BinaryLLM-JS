"use strict";

// ============================================================================
// benchmark/collisions.js
//
// Shows the NON-UNIQUE signatures: groups of DIFFERENT words that collapse to
// the SAME signature, for each of F / [L,R] / [L∪C]. For every colliding word it
// pulls a real sentence from the corpus (word highlighted in ⟦brackets⟧) so the
// differences are visible — you see distinct words in context mapping to one
// signature.
//
// Uses the English novels by default (readable prose). Collision RATES are the
// §4(a) numbers (measured on wiki); here we want legible EXAMPLES.
//
// Usage: node benchmark/collisions.js [corpus...]
// ============================================================================

import fs from "node:fs";
import { wordFreq } from "../../src/utilities/corpus/wordFreq.js";
import { generateParts } from "../../src/core/parts/generation/index.js";
import { addBigramBackstop, fuzzyEncode, bpeEncode } from "../../src/core/parts/encoding/index.js";

const paths = process.argv.slice(2).length ? process.argv.slice(2) : ["data/corpora/alice_en.txt", "data/corpora/pride_en.txt"];
const texts = paths.map((p) => fs.readFileSync(p, "utf8"));
const freq = wordFreq(texts.map((t) => new TextEncoder().encode(t)));
const { dict, tracks } = generateParts(freq);
addBigramBackstop(dict);
const words = [...freq.keys()];

// Signature keys + human-readable display for one word.
const sig = (x) => {
  const e = bpeEncode(dict, x, tracks);
  if (e.whole) return { LR: "W" + e.whole.id, LRd: `whole:${e.whole.value}`, LUC: "W" + e.whole.id, LUCd: `{${e.whole.value}}` };
  const LR = `${e.start ? e.start.id : "_"}|${e.end ? e.end.id : "_"}`;
  const LRd = `start=${e.start ? e.start.value : "∅"} end=${e.end ? e.end.value : "∅"}`;
  const lucIds = new Set(); const lucVals = []; if (e.start) { lucIds.add(e.start.id); lucVals.push(e.start.value); } for (const c of e.mid) { lucIds.add(c.id); lucVals.push(c.value); }
  return { LR, LRd, LUC: [...lucIds].sort((a, b) => a - b).join(","), LUCd: `{${lucVals.join(" ")}}` };
};

// Group alphabetic words (>=4 letters, for legibility) by each signature key.
const alpha = words.filter((x) => /^[a-z]{4,}$/.test(x));
const cache = new Map(alpha.map((x) => [x, sig(x)]));
const group = (field) => { const m = new Map(); for (const x of alpha) { const k = cache.get(x)[field]; let a = m.get(k); if (!a) m.set(k, (a = [])); a.push(x); } return m; };
const Fkey = new Map(alpha.map((x) => [x, fuzzyEncode(dict, x).join(",")]));
const groupF = () => { const m = new Map(); for (const x of alpha) { const k = Fkey.get(x); let a = m.get(k); if (!a) m.set(k, (a = [])); a.push(x); } return m; };

// Sentences: split corpus, index the first sentence containing each target word.
const sentences = texts.join(" ").replace(/\s+/g, " ").split(/(?<=[.!?])\s+/).filter((s) => s.length > 20 && s.length < 240);
const findSentence = (word) => {
  const re = new RegExp(`\\b${word}\\b`, "i");
  for (const s of sentences) if (re.test(s)) return s.replace(re, (m) => `⟦${m}⟧`);
  return null;
};

const w = process.stdout;
const showGroups = (title, map, n) => {
  w.write(`\n## ${title} collisions (different words → same signature)\n`);
  const collisions = [...map.entries()].filter(([, ws]) => ws.length >= 2)
    .map(([, ws]) => ({ ws, mass: ws.reduce((s, x) => s + freq.get(x), 0) }))
    .sort((a, b) => b.mass - a.mass).slice(0, n);
  for (const { ws } of collisions) {
    const shown = ws.slice(0, 4);
    w.write(`\n  signature ${cache.get(shown[0]).LRd || ""}${cache.get(shown[0]).LUCd && title.includes("L∪C") ? " → " + cache.get(shown[0]).LUCd : ""}  —  {${ws.join(", ")}}\n`);
    for (const word of shown) { const s = findSentence(word); if (s) w.write(`    ${word.padEnd(12)} “…${s}…”\n`); else w.write(`    ${word.padEnd(12)} (no sentence found)\n`); }
  }
};

w.write(`# Signature collisions — ${paths.map((p) => p.split("/").pop()).join(", ")}  (${alpha.length} alpha words)\n`);
showGroups("[L,R]", group("LR"), 5);
showGroups("[L∪C]", group("LUC"), 5);
showGroups("F", groupF(), 5);
