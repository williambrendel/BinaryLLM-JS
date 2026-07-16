"use strict";

// ============================================================================
// benchmark/nnTypoSweep.js
//
// Effectiveness of the NN normalizer under typos: recovery rate (corrupted word
// resolves to its original) vs. corruption level, and across edit types.
// Corruption is deterministic (seeded RNG) so results are reproducible.
//
// Reports:
//   - substitution RATE sweep (0.5% … 25% of characters): recovery% + mean J;
//   - single-edit comparison across edit types (sub / del / ins / transpose).
//
// Usage: node benchmark/nnTypoSweep.js [corpus...]  (default: wiki.test)
// ============================================================================

import fs from "node:fs";
import { wordFreq } from "../../src/utilities/corpus/wordFreq.js";
import { generateParts } from "../../src/core/parts/generation/index.js";
import { addBigramBackstop, fuzzyEncode } from "../../src/core/parts/encoding/index.js";
import { buildInvertedIndex, nearestNeighbor } from "../../src/core/parts/normalization/index.js";

const paths = process.argv.slice(2).length ? process.argv.slice(2) : ["data/corpora/wiki.test.txt"];
const bufs = paths.map((p) => new Uint8Array(fs.readFileSync(p)));
const freq = wordFreq(bufs);
const { dict } = generateParts(freq);
addBigramBackstop(dict);
const words = [...freq.keys()];
const index = buildInvertedIndex(words, words.map((x) => fuzzyEncode(dict, x)));

// Deterministic PRNG.
let seed = 0x9e3779b9;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const rchar = () => String.fromCharCode(97 + Math.floor(rnd() * 26));
const subst = (x, p) => { let o = ""; for (const c of x) { if (rnd() < p) { let d; do { d = rchar(); } while (d === c); o += d; } else o += c; } return o; };
const del = (x) => { const i = Math.floor(rnd() * x.length); return x.slice(0, i) + x.slice(i + 1); };
const ins = (x) => { const i = Math.floor(rnd() * (x.length + 1)); return x.slice(0, i) + rchar() + x.slice(i); };
const transpose = (x) => { if (x.length < 2) return x; const i = Math.floor(rnd() * (x.length - 1)); return x.slice(0, i) + x[i + 1] + x[i] + x.slice(i + 2); };

const sampleIdx = (() => { const out = []; const stride = Math.max(1, Math.floor(words.length / 3000)); for (let i = 0; i < words.length && out.length < 3000; i += stride) if (words[i].length >= 4) out.push(i); return out; })();

const run = (mutate) => {
  let rec = 0, js = 0, n = 0;
  for (const wi of sampleIdx) { const m = mutate(words[wi]); if (m === words[wi]) continue; n++; const r = nearestNeighbor(fuzzyEncode(dict, m), index); if (r.wi === wi) rec++; js += r.j; }
  return { recovery: 100 * rec / n, meanJ: js / n, n };
};

const w = process.stdout;
w.write(`\n# NN typo recovery — ${paths.join(", ")}  (${words.length} words, sample ${sampleIdx.length})\n`);
w.write(`\nsubstitution RATE sweep:\n| error % | recovery % | mean J |\n|---|---|---|\n`);
for (const p of [0.005, 0.05, 0.10, 0.15, 0.20, 0.25]) { const r = run((x) => subst(x, p)); w.write(`| ${(100 * p).toFixed(1)} | ${r.recovery.toFixed(1)} | ${r.meanJ.toFixed(3)} |\n`); }
w.write(`\nsingle-edit by type:\n| edit | recovery % | mean J |\n|---|---|---|\n`);
for (const [nm, fn] of [["substitution", (x) => subst(x, 1 / Math.max(1, x.length))], ["deletion", del], ["insertion", ins], ["transposition", transpose]]) { const r = run(fn); w.write(`| ${nm} | ${r.recovery.toFixed(1)} | ${r.meanJ.toFixed(3)} |\n`); }
