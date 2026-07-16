"use strict";

// ============================================================================
// benchmark/sigJaccardSweep.js
//
// Mean ± std pairwise Jaccard between two RANDOM context signatures, swept over
// pooling radius (1..10, and ∞ = full-sentence scope) for five signature types:
//
//   C        current word only        (radius-independent — the paradigmatic floor)
//   L        left pool only           (one band)
//   L∪C      union, F-wide            (the GPT/predictor UNION variant)
//   [L|C]    concat, 2F               (the position-preserving predictor variant)
//   [L,R]    concat, 2F               (the BERT/encoder signature)
//
// This is the missing measurement behind the concat-vs-union decision: concat
// doubles the bit space, so a lower Jaccard may be pure cardinality, not less
// sharing. The companion mean-popcount table is the confound to read alongside.
//
// A pair's Jaccard = |a∩b| / |a∪b| (0 if both empty). Mean/std are over a fixed,
// SEEDED set of random record-pairs — identical (i,j) indices across every radius
// and every variant, so each cell differs only by the representation.
//
// Usage: node benchmark/sigJaccardSweep.js <dict> <corpus> [N] [PAIRS]
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { streamSignatureChunks } from "../src/core/signatures/streamSignatures.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

// mulberry32 — a small seeded PRNG so pair indices are reproducible and shared
// across radii (differences between cells are the representation, not sampling).
const mkRng = (seed) => () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };

const jaccard = (a, b) => { let i = 0, j = 0, x = 0; while (i < a.length && j < b.length) { if (a[i] < b[j]) i++; else if (a[i] > b[j]) j++; else { x++; i++; j++; } } const u = a.length + b.length - x; return u === 0 ? 0 : x / u; };
const unionSorted = (a, b) => { const out = []; let i = 0, j = 0; while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } } while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out; };
const concat = (a, b, F) => { const out = new Array(a.length + b.length); let k = 0; for (let i = 0; i < a.length; i++) out[k++] = a[i]; for (let i = 0; i < b.length; i++) out[k++] = b[i] + F; return out; };

const VARIANTS = [
  ["C", (r, F) => r.C],
  ["L", (r, F) => r.L],
  ["L∪C", (r, F) => unionSorted(r.L, r.C)],
  ["[L|C]", (r, F) => concat(r.L, r.C, F)],
  ["[L,R]", (r, F) => concat(r.L, r.R, F)],
];

const FUZZY = process.env.FUZZY === "1";

const collect = async (dict, corpus, radius, N) => {
  const recs = [];
  for await (const { records, F } of streamSignatureChunks(dict, corpus, { chunkSize: 20000, radius, fuzzy: FUZZY })) {
    for (const r of records) { if (recs.length < N) recs.push(r); }
    if (recs.length >= N) return { recs, F };
  }
  return { recs, F: dict.size() };
};

const main = async () => {
  const [dictArg, corpusArg, nArg, pArg] = process.argv.slice(2);
  const N = Number(nArg || 20000), PAIRS = Number(pArg || 30000);
  const dict = loadDict(dictArg);
  const radii = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, null];

  // Fixed seeded (i,j) pair indices — shared across every radius and variant.
  const rng = mkRng(12345);
  const pairs = new Array(PAIRS);
  for (let s = 0; s < PAIRS; s++) { const i = Math.floor(rng() * N); let j = Math.floor(rng() * N); if (j === i) j = (j + 1) % N; pairs[s] = [i, j]; }

  // cells[variant][radiusIdx] = {mean, std}; pops[variant][radiusIdx] = mean popcount
  const cells = VARIANTS.map(() => []);
  const pops = VARIANTS.map(() => []);

  for (const radius of radii) {
    const { recs, F } = await collect(dict, corpusArg, radius, N);
    const n = recs.length;
    for (let v = 0; v < VARIANTS.length; v++) {
      const feat = VARIANTS[v][1];
      const sigs = recs.map((r) => feat(r, F));
      let sum = 0, sum2 = 0, psum = 0;
      for (const s of sigs) psum += s.length;
      for (const [i0, j0] of pairs) {
        const i = i0 % n, j = j0 % n;
        const jc = jaccard(sigs[i], sigs[j]);
        sum += jc; sum2 += jc * jc;
      }
      const mean = sum / PAIRS;
      const std = Math.sqrt(Math.max(0, sum2 / PAIRS - mean * mean));
      cells[v].push({ mean, std });
      pops[v].push(psum / n);
    }
    process.stderr.write(`  r=${radius === null ? "∞" : radius} done (n=${n})\n`);
  }

  const w = process.stdout;
  const colH = radii.map((r) => (r === null ? "∞" : String(r)));
  w.write(`# sigJaccardSweep — dict=${path.basename(dictArg)} corpus=${path.basename(corpusArg)} | N=${N} pairs=${PAIRS} | F=${dict.size()} | C=${FUZZY ? "FUZZY" : "sparse"}\n\n`);

  w.write(`## mean ± std pairwise Jaccard (random record pairs)\n\n`);
  w.write(`| sig | ` + colH.map((h) => `r${h}`).join(" | ") + ` |\n`);
  w.write(`|---|` + colH.map(() => "---").join("|") + `|\n`);
  for (let v = 0; v < VARIANTS.length; v++) {
    w.write(`| ${VARIANTS[v][0]} | ` + cells[v].map((c) => `${c.mean.toFixed(3)}±${c.std.toFixed(3)}`).join(" | ") + ` |\n`);
  }

  w.write(`\n## mean popcount (the cardinality confound)\n\n`);
  w.write(`| sig | ` + colH.map((h) => `r${h}`).join(" | ") + ` |\n`);
  w.write(`|---|` + colH.map(() => "---").join("|") + `|\n`);
  for (let v = 0; v < VARIANTS.length; v++) {
    w.write(`| ${VARIANTS[v][0]} | ` + pops[v].map((p) => p.toFixed(1)).join(" | ") + ` |\n`);
  }
};

main();
