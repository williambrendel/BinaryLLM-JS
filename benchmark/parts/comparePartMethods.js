"use strict";

// ============================================================================
// benchmark/comparePartMethods.js
//
// Compares the four-track COMBO part generator against the frequency PEELER on
// the same corpus. This is an ANALYSIS tool, not a unit test — it prints a
// report to stderr. All real logic lives in src/; here we only orchestrate.
//
// What we're checking (see benchmark/README.md for expected numbers):
//   - the two LEARNED tracks (generative + discriminative) are complementary,
//     not redundant (small gen∩disc overlap);
//   - the combo recovers common interior stems the pure tree drops;
//   - no learned parts at length 2 (bigrams are a fixed backstop, not learned);
//   - overall vocabulary size vs. the peeler.
//
// Usage: node benchmark/comparePartMethods.js <corpus.txt> [<corpus.txt> ...]
// ============================================================================

import fs from "node:fs";
import { Kind } from "../../src/core/parts/kind.js";
import { PartExtractor } from "../../src/core/parts/extractor.js";
import { tokenizeStream, StreamTokenType } from "../../src/core/parts/tokenize.js";
import { wordFreq } from "../../src/utilities/corpus/wordFreq.js";
import { generateParts } from "../../src/core/parts/generation/index.js";

const inPaths = process.argv.slice(2);
if (!inPaths.length) { process.stderr.write("usage: node benchmark/comparePartMethods.js <corpus...>\n"); process.exit(2); }
const bufs = inPaths.map((p) => new Uint8Array(fs.readFileSync(p)));
const w = process.stderr;
w.write(`\n################ ${inPaths.join(", ")} ################\n`);

// ---- COMBO (our generator) -------------------------------------------------
const freq = wordFreq(bufs);
const { dict: combo, stats } = generateParts(freq);
w.write(`\nCOMBO tracks: wholes ${stats.wholes} · generative ${stats.generative} · discriminative ${stats.discriminative}\n`);
w.write(`  gen ∩ disc: ${stats.sharedGenDisc}  ← small overlap ⇒ the two learned tracks are COMPLEMENTARY\n`);

// ---- PEELER (frequency baseline) -------------------------------------------
// The peeler trains its own dictionary by iteratively peeling uncovered runs.
const peeler = new PartExtractor();
for (const b of bufs) for (const t of tokenizeStream(b)) (t.type === StreamTokenType.Word) ? peeler.addWord(t.value) : peeler.addDelimiter(t.value);
const peelerDict = peeler.finalize();

// ---- report ----------------------------------------------------------------
const TRAINED = [Kind.Whole, Kind.Start, Kind.Mid, Kind.End];
const NAME = { [Kind.Whole]: "whole", [Kind.Start]: "start", [Kind.Mid]: "mid", [Kind.End]: "end" };
// "Learned" = trained word parts (length >= 2); excludes augmented atoms/letters.
const learned = (d) => d.allParts().filter((p) => TRAINED.includes(p.kind) && p.value.length >= 2);

const matrix = (label, parts) => {
  const maxL = parts.reduce((m, p) => Math.max(m, p.value.length), 0);
  const g = {}; for (const k of TRAINED) g[k] = new Array(maxL + 1).fill(0);
  for (const p of parts) g[p.kind][p.value.length]++;
  w.write(`\n  ${label} — parts by (kind × length):\n    len:${Array.from({ length: maxL - 1 }, (_, i) => String(i + 2).padStart(6)).join("")}   total\n`);
  for (const k of TRAINED) { let row = "    " + NAME[k].padEnd(6); let t = 0; for (let L = 2; L <= maxL; L++) { row += String(g[k][L]).padStart(6); t += g[k][L]; } w.write(row + String(t).padStart(8) + "\n"); }
  w.write(`    total learned: ${parts.length}\n`);
};

const comboParts = learned(combo), peelerParts = learned(peelerDict);
matrix("COMBO", comboParts);
matrix("PEELER", peelerParts);

const key = (p) => p.kind + "|" + p.value;
const cs = new Set(comboParts.map(key)), ps = new Set(peelerParts.map(key));
let shared = 0; for (const k of cs) if (ps.has(k)) shared++;
w.write(`\n  OVERLAP: combo=${cs.size} peeler=${ps.size} shared=${shared} combo-only=${cs.size - shared} peeler-only=${ps.size - shared} Jaccard=${(shared / (cs.size + ps.size - shared)).toFixed(3)}\n`);

// Spot-check: the interior stems the pure discriminative tree drops but the
// generative track (and the peeler) recover.
w.write(`\n  interior stems recovered (combo.hasMid):`);
for (const s of ["form", "graph", "tion", "ing"]) w.write(` ${s}=${combo.hasMid(s)}`);
w.write("\n");
