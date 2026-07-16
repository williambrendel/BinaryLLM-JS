"use strict";

// ============================================================================
// benchmark/partsByLayerType.js
//
// Part distribution per LAYER (length) and per TYPE (kind), across multiple
// corpora — the saturation view. Uses the frequency PEELER (it scales to large
// vocabularies; the discriminative tree does not), mirroring the C++ analysis:
// affixes skew long (7/6 the biggest buckets), wholes hump at 4–5, and parts
// grow strongly sub-linearly with corpus size.
//
// Corpora (named in the task): english (alice+pride), wiki (wiki.test),
// words_en (the ~470k-word list). Reports, per corpus:
//   - kind × length matrix (whole/start/mid/end),
//   - combined affix (S+M+E) by length,
//   - whole atoms by length.
//
// Usage: node benchmark/partsByLayerType.js
// ============================================================================

import fs from "node:fs";
import { Kind } from "../../src/core/parts/kind.js";
import { PartExtractor } from "../../src/core/parts/extractor.js";
import { tokenizeStream, StreamTokenType } from "../../src/core/parts/tokenize.js";

const CORPORA = [
  ["english", ["data/corpora/alice_en.txt", "data/corpora/pride_en.txt"]],
  ["wiki", ["data/corpora/wiki.test.txt"]],
  ["words_en", ["data/corpora/words_en.txt"]],
];

const TRAINED = [Kind.Whole, Kind.Start, Kind.Mid, Kind.End];
const NAME = { [Kind.Whole]: "whole", [Kind.Start]: "start", [Kind.Mid]: "mid", [Kind.End]: "end" };
const learned = (d) => d.allParts().filter((p) => TRAINED.includes(p.kind) && p.value.length >= 2);
const w = process.stdout;

for (const [label, paths] of CORPORA) {
  if (!paths.every((p) => fs.existsSync(p))) { w.write(`\n### ${label}: SKIP (missing ${paths.find((p) => !fs.existsSync(p))})\n`); continue; }
  const ex = new PartExtractor();
  let tokens = 0;
  for (const p of paths) for (const t of tokenizeStream(new Uint8Array(fs.readFileSync(p)))) { tokens++; (t.type === StreamTokenType.Word) ? ex.addWord(t.value) : ex.addDelimiter(t.value); }
  const dict = ex.finalize();
  const parts = learned(dict);
  const maxL = parts.reduce((m, p) => Math.max(m, p.value.length), 0);
  const grid = {}; for (const k of TRAINED) grid[k] = new Array(maxL + 1).fill(0);
  for (const p of parts) grid[p.kind][p.value.length]++;

  w.write(`\n### ${label}  (${tokens} tokens, ${parts.length} trained parts)\n\n`);
  let head = "| kind | " + Array.from({ length: maxL - 1 }, (_, i) => i + 2).join(" | ") + " | total |\n";
  w.write(head + "|" + "---|".repeat(maxL) + "\n");
  for (const k of TRAINED) { let tot = 0; let row = `| ${NAME[k]} | `; for (let L = 2; L <= maxL; L++) { row += grid[k][L] + " | "; tot += grid[k][L]; } w.write(row + tot + " |\n"); }
  // combined affix + whole by length
  w.write(`\naffix(S+M+E) by length: ` + Array.from({ length: maxL - 1 }, (_, i) => { const L = i + 2; return `${L}:${grid[Kind.Start][L] + grid[Kind.Mid][L] + grid[Kind.End][L]}`; }).join(" ") + "\n");
  w.write(`whole by length:        ` + Array.from({ length: maxL - 1 }, (_, i) => { const L = i + 2; return `${L}:${grid[Kind.Whole][L]}`; }).join(" ") + "\n");
}
