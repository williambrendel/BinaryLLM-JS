"use strict";

/**
 * @file scripts/buildDict.js
 * @brief CLI runner: build the combo part dictionary from a corpus and save it.
 *
 * Thin wrapper over the library — all logic lives in
 * {@link module:core/parts/generation/generateParts}. Writes the trained
 * dictionary (text format); the per-part track tags the encoder uses are folded
 * into the marker glyph (`"__"` discriminative, `"##"` generative), so no
 * separate side-car file is needed — `loadDictText` rebuilds them.
 *
 * Usage:
 *   node scripts/buildDict.js <out.txt|-> <input.txt> [<input.txt> ...]
 *
 * A bare filename lands in data/dictionaries/; a path is used as-is; "-" writes
 * the dictionary text to stdout.
 *
 * Example:
 *   node scripts/buildDict.js combo_en.txt data/corpora/alice_en.txt data/corpora/pride_en.txt
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { wordFreq } from "../src/utilities/corpus/wordFreq.js";
import { generateParts } from "../src/core/parts/generation/index.js";
import { saveDictText } from "../src/core/parts/dictionary.js";

const DICT_DIR = path.resolve(fileURLToPath(import.meta.url), "..", "..", "data", "dictionaries");

const main = () => {
  const args = process.argv.slice(2);
  if (args.length < 2) {
    process.stderr.write("usage: node scripts/buildDict.js <out.txt|-> <input.txt> [<input.txt> ...]\n");
    process.exit(2);
  }
  const [outPath, ...inPaths] = args;
  const bufs = inPaths.map((p) => new Uint8Array(fs.readFileSync(p === "-" ? 0 : p)));
  const freq = wordFreq(bufs);
  const { dict, tracks, stats } = generateParts(freq);

  process.stderr.write(`corpus: ${stats.words} words | wholes ${stats.wholes}, generative ${stats.generative}, discriminative ${stats.discriminative} (gen∩disc ${stats.sharedGenDisc})\n`);

  // Provenance (d/g/w) is folded into the marker glyph in the text file itself
  // ("__" = discriminative, "##" = generative), so loadDictText rebuilds `tracks`
  // — no separate .tracks.json side-car.
  const text = saveDictText(dict, tracks);
  if (outPath === "-") { process.stdout.write(Buffer.from(text, "latin1")); return; }
  const resolved = path.dirname(outPath) === "." && !path.isAbsolute(outPath) ? path.join(DICT_DIR, outPath) : outPath;
  fs.mkdirSync(path.dirname(resolved), { recursive: true });
  fs.writeFileSync(resolved, Buffer.from(text, "latin1"));
  process.stderr.write(`wrote ${resolved}\n`);
};

main();
