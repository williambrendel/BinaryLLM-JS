"use strict";

// ============================================================================
// scripts/extractParts.js
//
// Build a part dictionary from one or more corpora — the JS equivalent of the
// C++ `phase1_extract`. Reads raw text (as bytes), tokenizes, feeds a single
// PartExtractor, and writes the resulting dictionary in the trained-only
// format (singletons / connector specials are regenerated on load, not
// stored).
//
// Output format is chosen by extension:
//   .bin            -> binary (BDI2)
//   anything else   -> text ("##" convention, [delim] section)
//   -               -> text to stdout
//
// Output location: a bare filename (no directory component) lands in
// data/dictionaries/ by default; pass a path with a directory (or an absolute
// path) to write elsewhere. The target directory is created if needed.
//
// Usage:
//   node scripts/extractParts.js <output.txt|.bin|-> <input.txt|-> [<input> ...]
//
// Examples:
//   node scripts/extractParts.js dict.txt data/corpora/alice_en.txt   # -> data/dictionaries/dict.txt
//   node scripts/extractParts.js /tmp/d.bin data/corpora/alice_en.txt # explicit path used as-is
//   node scripts/extractParts.js - data/corpora/*.txt | less
//   cat book.txt | node scripts/extractParts.js dict.bin -
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Kind } from "../src/core/parts/kind.js";
import { tokenizeStream, StreamTokenType } from "../src/core/parts/tokenize.js";
import { PartExtractor } from "../src/core/parts/extractor.js";
import { saveDictText, saveDictBinary } from "../src/core/parts/dictionary.js";

// Default output directory for bare filenames: <repo>/data/dictionaries.
const DEFAULT_DICT_DIR = path.resolve(fileURLToPath(import.meta.url), "..", "..", "data", "dictionaries");

const usage = () => {
  process.stderr.write(
    "usage: node scripts/extractParts.js <output.txt|.bin|-> <input.txt|-> [<input> ...]\n",
  );
};

// A bare filename (no directory component) defaults into data/dictionaries/;
// a path with a directory, or an absolute path, is used verbatim.
const resolveOutput = (out) => {
  if (out === "-") return out;
  if (path.dirname(out) === "." && !path.isAbsolute(out)) return path.join(DEFAULT_DICT_DIR, out);
  return out;
};

// Read a file (or stdin for "-") as raw bytes.
const readInputBytes = (p) => new Uint8Array(fs.readFileSync(p === "-" ? 0 : p));

const main = () => {
  const args = process.argv.slice(2);
  if (args.length < 2) {
    usage();
    process.exit(2);
  }
  const outPath = args[0];
  const inPaths = args.slice(1);

  const ex = new PartExtractor();
  let totalTokens = 0;
  let totalWords = 0;
  let totalDelims = 0;

  for (const inPath of inPaths) {
    const bytes = readInputBytes(inPath);
    const tokens = tokenizeStream(bytes);
    let words = 0;
    let delims = 0;
    for (const t of tokens) {
      if (t.type === StreamTokenType.Word) {
        ex.addWord(t.value);
        words++;
      } else {
        ex.addDelimiter(t.value);
        delims++;
      }
    }
    process.stderr.write(`  ${inPath}: ${tokens.length} tokens (${words} words, ${delims} delims)\n`);
    totalTokens += tokens.length;
    totalWords += words;
    totalDelims += delims;
  }

  const dict = ex.finalize();

  const resolvedOut = resolveOutput(outPath);
  const binary = resolvedOut !== "-" && resolvedOut.endsWith(".bin");
  if (resolvedOut === "-") {
    // Text bytes to stdout (latin1 preserves the byte-string exactly).
    process.stdout.write(Buffer.from(saveDictText(dict), "latin1"));
  } else {
    fs.mkdirSync(path.dirname(resolvedOut), { recursive: true });
    if (binary) fs.writeFileSync(resolvedOut, Buffer.from(saveDictBinary(dict)));
    else fs.writeFileSync(resolvedOut, Buffer.from(saveDictText(dict), "latin1"));
  }

  process.stderr.write(`inputs: ${inPaths.length}\n`);
  process.stderr.write(
    `total tokens: ${totalTokens} (words=${totalWords}, delims=${totalDelims})\n`,
  );
  process.stderr.write(
    `dictionary size F = ${dict.size()} (saved as ${binary ? "binary" : "text"}` +
      `${resolvedOut === "-" ? "" : ` -> ${resolvedOut}`}, singletons/specials excluded from file)\n`,
  );
  process.stderr.write(`  whole:  ${dict.countOfKind(Kind.Whole)}\n`);
  process.stderr.write(`  start:  ${dict.countOfKind(Kind.Start)}\n`);
  process.stderr.write(`  mid:    ${dict.countOfKind(Kind.Mid)}\n`);
  process.stderr.write(`  end:    ${dict.countOfKind(Kind.End)}\n`);
  process.stderr.write(`  letter: ${dict.countOfKind(Kind.Letter)}\n`);
  process.stderr.write(`  delim:  ${dict.countOfKind(Kind.Delimiter)}\n`);
};

main();
