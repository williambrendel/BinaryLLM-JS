"use strict";

// ============================================================================
// scripts/decompose.js
//
// Load a part dictionary and decompose text into its parts — the inference
// side of "get the parts". Tokenizes the input the same way the extractor
// did, then decomposes each token: words via the greedy peel, delimiters
// literally.
//
// Dictionary format is chosen by extension (.bin -> binary, else text).
//
// Usage:
//   node scripts/decompose.js <dict.txt|.bin> <text...>
//   node scripts/decompose.js <dict.txt|.bin> -            # read text from stdin
//
// Examples:
//   node scripts/decompose.js data/dict.txt "unhappily running"
//   echo "the peppers" | node scripts/decompose.js data/dict.bin -
// ============================================================================

import fs from "node:fs";
import { Kind, kindToString } from "../src/core/parts/kind.js";
import { fromByteString } from "../src/core/parts/byteString.js";
import { tokenizeStream, StreamTokenType } from "../src/core/parts/tokenize.js";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { decomposeWord, decomposeDelimiter } from "../src/core/parts/decomposer.js";

const usage = () => {
  process.stderr.write("usage: node scripts/decompose.js <dict.txt|.bin> <text...|->\n");
};

const loadDict = (p) => {
  if (p.endsWith(".bin")) return loadDictBinary(new Uint8Array(fs.readFileSync(p)));
  // Read text dictionaries as latin1 so the on-disk bytes survive intact.
  return loadDictText(fs.readFileSync(p, "latin1"));
};

// Render one part id as "kind:value" (value decoded from its byte-string).
const renderPart = (dict, id) => {
  const { kind, value } = dict.at(id);
  return `${kindToString(kind)}:${fromByteString(value)}`;
};

const main = () => {
  const args = process.argv.slice(2);
  if (args.length < 2) {
    usage();
    process.exit(2);
  }
  const dict = loadDict(args[0]);

  // Remaining args are the text; "-" means read stdin.
  const rest = args.slice(1);
  const raw =
    rest.length === 1 && rest[0] === "-"
      ? new Uint8Array(fs.readFileSync(0))
      : rest.join(" ");

  for (const tok of tokenizeStream(raw)) {
    const isWord = tok.type === StreamTokenType.Word;
    const ids = isWord ? decomposeWord(dict, tok.value) : decomposeDelimiter(dict, tok.value);
    const label = isWord ? "word" : "delim";
    const parts = ids.map((id) => renderPart(dict, id)).join("  ");
    process.stdout.write(`${label}  ${JSON.stringify(fromByteString(tok.value))}  ->  [${ids.join(", ")}]  ${parts}\n`);
  }
};

main();
