"use strict";
// TEMP helper: streaming part extraction over large corpora (line-by-line,
// bounded memory). Feeds one PartExtractor across all inputs, then finalize.
//   node scripts/_streamExtract.mjs <out.txt|.bin> <input> [<input> ...]
import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";
import { PartExtractor } from "../src/core/parts/extractor.js";
import { tokenizeStream, StreamTokenType } from "../src/core/parts/tokenize.js";
import { saveDictText, saveDictBinary } from "../src/core/parts/dictionary.js";
import { Kind } from "../src/core/parts/kind.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveOut = (o) =>
  path.dirname(o) === "." && !path.isAbsolute(o) ? path.join(DICT_DIR, o) : o;

const [outArg, ...inputs] = process.argv.slice(2);
if (!outArg || inputs.length === 0) {
  process.stderr.write("usage: node scripts/_streamExtract.mjs <out> <input...>\n");
  process.exit(2);
}

const ex = new PartExtractor();
let words = 0, delims = 0, lines = 0;
const t0 = Date.now();

for (const inPath of inputs) {
  const rl = readline.createInterface({
    input: fs.createReadStream(inPath, { encoding: "latin1" }),
    crlfDelay: Infinity,
  });
  let first = true;
  for await (const line of rl) {
    if (!first) { ex.addDelimiter("\n"); delims++; }
    first = false;
    for (const t of tokenizeStream(line)) {
      if (t.type === StreamTokenType.Word) { ex.addWord(t.value); words++; }
      else { ex.addDelimiter(t.value); delims++; }
    }
    if ((++lines & 0x3ffff) === 0) {
      process.stderr.write(
        `  ${((Date.now() - t0) / 1000) | 0}s  lines=${lines}  words=${words}  ` +
        `distinct=${ex.distinctWordCount()}\n`);
    }
  }
  process.stderr.write(`  [done reading ${inPath}]\n`);
}

process.stderr.write(
  `feeding done: lines=${lines} words=${words} delims=${delims} ` +
  `distinct=${ex.distinctWordCount()} — finalizing (peel loop)...\n`);

const tFin = Date.now();
const dict = ex.finalize();
process.stderr.write(`finalize: ${((Date.now() - tFin) / 1000) | 0}s, peelIters=${ex.lastPeelIterations()}\n`);

const out = resolveOut(outArg);
fs.mkdirSync(path.dirname(out), { recursive: true });
if (out.endsWith(".bin")) fs.writeFileSync(out, Buffer.from(saveDictBinary(dict)));
else fs.writeFileSync(out, Buffer.from(saveDictText(dict), "latin1"));

process.stderr.write(
  `F=${dict.size()} -> ${out}  ` +
  `(whole=${dict.countOfKind(Kind.Whole)} start=${dict.countOfKind(Kind.Start)} ` +
  `mid=${dict.countOfKind(Kind.Mid)} end=${dict.countOfKind(Kind.End)} ` +
  `letter=${dict.countOfKind(Kind.Letter)} delim=${dict.countOfKind(Kind.Delimiter)})\n`);
