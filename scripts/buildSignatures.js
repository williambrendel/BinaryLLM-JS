"use strict";

// ============================================================================
// scripts/buildSignatures.js
//
// Build a signature dataset from a corpus — the JS analog of the C++
// build_candidates. For each context scope, extract its text, tokenize,
// encode 3F signatures, and emit one record per Word token: the three bands
// [L, C, R] plus the current-word and next-word labels and a scale tag.
//
// Context scopes come from utilities/textSegmentation:
//   - sentence (segmentText)        scale = 0
//   - paragraph (segmentTextSections) scale = 1
// The scope bounds the L/R context (like build_candidates' sentence/paragraph
// scales); pass --radius to further bound pooling within a scope.
//
// Output is the SIG1 binary (see core/signatures/signatureDataset.js). A bare
// output filename lands in data/signatures/; a bare dict name resolves from
// data/dictionaries/.
//
// Usage:
//   node scripts/buildSignatures.js <dict.txt|.bin> <out.bin> <corpus...>
//     [--scale both|sentence|paragraph] [--radius R]
//
// Examples:
//   node scripts/buildSignatures.js dict.txt sigs.bin data/corpora/alice_en.txt
//   node scripts/buildSignatures.js dict.bin sigs.bin data/corpora/*.txt --scale sentence
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { tokenizeStream, StreamTokenType } from "../src/core/parts/tokenize.js";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { encode, encodeWindowed } from "../src/core/signatures/encoder.js";
import { writeDataset, NO_LABEL } from "../src/core/signatures/signatureDataset.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";
import { segmentTextSections } from "../src/utilities/textSegmentation/segmentTextSections.js";
import Section from "../src/utilities/textSegmentation/Section.js";
import Segment from "../src/utilities/textSegmentation/Segment.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const OUT_DIR = path.join(REPO, "data", "signatures");

const usage = () => {
  process.stderr.write(
    "usage: node scripts/buildSignatures.js <dict.txt|.bin> <out.bin> <corpus...> " +
      "[--scale both|sentence|paragraph] [--radius R]\n",
  );
};

const resolveIn = (p, dir) =>
  path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(dir, p) : p;

const loadDict = (arg) => {
  const p = resolveIn(arg, DICT_DIR);
  return p.endsWith(".bin")
    ? loadDictBinary(new Uint8Array(fs.readFileSync(p)))
    : loadDictText(fs.readFileSync(p, "latin1"));
};

// Collect leaf paragraph sections (those whose elements are all Segments) from
// the segmentTextSections tree; recurse through header-nested sections.
const collectParagraphs = (node, out) => {
  const childSections = node.filter((el) => el instanceof Section);
  if (childSections.length === 0) {
    if (node.length && node.some((el) => el instanceof Segment)) out.push(node);
  } else {
    for (const s of childSections) collectParagraphs(s, out);
  }
};

const main = () => {
  const args = process.argv.slice(2);
  const positional = [];
  let scale = "both";
  let radius = null;
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--scale") scale = args[++i];
    else if (a === "--radius") radius = parseInt(args[++i], 10);
    else positional.push(a);
  }
  if (positional.length < 3) {
    usage();
    process.exit(2);
  }
  const [dictArg, outArg, ...corpora] = positional;
  const wantSentence = scale === "both" || scale === "sentence";
  const wantParagraph = scale === "both" || scale === "paragraph";

  const dict = loadDict(dictArg);
  const F = dict.size();

  // Vocabulary: word byte-string -> id (first-seen order), shared across scopes.
  const vocabIndex = new Map();
  const vocab = [];
  const wordId = (v) => {
    let id = vocabIndex.get(v);
    if (id === undefined) {
      id = vocab.length;
      vocabIndex.set(v, id);
      vocab.push(v);
    }
    return id;
  };

  const records = [];
  const counts = { sentence: 0, paragraph: 0, records: 0 };

  // Encode one scope's text into per-word records tagged with `scaleId`.
  const addScope = (text, scaleId) => {
    const toks = tokenizeStream(text);
    const words = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => t.value);
    if (words.length === 0) return;
    const sigs = radius === null ? encode(dict, toks) : encodeWindowed(dict, toks, radius);
    for (let i = 0; i < sigs.length; i++) {
      const [L, C, R] = sigs[i];
      records.push({
        scale: scaleId,
        curWord: wordId(words[i]),
        nextWord: i + 1 < words.length ? wordId(words[i + 1]) : NO_LABEL,
        L,
        C,
        R,
      });
    }
    counts.records += sigs.length;
  };

  for (const corpusArg of corpora) {
    const text = corpusArg === "-" ? fs.readFileSync(0, "utf8") : fs.readFileSync(corpusArg, "utf8");

    if (wantSentence) {
      const segs = segmentText(text);
      counts.sentence += segs.length;
      for (const seg of segs) addScope(text.slice(seg.start, seg.end), 0);
    }
    if (wantParagraph) {
      const paras = [];
      collectParagraphs(segmentTextSections(text), paras);
      counts.paragraph += paras.length;
      for (const p of paras) addScope(text.slice(p.start, p.end), 1);
    }
    process.stderr.write(`  ${corpusArg}: ${text.length} chars\n`);
  }

  const outPath = resolveIn(outArg, OUT_DIR);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, Buffer.from(writeDataset({ F, vocab, records })));

  process.stderr.write(`scopes: ${counts.sentence} sentence, ${counts.paragraph} paragraph\n`);
  process.stderr.write(
    `dataset: N=${counts.records} records, F=${F}, V=${vocab.length}` +
      `, bands=[L,C,R], scale=${scale}${radius === null ? "" : `, radius=${radius}`} -> ${outPath}\n`,
  );
};

main();
