"use strict";

/**
 * @file streamSignatures.js
 * @brief Stream a corpus into bounded signature-record chunks — never materialized.
 *
 * The batch builder ({@link module:scripts/buildSignatures}) holds every record in
 * memory before writing, which does not scale: a 535 MB corpus is tens of GB of
 * signatures. This source instead reads the corpus line by line, encodes signatures
 * one sentence at a time, and yields fixed-size record chunks — so the live
 * footprint is one chunk plus the (small) shared vocab, regardless of corpus size.
 * Signatures recompute at ~2 s/MB, cheaper than caching them, so nothing is cached:
 * a forest driver pulls a chunk, grows one tree, keeps the tree, drops the records.
 *
 * Sentence scope only (scale 0): in line-oriented corpora a sentence never spans a
 * newline, so each line is segmented and encoded independently — no cross-chunk
 * sentence can be split. The vocab (word byte-string → id, first-seen) is shared
 * across chunks so labels stay consistent for a later pooled readout; pass one in
 * to continue an existing id space.
 */

import fs from "node:fs";
import readline from "node:readline";

import { tokenizeStream, StreamTokenType } from "../parts/tokenize.js";
import { encode, encodeWindowed, encodePositional } from "./encoder.js";
import { encodeWord, fuzzyEncodeWord } from "./wordEncoder.js";
import { NO_LABEL } from "./signatureDataset.js";
import { segmentText } from "../../utilities/textSegmentation/segmentText.js";

/**
 * @function streamSignatureChunks
 * @description Async generator of `{records, F, vocab}` chunks from a corpus file.
 * Each record is `{scale:0, curWord, nextWord, L, C, R}` (same shape as the batch
 * builder). Chunks carry at most `chunkSize` records; the final chunk may be short.
 *
 * @param {Object} dict - Loaded part dictionary (provides `.size()` → F and encoding).
 * @param {string} corpusPath - Path to a UTF-8 corpus file.
 * @param {Object} [opts]
 * @param {number} [opts.chunkSize=20000] - Max records per yielded chunk.
 * @param {number|null} [opts.radius=null] - Pooling radius; null = full-sentence scope.
 * @param {Map<string,number>} [opts.vocabIndex] - Shared word→id map (grown in place).
 * @param {string[]} [opts.vocab] - Shared id→word list (grown in place).
 * @yields {{records: Object[], F: number, vocab: string[]}}
 */
export async function* streamSignatureChunks(dict, corpusPath, { chunkSize = 20000, radius = null, fuzzy = false, positional = false, vocabIndex = new Map(), vocab = [] } = {}) {
  const F = dict.size();
  const bagOf = fuzzy ? fuzzyEncodeWord : encodeWord;
  const wordId = (v) => {
    let id = vocabIndex.get(v);
    if (id === undefined) { id = vocab.length; vocabIndex.set(v, id); vocab.push(v); }
    return id;
  };

  let batch = [];
  const encodeScope = (text) => {
    const toks = tokenizeStream(text);
    const words = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => t.value);
    if (words.length === 0) return;
    const sigs = positional ? encodePositional(dict, toks, radius, 0, Infinity, bagOf)
      : radius === null ? encode(dict, toks, 0, Infinity, bagOf) : encodeWindowed(dict, toks, radius, 0, Infinity, bagOf);
    for (let i = 0; i < sigs.length; i++) {
      const [L, C, R] = sigs[i];
      batch.push({
        scale: 0,
        pos: i, // target word's index within its sentence (# words to the left)
        curWord: wordId(words[i]),
        nextWord: i + 1 < words.length ? wordId(words[i + 1]) : NO_LABEL,
        L, C, R,
      });
    }
  };

  const rl = readline.createInterface({ input: fs.createReadStream(corpusPath, "utf8"), crlfDelay: Infinity });
  for await (const line of rl) {
    // A sentence never spans a newline in line-oriented corpora, so segment per line.
    for (const seg of segmentText(line)) encodeScope(line.slice(seg.start, seg.end));
    while (batch.length >= chunkSize) {
      yield { records: batch.slice(0, chunkSize), F, vocab };
      batch = batch.slice(chunkSize);
    }
  }
  if (batch.length) yield { records: batch, F, vocab };
}

export default streamSignatureChunks;
