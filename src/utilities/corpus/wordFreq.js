"use strict";

/**
 * @file wordFreq.js
 * @brief Tokenize one or more byte sources into a word → token-count map.
 *
 * A small reusable corpus helper shared by the part generator and the
 * normalization index. Delimiters are ignored; only {@link Kind}-agnostic Word
 * tokens are counted, keyed by their (lowercased) byte-string value.
 */

import { tokenizeStream, StreamTokenType } from "../../core/parts/tokenize.js";

/**
 * @function wordFreq
 * @description Counts word-token frequencies across the given byte sources.
 *
 * @param {Iterable<Uint8Array|string>} sources - Raw corpus buffers (as read
 *   from files) or strings; each is tokenized independently.
 * @returns {Map<string, number>} Map from distinct word value to its total
 *   occurrence count across all sources.
 *
 * @example
 * const freq = wordFreq([fs.readFileSync("alice.txt")]);
 * freq.get("the");   // → number of "the" occurrences
 */
export const wordFreq = (sources) => {
  const freq = new Map();
  for (const src of sources)
    for (const t of tokenizeStream(src))
      if (t.type === StreamTokenType.Word) freq.set(t.value, (freq.get(t.value) || 0) + 1);
  return freq;
};

/**
 * @ignore
 */
export default wordFreq;
