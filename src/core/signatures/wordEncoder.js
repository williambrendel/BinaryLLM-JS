"use strict";

// ============================================================================
// core/signatures/wordEncoder.js
//
// Port of core/signatures/word_encoder.cpp — encode a word into its "Option B
// bag": the sorted-unique set of part IDs from decomposing the word against
// the dictionary. This is the `current` (C) band of a signature.
//
// One algorithm; change decomposition behavior by REORDERING THE DICT, not by
// touching the encoder. `word` is assumed to be a lowercased ASCII
// byte-string (the tokenizer lowercases at training time).
// ============================================================================

import { decomposeWord } from "../parts/decomposer.js";

/**
 * Encode a word into its Option B bag (sorted-unique part IDs in [0, F)).
 * @param {import("../parts/dictionary.js").PartDictionary} dict
 * @param {string} word byte-string.
 * @returns {number[]} sorted, de-duplicated part IDs.
 */
export const encodeWord = (dict, word) => {
  const uniq = Array.from(new Set(decomposeWord(dict, word)));
  uniq.sort((a, b) => a - b);
  return uniq;
};

export default encodeWord;
