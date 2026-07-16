"use strict";

/**
 * @file canonicalize.js
 * @brief Resolves a (possibly noisy / OOV) word to a canonical known word.
 *
 * Encodes the word's fuzzy signature F, finds its nearest neighbour in the
 * vocabulary index, and applies the **confidence gate**: if the match is close
 * enough (`J ≥ theta`) the word snaps to that known word; otherwise it is kept
 * as-is (genuine OOV). In-vocabulary words self-match at `J = 1` and are
 * therefore always kept as themselves — the gate only governs unseen inputs.
 */

import { fuzzyEncode } from "../encoding/fuzzyEncode.js";
import { nearestNeighbor } from "./nearestNeighbor.js";

/**
 * @typedef {Object} Canonicalization
 * @property {string} canonical - The resolved word (a known word, or the input).
 * @property {number} wi - Nearest word id (`-1` if nothing matched).
 * @property {number} j - Nearest-neighbour Jaccard similarity.
 * @property {boolean} changed - Whether the input was snapped to a different word.
 */

/**
 * @function canonicalize
 * @description Snaps `word` to its nearest known word when the fuzzy match
 * clears the gate, else returns it unchanged.
 *
 * @param {string} word - Input word (byte-string).
 * @param {import("../dictionary.js").PartDictionary} dict - Dictionary used to
 *   compute the fuzzy signature (must match the one used to build `index`).
 * @param {import("./invertedIndex.js").InvertedIndex} index - Vocabulary index.
 * @param {Object} [opts]
 * @param {number} [opts.theta=0.5] - Gate: minimum Jaccard to canonicalize.
 * @returns {Canonicalization}
 *
 * @example
 * canonicalize("runing", dict, index);          // → { canonical:"running", changed:true, j:0.74 }
 * canonicalize("qwerty", dict, index);          // → { canonical:"qwerty",  changed:false, j:0.41 } (below θ)
 * canonicalize("running", dict, index);         // → { canonical:"running", changed:false, j:1.0 }  (self-match)
 */
export const canonicalize = (word, dict, index, opts = {}) => {
  const theta = opts.theta ?? 0.5;
  const F = fuzzyEncode(dict, word);
  const { wi, j } = nearestNeighbor(F, index);
  if (wi >= 0 && j >= theta) {
    const canonical = index.words[wi];
    return { canonical, wi, j, changed: canonical !== word };
  }
  return { canonical: word, wi, j, changed: false };
};

export default canonicalize;
