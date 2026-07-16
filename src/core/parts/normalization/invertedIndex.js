"use strict";

/**
 * @file invertedIndex.js
 * @brief Inverted index over the vocabulary's fuzzy signatures — the memory the
 * canonicalizer searches.
 *
 * Maps each part id to the list of word indices whose containment signature F
 * (see {@link module:core/parts/encoding/fuzzyEncode}) includes it. A query then
 * gathers candidate words from the posting lists of its own part ids and scores
 * them by Jaccard (see {@link module:core/parts/normalization/nearestNeighbor}).
 */

/**
 * @typedef {Object} InvertedIndex
 * @property {string[]} words - Vocabulary, indexed by word id.
 * @property {Uint32Array[]} signatures - Each word's fuzzy signature F (sorted ids).
 * @property {number[]} sizes - `signatures[wi].length`, cached for Jaccard.
 * @property {Map<number, number[]>} postings - part id → word ids containing it.
 * @property {Uint32Array[]|null} peels - Optional per-word sparse peel cache: when
 *   supplied, `peels[wi]` is the sparse signature of `words[wi]`, so an NN hit can
 *   return the precomputed peel by `wi` instead of re-peeling the recovered word.
 */

/**
 * @function buildInvertedIndex
 * @description Builds an inverted index from a vocabulary and its precomputed
 * fuzzy signatures. Optionally caches each word's sparse peel so the recovery
 * pipeline (fuzzy F → NN → peel) can look the peel up by word id instead of
 * recomputing it per token.
 *
 * @param {string[]} words - Vocabulary (word types), indexed by position.
 * @param {Uint32Array[]} signatures - `signatures[wi]` is the sorted fuzzy
 *   signature F of `words[wi]` (from {@link fuzzyEncode}).
 * @param {Uint32Array[]} [peels=null] - Optional sparse-peel cache: `peels[wi]`
 *   is the sparse signature of `words[wi]` (from `sparseF`). One peel per word
 *   type, computed once — the NN returns `wi`, so the lookup is O(1).
 * @returns {InvertedIndex} The index (postings + cached sizes [+ peels]).
 *
 * @example
 * const F = words.map((w) => fuzzyEncode(dict, w));
 * const peels = words.map((w) => sparseF(dict, w));   // one peel per word type
 * const index = buildInvertedIndex(words, F, peels);
 * index.peels[index.words.indexOf("running")];        // cached peel, no re-encode
 */
export const buildInvertedIndex = (words, signatures, peels = null) => {
  const sizes = signatures.map((f) => f.length);
  const postings = new Map();
  for (let wi = 0; wi < words.length; wi++) {
    for (const id of signatures[wi]) {
      let a = postings.get(id);
      if (!a) postings.set(id, (a = []));
      a.push(wi);
    }
  }
  return { words, signatures, sizes, postings, peels };
};

export default buildInvertedIndex;
