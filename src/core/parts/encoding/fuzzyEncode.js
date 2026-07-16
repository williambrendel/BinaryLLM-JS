"use strict";

/**
 * @file fuzzyEncode.js
 * @brief The fuzzy (containment) word encoder — the typo-robust NN key.
 *
 * Unlike the peel ({@link module:core/parts/encoding/bpeEncode}), which produces
 * a non-redundant segmentation, `fuzzyEncode` returns the **containment
 * signature** F: the ids of *every* registered part the word contains
 * (overlapping prefixes, suffixes, interior substrings, and positional
 * letters). Redundancy is the point — a single-character edit only perturbs the
 * ids whose span overlaps it, so Jaccard over F degrades gracefully. This F is
 * the key indexed by the normalizer (see
 * {@link module:core/parts/normalization/invertedIndex}).
 */

import { Kind, kInvalidPartId } from "../kind.js";

const INV = kInvalidPartId;

/**
 * @function fuzzyEncode
 * @description Computes the containment signature of `word`: the sorted,
 * de-duplicated part ids of every registered *strict fragment* it contains — the
 * Whole (if any), every prefix (length 1..n-1) registered as a Start, every such
 * suffix as an End, and every interior substring as a Mid. Single characters are
 * simply the length-1 Start/Mid/End atoms; a 1-char word encodes to just its Whole.
 *
 * The result is a sorted `Uint32Array`, ready for the sparse set ops
 * (`and`/`or`/`xor`/`jaccard`) used in nearest-neighbour search.
 *
 * @param {import("../dictionary.js").PartDictionary} dict - Dictionary to look
 *   parts up in (typically seeded with {@link addBigramBackstop} for char-level
 *   granularity).
 * @param {string} word - The word to encode, as a byte-string.
 * @returns {Uint32Array} Ascending, de-duplicated part ids the word contains.
 *
 * @example
 * const F = fuzzyEncode(dict, "running");
 * // → sorted ids for whichever of run/unn/ing/nn/.. and letters are registered
 * jaccard(fuzzyEncode(dict, "running"), fuzzyEncode(dict, "runing")); // high
 */
export const fuzzyEncode = (dict, word) => {
  const n = word.length;
  const ids = new Set();
  if (dict.hasWhole(word)) ids.add(dict.lookup(Kind.Whole, word));
  // Strict fragments only: lengths 1 .. n-1. The full-length substring (L === n)
  // IS the word — it is the Whole above, never a Start/Mid/End. So a single-char
  // word has no fragments and encodes to just its Whole.
  //
  // Each occurrence contributes to EXACTLY ONE kind by position — a prefix is a
  // Start, a suffix is an End, only strictly-interior occurrences are Mids. So
  // `Start "ca"` does not also fire `Mid "ca"`, and a single char fires one of
  // Start/Mid/End (position-specific, like the old Letter). The redundancy that
  // makes F typo-robust comes from *length overlap* (a char sits under parts of
  // several lengths), not from double-counting a boundary as both Start and Mid.
  for (let L = 1; L < n; L++) {
    let id = dict.lookup(Kind.Start, word.slice(0, L)); if (id !== INV) ids.add(id);
    id = dict.lookup(Kind.End, word.slice(n - L, n)); if (id !== INV) ids.add(id);
    for (let p = 1; p + L < n; p++) { id = dict.lookup(Kind.Mid, word.slice(p, p + L)); if (id !== INV) ids.add(id); }
  }
  return Uint32Array.from([...ids].sort((a, b) => a - b));
};

export default fuzzyEncode;
