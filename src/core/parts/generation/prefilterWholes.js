"use strict";

/**
 * @file prefilterWholes.js
 * @brief Track 1 of part generation — whole-word atoms.
 *
 * Two behaviours by word length:
 * - **1–2 letter words** produce no Start/Mid/End candidate (Start/End need
 *   `len > L`, Mid needs `len ≥ L+2`), so the tree can never split them — they
 *   are taken **whole, all of them**, or they would fall back to letters.
 * - **3+ letter words** contribute their frequency head only, gated by
 *   {@link elbowEffectiveCount}: a clean head → take it; a flat/linear
 *   distribution → skip that length. This captures the common function words
 *   without atomizing the long tail.
 *
 * Words taken as wholes are flagged in `removed` and excluded from the affix
 * tracks (their substrings are still available to other words).
 */

import { Kind } from "../kind.js";
import { elbowEffectiveCount } from "../../math/adaptiveThreshold.js";

/**
 * @function prefilterWholes
 * @description Adds whole-word atoms to `dict`, marking the words taken.
 *
 * @param {string[]} words - Vocabulary, indexed by position.
 * @param {Float64Array|number[]} freq - Token frequency per word index.
 * @param {import("../dictionary.js").PartDictionary} dict - Dictionary to add to.
 * @param {Uint8Array} removed - Per-word flag, set to `1` when taken (mutated).
 * @param {Object} [opts]
 * @param {number} [opts.lMax=7] - Longest word length considered.
 * @param {number} [opts.shortWholeMax=2] - Lengths ≤ this are taken in full.
 * @param {number} [opts.elbowSig=0.01] - Elbow significance for the length heads.
 * @returns {number} Count of whole-word atoms added.
 *
 * @example
 * const removed = new Uint8Array(words.length);
 * prefilterWholes(words, freq, dict, removed);   // "mr","the",… become Wholes
 */
export const prefilterWholes = (words, freq, dict, removed, opts = {}) => {
  const { lMax = 7, shortWholeMax = 2, elbowSig = 0.01 } = opts;
  let n = 0;
  for (let L = lMax; L >= 1; L--) {
    const bucket = [];
    for (let wi = 0; wi < words.length; wi++) if (words[wi].length === L) bucket.push(wi);
    if (!bucket.length) continue;
    bucket.sort((a, b) => freq[b] - freq[a] || (words[a] < words[b] ? -1 : 1));
    if (L <= shortWholeMax) {
      for (const wi of bucket) { dict.add(Kind.Whole, words[wi]); removed[wi] = 1; n++; }
    } else {
      const k = elbowEffectiveCount(bucket.map((wi) => freq[wi]), elbowSig);
      if (k < bucket.length) for (let i = 0; i < k; i++) { dict.add(Kind.Whole, words[bucket[i]]); removed[bucket[i]] = 1; n++; }
    }
  }
  return n;
};

export default prefilterWholes;
