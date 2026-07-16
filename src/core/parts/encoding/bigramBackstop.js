"use strict";

/**
 * @file bigramBackstop.js
 * @brief Seeds the fixed 676-bigram positional backstop into a dictionary.
 *
 * The encoder's coverage layer at length 2 is *not* learned — it is the full,
 * corpus-independent set of `26×26` lowercase bigrams, registered under all
 * three affix positions (Start/Mid/End). With this present, any word interior
 * tiles into bigrams and only an odd-length leftover falls to a single Letter.
 * See {@link module:core/parts/encoding/bpeEncode}.
 */

import { Kind } from "../kind.js";

/**
 * @function addBigramBackstop
 * @description Adds every lowercase 2-letter combination (`aa`…`zz`, 676 of
 * them) to `dict` as a Start, Mid, and End part. Idempotent (relies on
 * {@link PartDictionary#add} deduping), so it is safe to call on an already
 * populated dictionary.
 *
 * @param {import("../dictionary.js").PartDictionary} dict - Dictionary to seed
 *   in place.
 * @returns {import("../dictionary.js").PartDictionary} The same `dict`, for chaining.
 *
 * @example
 * addBigramBackstop(dict);
 * dict.hasMid("th");    // → true
 * dict.hasStart("qu");  // → true
 */
export const addBigramBackstop = (dict) => {
  for (let a = 97; a <= 122; a++) for (let b = 97; b <= 122; b++) {
    const s = String.fromCharCode(a) + String.fromCharCode(b);
    dict.add(Kind.Start, s);
    dict.add(Kind.Mid, s);
    dict.add(Kind.End, s);
  }
  return dict;
};

export default addBigramBackstop;
