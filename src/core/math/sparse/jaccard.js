"use strict";

/**
 * @file jaccard.js
 * @brief Jaccard similarity, |A ∩ B| / |A ∪ B|, over two sparse index sets.
 *
 * A "sparse set" is a sorted, strictly-increasing array of non-negative
 * integers — the sparse representation of a bit vector. Derives both the
 * intersection and union counts from a single linear merge (O(|a| + |b|)) and
 * assumes both inputs are already sorted ascending.
 */

/**
 * @function jaccard
 * @description Jaccard similarity coefficient of two sorted index sets:
 * `|A ∩ B| / |A ∪ B|`, a value in `[0, 1]` measuring overlap. `1` means the
 * sets are equal, `0` means disjoint.
 *
 * Two empty sets have an undefined ratio (`0 / 0`) and return `NaN`; guard for
 * that case at the call site if empty inputs are possible.
 *
 * @param {number[]|Uint16Array|Uint32Array} a - Sorted, strictly-increasing set.
 * @param {number[]|Uint16Array|Uint32Array} b - Sorted, strictly-increasing set.
 * @returns {number} Similarity in `[0, 1]`, or `NaN` when both sets are empty.
 *
 * @example
 * jaccard([1, 3, 5], [2, 3, 6])   // → 0.2   (1 shared / 5 total)
 * jaccard([1, 2, 3], [1, 2, 3])   // → 1     (identical)
 * jaccard([1, 2], [3, 4])         // → 0     (disjoint)
 */
export const jaccard = (a, b) => {
  const alen = a.length, blen = b.length;
  let i = 0, j = 0, n = 0, ai, bj;
  
  while (i < alen && j < blen) {
    (ai = a[i]) < (bj = b[j]) && ++i
    || (ai > bj && ++j)
    || (++i, ++j, ++n);
  }

  return n / (alen + blen - n);
}

/**
 * @ignore
 * Default export.
 */
export default jaccard;