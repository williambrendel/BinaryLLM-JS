"use strict";

/**
 * @file xorCount.js
 * @brief Size of the sparse-set symmetric difference, |A △ B|, without
 * materializing it.
 *
 * A "sparse set" is a sorted, strictly-increasing array of non-negative
 * integers — the sparse representation of a bit vector. Computes
 * `|A| + |B| - 2·|A ∩ B|` by counting shared values during a single linear
 * merge (O(|a| + |b|)); use {@link module:core/math/sparse/xor} when you need
 * the elements. Assumes both inputs are already sorted ascending.
 */

/**
 * @function xorCount
 * @description Number of values present in exactly one of the two sorted sets —
 * the cardinality of the symmetric difference.
 *
 * @param {number[]|Uint16Array|Uint32Array} a - Sorted, strictly-increasing set.
 * @param {number[]|Uint16Array|Uint32Array} b - Sorted, strictly-increasing set.
 * @returns {number} `|A △ B| = |A| + |B| - 2·|A ∩ B|`; `0` when the sets are
 *   identical.
 *
 * @example
 * xorCount([1, 3, 5], [2, 3, 6])   // → 4   (1, 2, 5, 6 — 3 is shared)
 * xorCount([1, 2], [1, 2])         // → 0   (identical)
 */
export const xorCount = (a, b) => {
  const alen = a.length, blen = b.length;
  let i = 0, j = 0, n = 0, ai, bj;
  
  while (i < alen && j < blen) {
    (ai = a[i]) < (bj = b[j]) && ++i
    || (ai > bj && ++j)
    || (++i, ++j, ++n);
  }

  return alen + blen - (n << 1);
}

/**
 * @ignore
 * Default export.
 */
export default xorCount;