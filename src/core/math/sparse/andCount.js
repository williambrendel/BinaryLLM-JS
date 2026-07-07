"use strict";

/**
 * @file andCount.js
 * @brief Size of the sparse-set intersection, |A ∩ B|, without materializing it.
 *
 * A "sparse set" is a sorted, strictly-increasing array of non-negative
 * integers — the sparse representation of a bit vector. Counts matches during a
 * single linear merge (O(|a| + |b|)) without allocating a result array; use
 * {@link module:core/math/sparse/and} when you need the elements themselves.
 * Assumes both inputs are already sorted ascending.
 */

/**
 * @function andCount
 * @description Number of values present in BOTH sorted sets — the cardinality
 * of the intersection.
 *
 * @param {number[]|Uint16Array|Uint32Array} a - Sorted, strictly-increasing set.
 * @param {number[]|Uint16Array|Uint32Array} b - Sorted, strictly-increasing set.
 * @returns {number} `|A ∩ B|`; `0` when the sets are disjoint.
 *
 * @example
 * andCount([1, 2, 3, 5], [2, 3, 4, 5])   // → 3   (2, 3, 5)
 * andCount([1, 3, 5], [2, 4, 6])         // → 0   (disjoint)
 */
export const andCount = (a, b) => {
  const alen = a.length, blen = b.length;
  let i = 0, j = 0, n = 0, ai, bj;
  
  while (i < alen && j < blen) {
    (ai = a[i]) < (bj = b[j]) && ++i
    || (ai > bj && ++j)
    || (++i, ++j, ++n);
  }

  return n;
}

/**
 * @ignore
 * Default export.
 */
export default andCount;