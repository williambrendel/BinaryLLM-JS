"use strict";

/**
 * @file orCount.js
 * @brief Size of the sparse-set union, |A ∪ B|, without materializing it.
 *
 * A "sparse set" is a sorted, strictly-increasing array of non-negative
 * integers — the sparse representation of a bit vector. Counts distinct values
 * across both inputs during a single linear merge (O(|a| + |b|)) without
 * allocating a result array; use {@link module:core/math/sparse/or} when you
 * need the elements. Assumes both inputs are already sorted ascending.
 */

/**
 * @function orCount
 * @description Number of distinct values present in EITHER sorted set — the
 * cardinality of the union (shared values counted once).
 *
 * @param {number[]|Uint16Array|Uint32Array} a - Sorted, strictly-increasing set.
 * @param {number[]|Uint16Array|Uint32Array} b - Sorted, strictly-increasing set.
 * @returns {number} `|A ∪ B|`; equals `|A| + |B|` when the sets are disjoint.
 *
 * @example
 * orCount([1, 3, 5], [2, 3, 6])   // → 5   (1, 2, 3, 5, 6)
 * orCount([1, 3], [2, 4])         // → 4   (disjoint)
 */
export const orCount = (a, b) => {
  const alen = a.length, blen = b.length;
  let i = 0, j = 0, n = 0, ai, bj;
  
  while (i < alen && j < blen) {
    ++n;
    (ai = a[i]) < (bj = b[j]) && ++i
    || (ai > bj && ++j)
    || (++i, ++j);
  }

  return n + alen - i + blen - j;
}

/**
 * @ignore
 * Default export.
 */
export default orCount;