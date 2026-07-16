"use strict";

/**
 * @file andNotCount.js
 * @brief Size of the sparse-set difference, |A ∖ B|, without materializing it.
 *
 * A "sparse set" is a sorted, strictly-increasing array of non-negative
 * integers — the sparse representation of a bit vector. Counts the values
 * present in `a` but absent from `b` (equivalently `|a AND NOT b|`) during a
 * single linear merge (O(|a| + |b|)) without allocating a result array; use
 * {@link module:core/math/sparse/andNot} when you need the elements themselves.
 * Assumes both inputs are already sorted ascending.
 *
 * This is the gate-coverage primitive: `andNotCount(x, ¬M_τ)` is `|M_τ ∧ x|`,
 * the denominator of the signed-threshold score, computed straight from the
 * small stored drop-set `¬M_τ` without ever touching the kept bits.
 *
 * @see {@link module:core/math/sparse/andNot}  the materialized difference
 * @see {@link module:core/math/sparse/andCount}  intersection cardinality
 */

/**
 * @function andNotCount
 * @description Number of values present in `a` but NOT in `b` — the cardinality
 * of the relative complement `A ∖ B`.
 *
 * @param {number[]|Uint16Array|Uint32Array} a - Sorted, strictly-increasing set
 *   (the values to keep from).
 * @param {number[]|Uint16Array|Uint32Array} b - Sorted, strictly-increasing set
 *   (the drop-set to subtract).
 * @returns {number} `|A ∖ B|`; equals `a.length` when the sets are disjoint,
 *   `0` when `a ⊆ b`.
 *
 * @example
 * andNotCount([1, 2, 3, 5], [2, 3, 4])   // → 2   (1, 5)
 * andNotCount([1, 2, 3], [1, 2, 3, 4])   // → 0   (a ⊆ b)
 * andNotCount([1, 3, 5], [2, 4, 6])      // → 3   (disjoint → all of a)
 */
export const andNotCount = (a, b) => {
  const alen = a.length, blen = b.length;
  let i = 0, j = 0, n = 0, ai, bj;

  // `a[i] < b[j]` → a[i] is absent from b (count it); equality drops both;
  // `a[i] > b[j]` advances b only. Every branch ends on an always-truthy
  // increment so the `&&`/`||` chain never mis-selects on a zero value.
  while (i < alen && j < blen) {
    (ai = a[i]) < (bj = b[j]) && (++i, ++n)
    || (ai > bj && ++j)
    || (++i, ++j);
  }

  return n + (alen - i);
}

/**
 * @ignore
 * Default export.
 */
export default andNotCount;
