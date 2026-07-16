"use strict";

/**
 * @file and.js
 * @brief Sparse-set intersection (A ∩ B) over two sorted index arrays.
 *
 * A "sparse set" here is a sorted, strictly-increasing array of non-negative
 * integers — the sparse representation of a bit vector, where each value is the
 * position of a set bit. Every sparse op walks both inputs once in a linear
 * merge (O(|a| + |b|)) and assumes the inputs are already sorted ascending.
 *
 * @see {@link module:core/math/sparse/or}   union
 * @see {@link module:core/math/sparse/xor}  symmetric difference
 */

/**
 * @function and
 * @description Intersection of two sorted index sets — the values present in
 * BOTH `a` and `b` — returned in ascending order.
 *
 * The result is written into `out` when it is supplied and large enough
 * (`out.length >= min(|a|, |b|)`, the maximum possible intersection size);
 * otherwise a fresh typed array is allocated. The element type is `Uint32Array`
 * when either input is a `Uint32Array`, and `Uint16Array` otherwise. The return
 * value is always a `.subarray` view trimmed to the match count, so `out` — when
 * provided — must be a typed array.
 *
 * @param {number[]|Uint16Array|Uint32Array} a - Sorted, strictly-increasing set.
 * @param {number[]|Uint16Array|Uint32Array} b - Sorted, strictly-increasing set.
 * @param {Uint16Array|Uint32Array} [out] - Optional scratch buffer, reused when
 *   `out.length >= min(a.length, b.length)` to avoid per-call allocation.
 * @returns {Uint16Array|Uint32Array} Ascending view of the common elements;
 *   empty when the sets are disjoint.
 *
 * @example
 * Array.from(and([1, 2, 3, 5], [2, 3, 4, 5]))   // → [2, 3, 5]
 * Array.from(and([1, 3, 5], [2, 4, 6]))         // → []  (disjoint)
 *
 * @example
 * // Reuse a scratch buffer in a hot loop to avoid allocation.
 * const out = new Uint32Array(1024);
 * const hits = and(postingsA, postingsB, out);  // view onto `out`
 */
export const and = (a, b, out) => {
  const alen = a.length, blen = b.length;
  let i = 0, j = 0, k = 0, ai, bj, n = Math.min(alen, blen);

  out && out.length >= n || (
    out = ((a instanceof Uint32Array) || (b instanceof Uint32Array)) && new Uint32Array(n)
    || new Uint16Array(n)
  );
  
  while (i < alen && j < blen) {
    (ai = a[i]) < (bj = b[j]) && ++i
    || (ai > bj && ++j)
    || (out[k++] = ai, ++i, ++j);
  }

  return out.subarray(0, k);
}

/**
 * @ignore
 * Default export.
 */
export default and;