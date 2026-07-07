"use strict";

/**
 * @file or.js
 * @brief Sparse-set union (A ∪ B) over two sorted index arrays.
 *
 * A "sparse set" is a sorted, strictly-increasing array of non-negative
 * integers — the sparse representation of a bit vector. This is a sorted merge
 * with de-duplication: shared values are emitted once. Runs in O(|a| + |b|)
 * and assumes both inputs are already sorted ascending.
 *
 * @see {@link module:core/math/sparse/and}  intersection
 * @see {@link module:core/math/sparse/xor}  symmetric difference
 */

/**
 * @function or
 * @description Union of two sorted index sets — every value present in either
 * `a` or `b`, each appearing once — returned in ascending order.
 *
 * The result is written into `out` when supplied and large enough
 * (`out.length >= |a| + |b|`, the maximum possible union size); otherwise a
 * fresh typed array is allocated. The element type is `Uint32Array` when either
 * input is a `Uint32Array`, and `Uint16Array` otherwise. The return value is a
 * `.subarray` view trimmed to the union size, so `out` — when provided — must be
 * a typed array.
 *
 * @param {number[]|Uint16Array|Uint32Array} a - Sorted, strictly-increasing set.
 * @param {number[]|Uint16Array|Uint32Array} b - Sorted, strictly-increasing set.
 * @param {Uint16Array|Uint32Array} [out] - Optional scratch buffer, reused when
 *   `out.length >= a.length + b.length`.
 * @returns {Uint16Array|Uint32Array} Ascending, de-duplicated view of the union.
 *
 * @example
 * Array.from(or([1, 3, 5], [2, 3, 6]))   // → [1, 2, 3, 5, 6]  (3 merged once)
 * Array.from(or([1, 2], []))             // → [1, 2]
 */
export const or = (a, b, out) => {
  const alen = a.length, blen = b.length;
  let i = 0, j = 0, k = 0, ai, bj, n = alen + blen;

  out && out.length >= n || (
    out = ((a instanceof Uint32Array) || (b instanceof Uint32Array)) && new Uint32Array(n)
    || new Uint16Array(n)
  );
  
  while (i < alen && j < blen) {
    (ai = a[i]) < (bj = b[j]) && (
      ++i, out[k++] = ai
    ) || (
      ai > bj && (
        ++j,
        out[k++] = bj
      )
    ) || (++i, ++j, out[k++] = ai);
  }

  while (i < alen) out[k++] = a[i++];

  while (j < blen) out[k++] = b[j++];

  return out.subarray(0, k);
}

/**
 * @ignore
 * Default export.
 */
export default or;