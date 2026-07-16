"use strict";

/**
 * @file xor.js
 * @brief Sparse-set symmetric difference (A △ B) over two sorted index arrays.
 *
 * A "sparse set" is a sorted, strictly-increasing array of non-negative
 * integers — the sparse representation of a bit vector. The symmetric
 * difference keeps values present in exactly one input and drops shared ones.
 * Runs in O(|a| + |b|) and assumes both inputs are already sorted ascending.
 *
 * @see {@link module:core/math/sparse/and}  intersection
 * @see {@link module:core/math/sparse/or}   union
 */

/**
 * @function xor
 * @description Symmetric difference of two sorted index sets — the values in
 * exactly one of `a` or `b`, dropping any they share — returned in ascending
 * order.
 *
 * The result is written into `out` when supplied and large enough
 * (`out.length >= |a| + |b|`, the maximum possible size); otherwise a fresh
 * typed array is allocated. The element type is `Uint32Array` when either input
 * is a `Uint32Array`, and `Uint16Array` otherwise. The return value is a
 * `.subarray` view trimmed to the result size, so `out` — when provided — must
 * be a typed array.
 *
 * @param {number[]|Uint16Array|Uint32Array} a - Sorted, strictly-increasing set.
 * @param {number[]|Uint16Array|Uint32Array} b - Sorted, strictly-increasing set.
 * @param {Uint16Array|Uint32Array} [out] - Optional scratch buffer, reused when
 *   `out.length >= a.length + b.length`.
 * @returns {Uint16Array|Uint32Array} Ascending view of the non-shared elements;
 *   empty when the sets are identical.
 *
 * @example
 * Array.from(xor([1, 3, 5], [2, 3, 6]))   // → [1, 2, 5, 6]  (3 is shared, dropped)
 * Array.from(xor([1, 2], [1, 2]))         // → []            (identical)
 */
export const xor = (a, b, out) => {
  const alen = a.length, blen = b.length;
  let i = 0, j = 0, k = 0, ai, bj, n = alen + blen;

  out && out.length >= n || (
    out = ((a instanceof Uint32Array) || (b instanceof Uint32Array)) && new Uint32Array(n)
    || new Uint16Array(n)
  );
  
  // NB: the emitting branches end their comma-sequence on `++i`/`++j` (always
  // truthy), NOT on `out[k++]=v` — v can be 0 (part id 0 is the first dict
  // entry) and a falsy result would fall through to the next `||` and
  // double-process. Ordering the increment last keeps the `&&`/`||` chain sound.
  while (i < alen && j < blen) {
    (ai = a[i]) < (bj = b[j]) && (
      out[k++] = ai, ++i
    ) || (
      ai > bj && (
        out[k++] = bj,
        ++j
      )
    ) || (++i, ++j);
  }

  while (i < alen) out[k++] = a[i++];

  while (j < blen) out[k++] = b[j++];

  return out.subarray(0, k);
}

/**
 * @ignore
 * Default export.
 */
export default xor;