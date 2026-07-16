"use strict";

/**
 * @file andNot.js
 * @brief Sparse-set difference (A ∖ B) over two sorted index arrays.
 *
 * A "sparse set" is a sorted, strictly-increasing array of non-negative
 * integers — the sparse representation of a bit vector, where each value is the
 * position of a set bit. The difference keeps the values present in `a` but
 * absent from `b`; equivalently `a AND NOT b`. Runs in O(|a| + |b|) and assumes
 * both inputs are already sorted ascending.
 *
 * This is the mask-application primitive: to apply a keep-mask `M_τ` to a
 * context `x`, you never enumerate the (huge) kept set — you store its small
 * complement `¬M_τ` (the drop-set) and compute `andNot(x, ¬M_τ)`. The cost is
 * O(|x| + |¬M_τ|), independent of how many bits `M_τ` keeps.
 *
 * @see {@link module:core/math/sparse/and}  intersection
 * @see {@link module:core/math/sparse/xor}  symmetric difference
 * @see {@link module:core/math/sparse/andNotCount}  its cardinality
 */

/**
 * @function andNot
 * @description Relative complement of two sorted index sets — the values in `a`
 * that are NOT in `b` — returned in ascending order. `b` acts as a drop-set
 * filtered out of `a`.
 *
 * The result is written into `out` when supplied and large enough
 * (`out.length >= |a|`, the maximum possible size — reached when the sets are
 * disjoint); otherwise a fresh typed array is allocated. The element type is
 * `Uint32Array` when either input is a `Uint32Array`, and `Uint16Array`
 * otherwise. The return value is a `.subarray` view trimmed to the result size,
 * so `out` — when provided — must be a typed array.
 *
 * @param {number[]|Uint16Array|Uint32Array} a - Sorted, strictly-increasing set
 *   (the values to keep from).
 * @param {number[]|Uint16Array|Uint32Array} b - Sorted, strictly-increasing set
 *   (the drop-set to subtract).
 * @param {Uint16Array|Uint32Array} [out] - Optional scratch buffer, reused when
 *   `out.length >= a.length` to avoid per-call allocation.
 * @returns {Uint16Array|Uint32Array} Ascending view of the elements of `a`
 *   absent from `b`; equals `a` when the sets are disjoint, empty when
 *   `a ⊆ b`.
 *
 * @example
 * Array.from(andNot([1, 2, 3, 5], [2, 3, 4]))   // → [1, 5]   (2,3 dropped; 4 absent from a)
 * Array.from(andNot([1, 2, 3], [1, 2, 3, 4]))   // → []       (a ⊆ b)
 * Array.from(andNot([1, 3, 5], [2, 4, 6]))      // → [1, 3, 5] (disjoint → all of a)
 *
 * @example
 * // Apply a stored drop-set (¬M_τ) to a context in a hot loop, no allocation.
 * const out = new Uint32Array(x.length);
 * const kept = andNot(x, dropSet, out);   // view onto `out`
 */
export const andNot = (a, b, out) => {
  const alen = a.length, blen = b.length;
  let i = 0, j = 0, k = 0, ai, bj, n = alen;

  out && out.length >= n || (
    out = ((a instanceof Uint32Array) || (b instanceof Uint32Array)) && new Uint32Array(n)
    || new Uint16Array(n)
  );

  // NB: as in xor, the emitting branch ends its comma-sequence on `++i` (always
  // truthy), NOT on `out[k++]=ai` — ai can be 0 (part id 0 is the first dict
  // entry) and a falsy result would fall through the `||` chain and
  // double-process. `a[i] < b[j]` means a[i] is below every remaining b, hence
  // absent from b → emit; equality drops both; `a[i] > b[j]` advances b only.
  while (i < alen && j < blen) {
    (ai = a[i]) < (bj = b[j]) && (
      out[k++] = ai, ++i
    ) || (
      ai > bj && ++j
    ) || (++i, ++j);
  }

  while (i < alen) out[k++] = a[i++];

  return out.subarray(0, k);
}

/**
 * @ignore
 * Default export.
 */
export default andNot;
