"use strict";

/**
 * @file adaptedGate.js
 * @brief Signed-threshold discriminator gate under the ADAPTED mask, evaluated
 * without ever materializing the mask.
 *
 * The gate scores a context `x` (a sorted sparse bit set) for one target word:
 *
 *   score(x) = ( |p⁺ ∧ M_τ ∧ x| − |p⁻ ∧ M_τ ∧ x| ) / |M_τ ∧ x|      (definition)
 *
 * with the ADAPTED mask `M_τ = ¬(M_glob ∧ p⁺)` — drop a bit iff it is common
 * AND positive. Because the whole drop-set lives inside `p⁺`, it equals the raw
 * complement, `D_τ = p⁺ ∧ ¬M_τ = ¬M_τ` (the ~429 dropped bits), so `M_τ = ¬D_τ`
 * and the score reduces to counts over three SMALL stored sets:
 *
 *   score(x) = ( a − m ) / d                                        (simplified)
 *     a = |p⁺_adapted ∧ x| = andCount(x, p⁺_adapted)   p⁺_adapted = p⁺ ∧ M_τ
 *     m = |p⁻ ∧ x|         = andCount(x, p⁻)
 *     d = |M_τ ∧ x| = |x| − |D_τ ∧ x| = andNotCount(x, D_τ)         (the support)
 *
 * The denominator is the masked support read through the complement: one
 * andNotCount against the ~429-bit drop-set, never against p⁺. Stored sets
 * partition-and-tail p⁺ (`p⁺ = p⁺_adapted ⊔ D_τ`), so p⁺ itself is never stored.
 * No dense mask, no scan of the ~50k kept bits: the cost is
 * O(|x| + |p⁺_adapted| + |p⁻| + |D_τ|), all against sparse index sets.
 *
 * NB: this identity is specific to the adapted mask, whose drop-set ⊆ p⁺. It
 * does NOT hold for a band mask that drops bits in ¬p⁺ (e.g. the
 * discriminativeness middle) — there the denominator is |p⁺∧x| + |p⁻∧x|.
 *
 * @see {@link module:core/math/sparse/andCount}     a, m
 * @see {@link module:core/math/sparse/andNotCount}  d = |M_τ ∧ x|
 */

import andCount from "../math/sparse/andCount.js";
import andNotCount from "../math/sparse/andNotCount.js";

/**
 * @typedef {Object} AdaptedGate
 * @property {number[]|Uint16Array|Uint32Array} pPlusAdapted - `p⁺ ∧ M_τ`, the
 *   positive bits that survive the mask (sorted, strictly increasing).
 * @property {number[]|Uint16Array|Uint32Array} pMinus - `p⁻`, the negative tail
 *   (sorted, strictly increasing). Always ⊆ M_τ, so stored raw.
 * @property {number[]|Uint16Array|Uint32Array} dropSet - `D_τ = ¬M_τ`, the
 *   dropped bits (common ∧ positive; sorted, strictly increasing). Drives the
 *   support via `|M_τ ∧ x| = |x| − |D_τ ∧ x|`. Disjoint from p⁺_adapted; their
 *   union is p⁺.
 */

/**
 * @function adaptedGateScore
 * @description Signed-threshold score of a context under the adapted mask,
 * computed via the simplified `(a − m)/(a + u)` form.
 *
 * Returns `0` for the empty-support case `a + u === 0` — i.e. when `x` retains
 * no kept bit — matching the convention that a context with no evidence scores
 * neutral rather than dividing by zero.
 *
 * @param {number[]|Uint16Array|Uint32Array} x - Context bit set (sorted,
 *   strictly increasing).
 * @param {AdaptedGate} gate - The three stored sparse sets.
 * @returns {number} `(a − m) / d` in `[-1, 1]`, or `0` when the support
 *   `d = |M_τ ∧ x|` is 0.
 *
 * @example
 * const gate = { pPlusAdapted: [2, 7], pMinus: [9], dropSet: [5] };
 * adaptedGateScore([2, 3, 9], gate);   // a=1 (2), m=1 (9), d=|{2,3,9}∖{5}|=3 → (1−1)/3 = 0
 */
export const adaptedGateScore = (x, gate) => {
  const a = andCount(x, gate.pPlusAdapted);
  const m = andCount(x, gate.pMinus);
  const d = andNotCount(x, gate.dropSet);   // |M_τ ∧ x| = |x| − |D_τ ∧ x|

  return d > 0 ? (a - m) / d : 0;
}

/**
 * @function adaptedGateFires
 * @description Whether the gate fires on `x` — `score(x) > t`. Avoids the
 * division by comparing `a − m > t·d`, which is exact and never divides.
 *
 * @param {number[]|Uint16Array|Uint32Array} x - Context bit set (sorted,
 *   strictly increasing).
 * @param {AdaptedGate} gate - The three stored sparse sets.
 * @param {number} [t=0] - Threshold; the default `0` fires on net-positive
 *   evidence.
 * @returns {boolean} `true` iff `score(x) > t` (and `x` retains a kept bit).
 */
export const adaptedGateFires = (x, gate, t = 0) => {
  const a = andCount(x, gate.pPlusAdapted);
  const m = andCount(x, gate.pMinus);
  const d = andNotCount(x, gate.dropSet);   // |M_τ ∧ x|

  return d > 0 && (a - m) > t * d;
}

/**
 * @ignore
 * Default export.
 */
export default adaptedGateScore;
