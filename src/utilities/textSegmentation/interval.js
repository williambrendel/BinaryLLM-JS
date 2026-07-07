"use strict";

/**
 * @file interval.js
 * @brief Lightweight range normalization and geometric query utilities.
 *
 * Provides a single entry point — `interval()` — that normalizes any
 * supported range representation into a plain `{ start, end }` object,
 * plus three standalone geometric predicates that operate on normalized
 * intervals: `intersect`, `intersects`, and `contains`.
 *
 * All functions accept any form that `interval()` accepts, so callers never
 * need to pre-normalize their inputs.
 *
 * **Accepted input forms:**
 * - `(start, end)` — two numeric positional arguments.
 * - `{ start, end }` — any object carrying `.start` and `.end` (duck typing).
 *   Covers {@link Segment}, {@link Section}, {@link Header}, or plain objects.
 * - `[start, end]` — a flat two-element array or typed array.
 * - `[[s,e], [s,e], ...]` — array of pairs; normalizes to the span from the
 *   first pair's start to the last pair's end.
 *
 * **Exported surface:**
 * - `interval(input, ...other)` → `{ start, end }`
 * - `interval.intersect(a, b)`  → `{ start, end } | null`
 * - `interval.intersects(a, b)` → `boolean`
 * - `interval.contains(big, small)` → `boolean`
 *
 * @example
 * interval(5, 10)                        // → { start: 5, end: 10 }
 * interval([5, 10])                      // → { start: 5, end: 10 }
 * interval({ start: 5, end: 10 })        // → { start: 5, end: 10 }
 * interval(new Segment(5, 10))           // → { start: 5, end: 10 }  (duck typing)
 * interval([[0,5], [7,12]])              // → { start: 0, end: 12 }
 *
 * @see {@link Segment}
 * @see {@link Section}
 */

/**
 * @function interval
 * @description Normalizes variadic range input into `{ start, end }`.
 *
 * Resolution order:
 * 1. Two numbers `(start, end)` — direct positional arguments.
 * 2. Object with `.start` and `.end` — passed through unchanged (duck typing).
 * 3. Array or array-like — recurses as `interval(input[0], input[input.length-1])`,
 *    which handles flat `[s, e]`, typed arrays, and multi-pair arrays.
 *
 * The variadic `...other` signature means extra arguments are ignored, making
 * the function safe to call from spread contexts.
 *
 * @param {number|Object|Array|Uint32Array} input
 *   Start value, a range object, or an array-like range.
 * @param {...number} other
 *   When `input` is a number, the last element of `other` is used as `end`.
 *   All other elements are ignored.
 * @returns {{ start: number, end: number }}
 *
 * @example
 * interval(3, 9)                   // → { start: 3, end: 9 }
 * interval([3, 9])                 // → { start: 3, end: 9 }
 * interval({ start: 3, end: 9 })   // → { start: 3, end: 9 }
 * interval([[0,5],[7,12]])         // → { start: 0, end: 12 }
 */
export const interval = (input, ...other) => (
  // 1. Handle (start, end) positional arguments
  typeof input === "number" && { start: input, end: other[other.length - 1] || input }
  // 2. Invalid input.
  || ((input === null || input === undefined) && {})
  // 3. Handle Segment or Section instances (duck typing on .start / .end)
  || (typeof input === "object" && "start" in input && "end" in input && input)
  // 4. Handle [start, end] or [[s,e], [s,e], ...]
  || (
    interval(input?.[0], (other = other[other.length - 1])?.[other.length - 1] || input?.[input.length - 1])
  )
);

/**
 * @function intersect
 * @description Returns the overlapping range of two intervals, or `null` if
 * they do not overlap. Touching ranges (where one ends exactly where the other
 * starts) do not count as intersecting and return `null`.
 *
 * The result is written into `output` (default `{}`) to allow callers to
 * provide a pre-allocated object and avoid allocation in hot paths.
 *
 * @param {*} interval1 - Any form accepted by {@link interval}.
 * @param {*} interval2 - Any form accepted by {@link interval}.
 * @param {Object} [output={}] - Object to write `start`/`end` into.
 * @returns {{ start: number, end: number } | null}
 *   The intersection range, or `null` if the intervals do not overlap.
 *
 * @example
 * intersect([0, 10], [5, 15])    // → { start: 5, end: 10 }
 * intersect([0, 5],  [5, 10])    // → null  (touching, not overlapping)
 * intersect([0, 5],  [6, 10])    // → null  (disjoint)
 */
export const intersect = (interval1, interval2, output = {}) => (
  interval1 = interval(interval1),
  interval2 = interval(interval2),
  (output.start = Math.max(interval1.start, interval2.start)) < (output.end = Math.min(interval1.end, interval2.end)) && output || null
);

/**
 * @function intersects
 * @description Returns `true` if two intervals overlap. Touching ranges do
 * not count as intersecting.
 *
 * @param {*} interval1 - Any form accepted by {@link interval}.
 * @param {*} interval2 - Any form accepted by {@link interval}.
 * @returns {boolean}
 *
 * @example
 * intersects([0, 10], [5, 15])   // → true
 * intersects([0, 5],  [5, 10])   // → false  (touching)
 * intersects([0, 5],  [6, 10])   // → false  (disjoint)
 */
export const intersects = (interval1, interval2) => (
  interval1 = interval(interval1),
  interval2 = interval(interval2),
  Math.max(interval1.start, interval2.start) < Math.min(interval1.end, interval2.end)
);

/**
 * @function contains
 * @description Returns `true` if `big` fully contains `small` — i.e. `small`
 * falls entirely within `big` (inclusive on both ends).
 *
 * @param {*} big   - The outer range. Any form accepted by {@link interval}.
 * @param {*} small - The inner range. Any form accepted by {@link interval}.
 * @returns {boolean}
 *
 * @example
 * contains([0, 20], [5, 15])    // → true
 * contains([0, 20], [0, 20])    // → true   (exact match counts)
 * contains([5, 15], [0, 20])    // → false  (big is smaller)
 * contains([0, 20], [15, 25])   // → false  (small extends beyond)
 */
export const contains = (big, small) => (
  big = interval(big),
  small = interval(small),
  big.start <= small.start && big.end >= small.end
);

/**
 * @ignore
 */
export default interval;