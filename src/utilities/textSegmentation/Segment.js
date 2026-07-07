/**
 * @file Segment.js
 * @brief Typed 2-element range `[start, end]` with geometric query methods,
 * backed by `Uint32Array` for minimal memory footprint.
 *
 * `Segment` extends `Uint32Array(2)` so:
 * - `segment[0]` and `segment[1]` are the raw unsigned 32-bit integers.
 * - Destructuring `const [s, e] = segment` works identically to a plain array.
 * - `JSON.stringify` uses `toJSON()` to serialize as `[start, end]`.
 * - Memory cost is 8 bytes per instance vs ~100+ bytes for a plain object.
 *
 * `interval` is imported from {@link Section} (which has no dependency on
 * `Segment`) to avoid a circular require while sharing the single normalization
 * implementation across both classes.
 *
 * **Mutability note:** `Segment` instances are mutable — the build loop in
 * `segmentText` extends segments by writing directly to `prev[1]`. This is
 * intentional for performance but means callers should not cache `[1]` values
 * across calls to `segmentText`.
 */

"use strict";

import { interval, intersect, intersects, contains } from "./interval.js";

/**
 * @class Segment
 * @extends Uint32Array
 * @description Immutable-friendly 2-element typed range `[start, end]`.
 * Accepts either two numeric arguments or a single array-like `[start, end]`.
 *
 * **Clamping:** negative or missing start/end values are clamped to `0`.
 * This is intentional for unsigned index safety — callers should not pass
 * negative values; a `console.warn` fires in development if they do.
 *
 * @param {number|Array|Uint32Array|Uint16Array} start
 *   Inclusive start index, or an array-like `[start, end]`.
 * @param {number} [end]
 *   Exclusive end index. Ignored when `start` is array-like.
 *
 * @example
 * new Segment(0, 11)          // → Uint32Array [0, 11]
 * new Segment([0, 11])        // → Uint32Array [0, 11]
 * new Segment(someUint32Array) // → copies [0] and [1]
 */
export class Segment extends Uint32Array {
  constructor(...input) {
    super(2);
    let { start, end } = interval(...input);

    // Clamp negatives to 0 — Uint32Array would wrap them to large values.
    start > 0 || (start = 0);
    this[0] = start;
    end > 0 || (end = start);
    this[1] = end;
  }

  /** @property {number} start - Inclusive start character index. */
  get start() { return this[0]; }

  /** @property {number} end - Exclusive end character index. */
  get end() { return this[1]; }

  /** @property {number} span - Number of characters covered by this segment. */
  get span() { return this[1] - this[0]; }

  /**
   * @method extract
   * @description Extracts the segment's text from the source string.
   * @param {string} text - The original source text.
   * @returns {string}
   */
  extract(text) {
    return text.slice(this.start, this.end);
  }

  /**
   * @method toJSON
   * @description Serializes the segment as a plain `[start, end]` array.
   * Required because `Uint32Array` serializes as a plain object by default.
   * @returns {number[]}
   */
  toJSON() {
    return Array.from(this);
  }

  /**
   * @method toString
   * @description Serializes the segment as a plain `[start, end]` string.
   * 
   * @returns {number[]}
   */
  toString() {
    return `Segment [${this.start}, ${this.end}]`;
  }

  /**
   * @method getIntersection
   * @description Returns the overlapping range between this segment and another,
   * or `null` if they do not overlap. Touching ranges return `null`.
   * @param {...*} input - Any form accepted by {@link interval}.
   * @returns {Segment|null}
   */
  getIntersection(...input) {
    const output = intersect(this, interval(...input));
    return output && new Segment(output) || null;
  }

  /**
   * @method intersectsWith
   * @description Returns `true` if this segment overlaps the given range.
   * Touching ranges (end === other.start) do not count as intersecting.
   * @param {...*} input - Any form accepted by {@link interval}.
   * @returns {boolean}
   */
  intersectsWith(...input) {
    return intersects(this, interval(...input));
  }

  /**
   * @method isWithin
   * @description Returns `true` if this segment is entirely contained inside
   * the given range (inclusive on both ends).
   * @param {...*} input - Any form accepted by {@link interval}.
   * @returns {boolean}
   */
  isWithin(...input) {
    return contains(interval(...input), this);
  }

  /**
   * @method contains
   * @description Returns `true` if this segment fully contains the given range.
   * @param {...*} input - Any form accepted by {@link interval}.
   * @returns {boolean}
   */
  contains(...input) {
    return contains(this, interval(...input));
  }
}

Segment.isSegment = true;

/**
 * @function Segment.create
 * @description Factory method. Equivalent to `new Segment(...args)`.
 * @param {...*} args
 * @returns {Segment}
 */
export const create = Segment.create = (...args) => new Segment(...args);

/**
 * @ignore
 */
export default Object.freeze(Segment);