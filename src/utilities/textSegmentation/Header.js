"use strict";

/**
 * @file Header.js
 * @brief Heading-line segment extending {@link Segment} with level, title,
 * and an `.isHeader` identity flag.
 *
 * `Header` extends `Segment` — it is still a `Uint32Array(2)` with `.start`,
 * `.end`, `.extract()`, `.toJSON()`, and all geometric methods. The additions
 * are purely semantic:
 *
 * - `.level`    — heading depth (1 for `#`, 2 for `##`, … 6 for `######`).
 * - `.title`    — heading text stripped of marker characters and whitespace.
 * - `.isHeader` — always `true`; used by {@link Section} to identify whether
 *                 `this[0]` is a header or a plain body segment.
 *
 * Constructed by format-specific parsers (e.g. {@link segmentMarkdownTextSection})
 * and placed as `this[0]` in the resulting {@link Section} nodes.
 *
 * `toJSON()` returns a clean `[start, end]` pair — `_level` and `_title` are
 * plain own properties and would appear in `JSON.stringify` unless excluded.
 * Override `toJSON` here to ensure consistent serialization with `Segment`.
 *
 * @see {@link Segment}
 * @see {@link Section}
 * @see {@link segmentMarkdownTextSection}
 */

import Segment from "./Segment.js";

/**
 * @class Header
 * @extends Segment
 * @description Heading-line segment. Carries `.level`, `.title`, and `.isHeader`
 * in addition to all {@link Segment} geometry. Placed as `this[0]` in a
 * {@link Section} by format-specific parsers.
 *
 * @param {number} start         - Inclusive start index of the heading line.
 * @param {number} end           - Exclusive end index of the heading line.
 * @param {number} [level]       - Heading depth (1–6).
 * @param {number} [titleOffset] - Where the real title begin.
 *
 * @example
 * const h = new Header(0, 9, 1);
 * h.level;           // → 1
 * h.isHeader;        // → true
 * h.extract(text);   // → "# My Title"
 * h.toJSON();        // → [0, 9]
 */
export class Header extends Segment {
  #titleOffset = 0;
  constructor(start, end, level, titleOffset) {
    super(start, end);
    
    isNaN(level) || Object.defineProperty(this, "level", {
      value: level,
      enumerable: true
    });

    this.#titleOffset = Math.min(Math.max(titleOffset || 0), this.span);
  }

  /**
   * @method toJSON
   * @description Serializes as a clean `[start, end]` pair, identical to
   * {@link Segment#toJSON}. `_level` and `_title` are excluded so embedding
   * a `Header` in a JSON document is safe.
   * @returns {number[]}
   */
  toJSON() {
    return [this[0], this[1]];
  }

  /**
   * @method toString
   * @description Serializes the segment as a plain `[start, end]` string.
   * 
   * @returns {number[]}
   */
  toString() {
    return `Header [${this.start}, ${this.end}]`;
  }

  /**
   * @method extractTitle
   * @description Extracts the heading's clean title text from the source
   * string. Skips the leading marker (`#titleOffset` characters past the
   * segment start, e.g. past `"## "` for an ATX heading) and strips any
   * surrounding `#` characters and whitespace via {@link TRIM_HEADING_RE}.
   * @param {string} text - The original source text this header indexes into.
   * @returns {string} The heading title with markers and padding removed.
   * @example
   *   // For a header parsed from the line "## Methods":
   *   header.extractTitle(text);   // → "Methods"
   */
  extractTitle(text) {
    return text.slice(this.start + this.#titleOffset, this.end).replace(TRIM_HEADING_RE, "");
  }
}

Header.isHeader = true;

/** @type {RegExp} Matches ATX headings: optional leading space, 1-6 #, space, title. */
const TRIM_HEADING_RE = /^\s*\#*\s*|\s*\#*\s*$/g;

/**
 * @ignore
 */
export default Object.freeze(Header);