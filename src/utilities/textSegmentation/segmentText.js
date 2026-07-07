"use strict";

/**
 * @file segmentText.js
 * @brief Sentence-level text segmentation returning {@link Segment} instances,
 *        plus paragraph/header detection and clause/word-level sub-segmentation.
 *
 * **Pipeline overview.**
 * The primary entry point, {@link segmentText}, splits text on sentence-ending
 * punctuation (`.` `!` `?` `;`) and control characters, then runs a merge pass
 * that collapses adjacent candidates separated only by a single `\n` or plain
 * whitespace into one segment. Paragraph breaks (`\n\n+`) and explicit
 * sentence delimiters always remain split. Horizontal-rule lines (`---`,
 * `===`, etc.) are detected as their own delimiter segments and stripped
 * from the result, but their adjacency information is preserved on the
 * neighbours via `hasDelimLineBefore`. When `checkForHeader` is enabled,
 * a post-pass ({@link updateHeaders}) promotes qualifying segments to
 * {@link Header} instances.
 *
 * Two further entry points operate on already-segmented input:
 * - {@link subsegment} decomposes a single {@link Segment} into a flat list
 *   of clause-, sub-clause-, word-, pair-, and triplet-level candidates,
 *   intended as input to span-selection / concept-extraction algorithms.
 * - {@link subsegmentText} is the batch wrapper: it accepts strings,
 *   {@link Segment}s, {@link Section}s, or any mix thereof and returns the
 *   concatenated sub-segments.
 *
 * **Exported surface.**
 * - Default export: `segmentText(text, checkForHeader)` → `Segment[]`
 * - `segmentText.Segment` — the {@link Segment} class, re-exported for
 *   callers that want to type-check results without a separate import.
 * - `segmentText.updateHeaders` — see {@link updateHeaders}.
 * - `segmentText.subsegment` — see {@link subsegment}.
 * - `segmentText.subsegmentText` — see {@link subsegmentText}.
 *
 * **Segment mutability.** The merge step in {@link addSegment} writes
 * directly to `prev[1]` on an existing `Segment` to extend its end. This is
 * intentional — creating a new `Segment` per merge would generate
 * significant GC pressure on long texts. Callers must not cache
 * `segment[1]` values across calls to `segmentText`.
 *
 * **Dot protection.** Before any scanning, the input is run through
 * {@link protectDots} so that abbreviations, decimals, ellipses, etc. do
 * not produce spurious sentence breaks. Index ranges are remapped back to
 * the original text before being returned, so `Segment` ranges always
 * refer to offsets in the user-supplied string.
 *
 * @see {@link Segment}
 * @see {@link Section}
 * @see {@link Header}
 * @see {@link protectDots}
 * @see {@link detectOrderedHeader}
 */

import Segment from "./Segment.js";
import Section from "./Section.js";
import Header from "./Header.js";
import { protectDots, restore } from "./protectDots.js";
import detectOrderedHeader from "./detectOrderedHeader.js";

/**
 * @function segmentText
 * @description Splits text into sentence-level `[start, end)` index pairs.
 * Delimiters are `.`, `!`, `?`, `;`, and whitespace control characters
 * (tab, newline, carriage return). Leading and trailing punctuation and
 * whitespace are trimmed from the input before processing.
 *
 * Adjacent segments separated by a single newline and no blank line are
 * merged into one segment. A new segment is started when the gap between
 * two candidates contains either:
 * - Two or more `\n` characters (paragraph break — sets `hasBlankLineBefore`),
 *   or
 * - Any non-whitespace character (`c > 32`) — i.e. a sentence-ending
 *   delimiter that was itself recorded as a split point (sets
 *   `hasDelimBefore`).
 *
 * Horizontal-rule lines (3+ consecutive `*`/`+`/`-`/`=`/`_`/`~`) are
 * detected by {@link isDelimiterSegment}, marked `isDelim = true`, and
 * filtered out of the result. Segments immediately following a delimiter
 * line carry `hasDelimLineBefore = true`.
 *
 * This design means single-newline line breaks within a paragraph do not
 * produce separate segments, while blank lines and horizontal rules
 * always do.
 *
 * Index ranges are remapped from the dot-protected text back to the
 * original text before return, so callers can index into the original
 * string directly.
 *
 * Delimiter character codes used internally:
 * - 9:  `\t`  (tab)
 * - 10: `\n`  (newline)
 * - 13: `\r`  (carriage return)
 * - 32: space
 * - 33: `!`
 * - 44: `,`
 * - 46: `.`
 * - 58: `:`
 * - 59: `;`
 * - 63: `?`
 *
 * @param {string|*} text - Input text. Non-string values are coerced via
 *   template literal. Falsy values return `[]` immediately.
 * @param {boolean} [checkForHeader=false] - When `true`, runs
 *   {@link updateHeaders} as a post-pass, replacing qualifying segments
 *   with {@link Header} instances in place.
 *
 * @returns {Segment[]} Array of `Segment` instances (each behaves as a
 *   `[start, end)` tuple) where `start` is inclusive and `end` is
 *   exclusive, suitable for `text.slice(start, end)`. When `checkForHeader`
 *   is `true`, some entries may be {@link Header} instances. Returns `[]`
 *   if the input is empty or contains only delimiters and whitespace.
 *
 * @example
 * // Basic sentence splitting
 * segmentText("Hello world. How are you?");
 * // → [ [0, 11], [13, 24] ]
 * // text.slice(0, 11) → "Hello world"
 * // text.slice(13, 24) → "How are you"
 *
 * @example
 * // Single newline — merged into one segment
 * segmentText("Hello world.\nHow are you?");
 * // → [ [0, 24] ]
 *
 * @example
 * // Double newline — paragraph break, two segments
 * segmentText("Hello world.\n\nHow are you?");
 * // → [ [0, 11], [14, 25] ]
 *
 * @example
 * // Leading and trailing punctuation trimmed
 * segmentText("...Hello world. How are you?!");
 * // → [ [3, 25] ]
 *
 * @example
 * // Header detection
 * segmentText("# Intro\n\nBody text.", true);
 * // → [ Header(0, 7, level=1), Segment(9, 19) ]
 *
 * @example
 * // Reconstruct text from segments
 * const segments = segmentText(text);
 * const sentences = segments.map(([s, e]) => text.slice(s, e));
 */
export const segmentText = (text, checkForHeader = false) => {
  if (!text) return [];
  typeof text === "string" || (text = `${text}`);

  // Convert text to protect dots.
  const { protectedText, dictionary } = protectDots(text);
  const origText = text;
  text = protectedText;

  const len = text.length, delimIndices = new Uint32Array(len);
  let n = 0, m = 0, p = 0, s = 0, e = len - 1, c;

  // Trim leading delimiters and whitespace.
  // 9: \t, 10: \n, 13: \r, 32: space, 33: !, 44: ,, 46: ., 58: :, 59: ;, 63: ?
  while (s !== len && (c = text.charCodeAt(s)) < 64 && (
    c < 14 && c > 8 || c === 32
    || c === 33 || c === 44 || c === 46
    || c === 58 || c === 59 || c === 63
  )) ++s;

  // Trim trailing delimiters and whitespace.
  while (e > s && (c = text.charCodeAt(e)) < 64 && (
    c < 14 && c > 8 || c === 32
    || c === 33 || c === 44 || c === 46
    || c === 58 || c === 59 || c === 63
  )) --e;

  if (s === len) return [];
  ++e < s && (e = s);

  // Collect delimiter positions.
  for (let i = s; i !== e; ++i) {
    c = text.charCodeAt(i);
    c < 64 && (
      (c < 14 && c > 9) || c === 33 ||
      c === 46 || c === 59 || c === 63
    ) && (delimIndices[n++] = i);
  }

  const segments = new Array(n + 2);
  p = s;

  for (let i = 0, j, k; i !== n; ++i) {
    // Trim whitespace from segment end (scan back from delimiter).
    j = (k = delimIndices[i]) - 1;
    while (j >= p && ((c = text.charCodeAt(j)) < 14 && c > 8 || c === 32)) --j;
    m = addSegment(segments, m, p, ++j, text);

    // Trim whitespace from segment start (scan forward past delimiter).
    p = k + 1;
    while (p !== e && ((c = text.charCodeAt(p)) < 14 && c > 8 || c === 32)) ++p;
  }

  // Capture final segment after last delimiter.
  m = addSegment(segments, m, p, e, text);

  // Update segment length.
  segments.length = m;

  // Remove delim segments.
  m = 0;
  for (let i = 0, l = segments.length, segment; i !== l; ++i) {
    (segment = segments[i]).isDelim || (segments[m++] = segment);
  }

  // Update segment length.
  segments.length = m;

  // Convert text back and remap start/end.
  for (let i = 0, l = segments.length, segment, offset = 0; i !== l; ++i) {
    segment = segments[i];

    // Restore.
    const protectedSlice = text.slice(segment.start, segment.end);
    const restored = restore(protectedSlice, dictionary);

    // Remap the start/end.
    segment[0] += offset;
    segment[1] += (offset += restored.length - protectedSlice.length);
  }

  // Check for header.
  checkForHeader && updateHeaders(segments, origText);

  // return segments.
  return segments;
};

/**
 * @function addSegment
 * @private
 * @description
 * Appends a new segment to the segments array, or merges with the previous
 * segment when the gap between them is just a single newline (no blank line,
 * no intervening sentence delimiter).
 *
 * Behavior summary (in evaluation order):
 * - **Empty range** (`start >= end`): no-op, returns `m` unchanged.
 * - **Horizontal-rule run** (3+ consecutive `*`/`+`/`-`/`=`/`_`/`~`,
 *   detected by {@link isDelimiterSegment}): pushed as a new segment with
 *   `isDelim = true`. These are filtered out of the final result by
 *   {@link segmentText} but signal `hasDelimLineBefore` on the next segment.
 * - **First segment** in the array: pushed as-is, no flags set.
 * - **Previous segment is a delimiter**: pushed with
 *   `hasDelimLineBefore = true`.
 * - **Gap analysis** (otherwise): scans the characters between
 *   `prev[1]` and `start`, counting newlines (`nl`) and any
 *   non-whitespace characters (`dl`, where `c > 32`).
 *   - If `nl > 1` (blank line) or `dl > 0` (sentence delimiter character
 *     present in the gap), pushes a new segment. Sets `hasBlankLineBefore`
 *     when `nl > 1` and `hasDelimBefore` when `dl > 0` (both can be set).
 *   - Otherwise (gap is a single newline / pure whitespace): merges into
 *     the previous segment by extending `prev[1] = end`, and sets
 *     `prev.hasNewline = true`.
 *
 * Index access on `Segment` instances (`prev[1]`) is valid because
 * `Segment` extends `Array`, with index `0` holding the start and index
 * `1` holding the end. Mutating `prev[1]` directly avoids allocating a
 * new `Segment` per merge — a meaningful win on long texts where most
 * candidates merge.
 *
 * @param {Segment[]} segments - Pre-allocated segments array being filled in.
 * @param {number} m - Current write index into `segments` (i.e. the count
 *   of segments already pushed).
 * @param {number} start - Inclusive start index of the candidate segment in `text`.
 * @param {number} end - Exclusive end index of the candidate segment in `text`.
 * @param {string} text - The (already-protected) text being segmented.
 *
 * @returns {number} The updated write index. Equals `m` if nothing was
 *   pushed (empty range or merge), or `m + 1` if a new segment was appended.
 */
const addSegment = (segments, m, start, end, text) => {
  if (start < end) {
    // Check if current segment is a delimiter.
    if (isDelimiterSegment(start, end, text)) {
      const segment = segments[m] = new Segment(start, end);
      segment.isDelim = true;
      return ++m;
    }
    
    // If first segment.
    if (!m) {
      segments[m] = new Segment(start, end);
      return ++m
    }
   
    const prev = segments[m - 1];

    // If previous segment is a delimiter segment.
    if (prev.isDelim) {
      (segments[m] = new Segment(start, end)).hasDelimLineBefore = true;
      return ++m
    }

    // Analyse gap between segments.
    let dl = 0, nl = 0;
    for (let i = Math.min(prev[1], start), c; i !== start; ++i) {
      nl += (c = text.charCodeAt(i)) === 10; // count newlines
      dl += c > 32;                          // count sentence delimiter in gap
    }

    // If blank line detected, or has been partinioned via a delimiter, it's a new segment.
    if (nl > 1 || dl) {
      const segment = segments[m] = new Segment(start, end);
      nl > 1 && (segment.hasBlankLineBefore = true);
      dl && (segment.hasDelimBefore = true);
      return ++m
    }

    // Just a newline --> merge it.
    prev[1] = end;
    prev.hasNewline = true;
  }
  return m;
}

/**
 * @function isDelimiterSegment
 * @private
 * @description
 * Tests whether a `[start, end)` range in `text` consists entirely of a
 * single repeated "rule" character — used to detect horizontal rules and
 * separator lines like `---`, `===`, `***`, `___`, `+++`, `~~~`.
 *
 * Requires at least 3 consecutive identical characters from the allowed set.
 * Allowed character codes:
 * - 42:  `*`
 * - 43:  `+`
 * - 45:  `-`
 * - 61:  `=`
 * - 95:  `_`
 * - 126: `~`
 *
 * @param {number} start - Inclusive start index of the range in `text`.
 * @param {number} end - Exclusive end index of the range in `text`.
 * @param {string} text - The text being scanned.
 *
 * @returns {boolean} `true` if the range is a homogeneous run of 3+ allowed
 *   delimiter characters; `false` otherwise.
 *
 * @example
 *   isDelimiterSegment(0, 5, "-----")  // → true
 *   isDelimiterSegment(0, 5, "===  ")  // → false (mixed)
 *   isDelimiterSegment(0, 2, "--")     // → false (too short)
 *   isDelimiterSegment(0, 5, "abcde")  // → false (not allowed chars)
 */
const isDelimiterSegment = (start, end, text) => {
  if (end - start < 3) return false; // Need at least 3 concecutive symbols.
  const ref = text.charCodeAt(start);
  if (ref !== 42 && ref !== 43 && ref !== 45 && ref !== 61 && ref !== 95 && ref !== 126) return false;
  let c;
  for (let i = start + 1; i !== end && (c = text.charCodeAt(i)) === ref; ++i);
  return c === ref;
}

/**
 * @function updateHeader
 * @private
 * @description
 * Inspects a single segment and, if it qualifies as a header, replaces it
 * in place in the segments array with a {@link Header} instance carrying
 * level and title-offset metadata.
 *
 * A segment qualifies for header detection when **all** of the following hold:
 * - It has a "break" before it: blank line, delimiter line, or it is the
 *   first segment in the document.
 * - It is **not** a merged multi-line segment (`!hasNewline`). Headers
 *   are by definition single-line constructs; a segment that absorbed a
 *   subsequent line via the merge pass cannot be one.
 * - It has a "break" after it: blank line, delimiter line, or it is the
 *   last segment in the document.
 *
 * When eligible, two header forms are recognized:
 * 1. **Markdown ATX headers** — leading `#` characters (char code 35),
 *    with the count of `#`s determining the level. After the `#` run and
 *    any whitespace, the trailing {@link detectOrderedHeader} call extracts
 *    an optional ordered-title offset (e.g. `## 1.2 Methods` → level 2,
 *    title offset advanced past `1.2 `).
 * 2. **Plain ordered headers** — detected by {@link detectOrderedHeader},
 *    which recognizes patterns such as `1. Title`, `A. Title`,
 *    `IV. Title`, etc., returning the implied level and the offset to the
 *    title text.
 *
 * If neither pattern matches, the segment is left unchanged.
 *
 * @param {Segment[]} segments - The segments array (will be mutated in place).
 * @param {number} m - Index of the segment to inspect.
 * @param {string} text - The text the segment indexes into.
 * @param {Object} [context={}] - Adjacency flags used to gate header detection.
 * @param {boolean} [context.hasBlankLineBefore] - The segment is preceded by a blank line.
 * @param {boolean} [context.hasDelimLineBefore] - The segment is preceded by a delimiter line (`---`, `===`, etc.).
 * @param {boolean} [context.hasNewline] - The segment was formed by merging across a single newline (disqualifies it from being a header).
 * @param {boolean} [context.isFirstSegment] - The segment is the first in the document.
 * @param {boolean} [context.isLastSegment] - The segment is the last in the document.
 * @param {boolean} [context.hasBlankLineAfter] - The segment is followed by a blank line.
 * @param {boolean} [context.hasDelimLineAfter] - The segment is followed by a delimiter line.
 *
 * @returns {void} Mutates `segments[m]` in place when a header is detected.
 *
 * @see {@link Header}
 * @see {@link detectOrderedHeader}
 * @see {@link updateHeaders} for the batch driver.
 */
export const updateHeader = (
  segments,
  m,
  text,
  {
    hasBlankLineBefore,
    hasDelimLineBefore,
    hasNewline,
    isFirstSegment,
    isLastSegment,
    hasBlankLineAfter,
    hasDelimLineAfter
  } = {}
) => {
  if (
    (hasBlankLineBefore || hasDelimLineBefore || isFirstSegment)
    && !hasNewline
    && (hasBlankLineAfter || hasDelimLineAfter || isLastSegment)
  ) {
    // Check markdown.
    const segment = segments[m];
    let i = segment.start, c;
    for (let e = segment.end; i !== e && (c = text.charCodeAt(i)) === 35; ++i);
    if (i > segment.start) {
      const level = i - segment.start;
      for (let e = segment.end; i !== e && (c = text.charCodeAt(i)) < 33; ++i);

      // Change segment to header.
      const {
        titleOffset = 0
      } = detectOrderedHeader(text.slice(i, segment.end)) || {};
      segments[m] = new Header(segment.start, segment.end, level, i - segment.start + titleOffset);
      return;
    }

    // Check title patterns.
    const res = detectOrderedHeader(text.slice(segment.start, segment.end));
    if (res) {
      const {
        level,
        titleOffset
      } = res;
      segments[m] = new Header(segment.start, segment.end, level, titleOffset);
    }
  }
}

/**
 * @function updateHeaders
 * @description
 * Walks the segments array and runs {@link updateHeader} on each entry,
 * deriving each segment's adjacency flags from its own and its neighbour's
 * properties. Qualifying segments are replaced in place with {@link Header}
 * instances.
 *
 * Adjacency derivation:
 * - `hasBlankLineBefore` / `hasDelimLineBefore`: read directly from the
 *   current segment (set during {@link addSegment}).
 * - `hasBlankLineAfter` / `hasDelimLineAfter`: read from the **next**
 *   segment's `hasBlankLineBefore` / `hasDelimLineBefore` — i.e. a blank
 *   line "after" segment `i` is the same blank line that is "before"
 *   segment `i + 1`.
 * - `isFirstSegment`: index `0`.
 * - `isLastSegment`: only `true` for the actual last segment, which is
 *   handled by a separate post-loop call (the loop iterates only up to
 *   `length - 1` so it can safely access `segments[i + 1]`).
 * - `hasNewline`: read directly from the current segment.
 *
 * @param {Segment[]} segments - The segments array (mutated in place).
 * @param {string} text - The original (un-protected) text the segments
 *   index into. Passed through to {@link updateHeader} for slicing.
 *
 * @returns {Segment[]} The same `segments` array, returned for chaining.
 *
 * @see {@link updateHeader}
 */
export const updateHeaders = (segments, text) => {
  // Check for header.
  for (let i = 0, l = segments.length - 1, cur, next; i < l; ++i) {
    cur = segments[i];
    next = segments[i + 1];
    updateHeader(
      segments,
      i,
      text,
      {
        hasBlankLineBefore: cur.hasBlankLineBefore,
        hasDelimLineBefore: cur.hasDelimLineBefore,
        isFirstSegment: !i,
        isLastSegment: false,
        hasBlankLineAfter: next.hasBlankLineBefore,
        hasDelimLineAfter: next.hasDelimLineBefore,
        hasNewline: cur.hasNewline
      }
    );
  }

  // Check for last header.
  const lastIndex = segments.length - 1, last =  segments[lastIndex];
  updateHeader(
    segments,
    lastIndex,
    text,
    {
      hasBlankLineBefore: last.hasBlankLineBefore,
      hasDelimLineBefore: last.hasDelimLineBefore,
      isFirstSegment: !lastIndex,
      isLastSegment: true,
      hasNewline: last.hasNewline
    }
  );

  return segments;
}

/**
 * @function subsegmentText
 * @description
 * Batch wrapper around {@link subsegment}. Accepts heterogeneous input —
 * a raw string, a single {@link Segment} or {@link Section}, or an array
 * mixing any of those — and returns the flat list of sub-segments produced
 * by running {@link subsegment} over each leaf segment.
 *
 * **Input normalization (in order):**
 * 1. If `text` is an object, it is treated as the options bag (allows the
 *    short call form `subsegmentText(input, options)`); the real `options`
 *    argument, if any, is merged on top.
 * 2. If `output` is missing, a fresh array is allocated.
 * 3. If `input` is a string, it is replaced by `segmentText(input, checkForHeader)`
 *    and `text` is set to the original string.
 * 4. If `input` is not already an array, it is wrapped in one.
 * 5. Each entry is then dispatched:
 *    - **string** → expanded via `segmentText` and concatenated into the
 *      working list.
 *    - **{@link Segment}** → used as-is.
 *    - **{@link Section}** → recursed into (the section is itself iterable
 *      over its own segments), with results appended directly to `output`.
 *    - **falsy** → silently skipped.
 *
 * **Per-segment processing:**
 * For each input segment, {@link subsegment} is called. Because subsegment
 * always emits the input range at `output[0]`, the original segment is
 * stripped via `slice(1)` when `includeOriginalSegment` is `false`.
 * If subsegment couldn't decompose the input (`length <= 1`), the
 * original segment is used as a fallback so callers don't get empty
 * output for a non-empty input.
 *
 * @param {string|Segment|Section|Array<string|Segment|Section>} input -
 *   The thing(s) to sub-segment.
 * @param {string|Object} [text] - The source text the segments index into,
 *   **or** the options bag (see normalization step 1). When `input` is a
 *   string, this is set to that string automatically.
 * @param {Object} [options]
 * @param {boolean} [options.checkForHeader] - Forwarded to {@link segmentText}
 *   when a string input needs to be segmented.
 * @param {boolean} [options.includeOriginalSegment] - When `true`, the
 *   pre-decomposition segment is included in the output ahead of its
 *   sub-segments. When `false` (default), it is stripped out.
 * @param {Segment[]} [output] - Accumulator array. Allocated if omitted.
 *   Pass an array to merge results from multiple calls.
 *
 * @returns {Segment[]} The `output` array, with all sub-segments appended.
 *
 * @see {@link subsegment}
 * @see {@link segmentText}
 */
export const subsegmentText = (input, text, options, output) => {
  // Normalize options and text.
  typeof text === "object" && (options = {...text, ...(options || {})});
  const {
    checkForHeader,
    includeOriginalSegment
  } = options || {};

  // Normalize output.
  output || (output = []);
  
  // Normalize input.
  typeof input === "string" && (text = input, input = segmentText(input, checkForHeader));
  Array.isArray(input) || (input = [input]);
  const _input = [];
  for (let i = 0, l = input.length, segment; i !== l; ++i) {
    (segment = input[i]) && (
      typeof segment === "string" && (
        text || (text = segment),
        _input.push(...segmentText(segment, checkForHeader))
      )
      || (segment instanceof Segment && _input.push(segment))
      || (segment instanceof Section && subsegmentText(segment, text, options, output))
    );
  }
  input = _input;

  // Sub-segment input.
  for (let i = 0, l = input.length, segment; i !== l; ++i) {
    segment = input[i];
    let res = subsegment(segment, text), tiers = res.tiers;
    res && res.length > 1 || (res = [segment]);
    includeOriginalSegment || (
      res = res.slice(1),
      res.tiers = tiers.map(t => t - 1)
    );
    output.push(...res);
  }

  return output;
}

/**
 * @function subsegment
 * @description
 * Decomposes a single {@link Segment} into a flat list of finer-grained
 * sub-segments suitable for use as candidate spans in concept-extraction
 * algorithms (e.g. representative-span selection over an embedding space).
 *
 * The decomposition runs three sweeping passes that *add to* (not replace)
 * the output. The output is intentionally not a partition: each pass
 * refines the previous one, and overlapping or duplicate ranges are
 * permitted by design.
 *
 * **Output invariant.** `output[0]` is always the input segment's full
 * range — either the segment instance itself (when no hard-clause split
 * fires) or a fresh `Segment(start, end)` covering the same range. Callers
 * that don't want the original in their result can `slice(1)` it off;
 * {@link subsegmentText} relies on this invariant.
 *
 * **Pass 1 — Hard-clause split (`:` and `;`).**
 *   Split the input segment at each colon and semicolon (with surrounding
 *   whitespace and runs of `:` / `;` consumed). Each resulting range is
 *   appended to the output as a "hard clause". If no `:` / `;` is present
 *   in the input, the original segment is appended unchanged so subsequent
 *   passes have something to walk.
 *
 * **Pass 2 — Soft-clause split (commas, parens, brackets, quotes, repeat of `:`/`;`).**
 *   For each hard clause, split at any of: `"` `(` `)` `,` `:` `;` `[` `]`.
 *   Each resulting range is appended. (`:` and `;` appear here as well so
 *   that any colons/semicolons left inside a hard clause are still
 *   subdivided.) If no soft-clause delimiter is found in a hard clause,
 *   nothing extra is appended for that clause — avoiding duplicate
 *   entries that would otherwise mirror Pass 1's output.
 *
 * **Pass 3 — Word, pair, triplet extraction.**
 *   For each entry currently in the output, tokenize on whitespace
 *   (`c < 33`) and commas (`c === 44`) into words. Each word is tagged
 *   `notAStopWord = true` unless it appears in the {@link stopWords} set.
 *   Then, walking the word list:
 *     - Each individual word with `span > 4` AND `notAStopWord` is
 *       appended. Indices `0..length-3` are emitted by the main
 *       pair-and-triplet loop; `words[length-2]` and `words[length-1]`
 *       are emitted by separate trailing handlers so the last two
 *       indices are not lost.
 *     - Each adjacent pair `(word[i], word[i+1])` where both are
 *       `notAStopWord` is appended (whitespace included in the span).
 *       The final pair (`words[length-2]` + `words[length-1]`) is
 *       emitted by the trailing handler.
 *     - Each adjacent triplet `(word[i], word[i+1], word[i+2])` where
 *       `word[i]` and `word[i+2]` are `notAStopWord` is appended (the
 *       middle word may be a stopword — this catches "X and Y", "X or Y"
 *       forms).
 *
 * **Why three passes?**
 * - Hard-clause boundaries (`:` `;`) typically separate distinct ideas.
 *   Splitting first preserves them from being merged in pass 2.
 * - Soft-clause boundaries refine ideas into sub-thoughts (parentheticals,
 *   quoted phrases, list items separated by commas).
 * - Word-level extraction adds compositional candidates (single concepts,
 *   noun phrases) that punctuation cannot identify.
 *
 * **Order:** clauses appear before their sub-clauses, which appear before
 * their words/pairs/triplets. Within each level, document order is
 * preserved.
 *
 * **Stopwords:** the role here is not IR-style filtering. The set is
 * intentionally small and reserved for clause connectors ("and", "or",
 * etc.) that would dominate pair/triplet extraction without contributing
 * concept content. The 4-char minimum on standalone words handles most
 * short function words without an explicit list.
 *
 * @param {Segment} segment - A `[start, end)` range to decompose.
 * @param {string}  text    - The source text the segment indexes into.
 *
 * @returns {Segment[]} Flat array of sub-segments with the input segment's
 *   range at index 0. Each entry is a `Segment` with valid `[start, end)`
 *   ranges. Word-level segments carry a `notAStopWord` flag for downstream
 *   use. May contain duplicate or overlapping ranges by design.
 *
 * @see {@link subsegmentText} for batch processing.
 * @see {@link stopWords} for the filter set.
 */
export const subsegment = (segment, text) => {
  const output = [segment], [start, end] = segment;
  let tier = 0;
  output.tiers = [];
  let i = start, s = start, c, j, seg, split = false, hasComma;
  for (; i !== end; ++i) {
    c = text.charCodeAt(i);
    if (c === 58 || c === 59) {
      for (j = i; j >= 0 && ((c = text.charCodeAt(j)) === 58 || c === 59 || c < 34); --j);
      ++j > s && (
        output.push(seg = new Segment(s, j)),
        seg.tier = tier,
        split = true
      );
      for (s = i; s !== end && ((c = text.charCodeAt(s)) === 58 || c === 59 || c < 34); ++s);
      i = s - 1;
    }
  }
  split && s < end && (
    output.push(seg = new Segment(s, end)),
    seg.tier = tier
  );

  // Continue splitting by parenthesis, obvious quote, and additional punctuation and add it to output.
  const n = output.length;
  output.tiers.push(n);
  ++tier;
  for (let k = n > 1 && 1 || 0; k !== n; ++k) {
    const [start, end] = output[k];
    i = s = start;
    split = false;
    for (; i !== end; ++i) {
      c = text.charCodeAt(i);
      if (c < 94 && c > 33 && (
          c === 34 || c === 40 || c === 41 || c === 44 || c === 58 || c === 59 || c === 91 || c === 93
      )) {
        for (j = i; j >= 0 && ((c = text.charCodeAt(j)) < 94 && (
          c < 35 || c === 40 || c === 41 || c === 44 || c === 58 || c === 59 || c === 91 || c === 93
        )); --j);
        ++j > s && (
          output.push(seg = new Segment(s, j)),
          seg.tier = tier,
          split = true
        );
        for (s = i; s !== end && ((c = text.charCodeAt(s)) < 94 && (
          c < 35 || c === 40 || c === 41 || c === 44 || c === 58 || c === 59 || c === 91 || c === 93
        )); ++s);
        i = s - 1;
      }
    }
    split && s < end && (
      output.push(seg = new Segment(s, end)),
      seg.tier = tier
    );
  }

  // spit by words.
  const m = output.length;
  output.tiers.push(m);
  ++tier;
  for (let k = m > n && n || (n > 1 && 1 || 0); k !== m; ++k) {
    const [start, end] = output[k], words = [];

    // Extract words.
    // For each new split: extract words of size > 4 and add it to output.
    i = s = start;
    for (; i !== end; ++i) {
      c = text.charCodeAt(i);
      if (c < 33 || c === 44 || c === 8212) {
        for (j = i; j >= 0 && ((c = text.charCodeAt(j)) < 33 || c === 44 || c === 8212); --j);
        ++j > s && (
          words.push(seg = new Segment(s, j)),
          hasComma && (seg.precededByDelimiter = true),
          seg.tier = tier,
          stopWords.has(text.slice(s, j).toLowerCase()) || (seg.notAStopWord = true)
        );
        hasComma = false;
        for (s = i; s !== end && ((c = text.charCodeAt(s)) < 33 || ((c === 44 || c === 8212) && (hasComma = true))); ++s);
        i = s - 1;
      }
    }
    s < end && (
      words.push(seg = new Segment(s, end)),
      hasComma && (seg.precededByDelimiter = true),
      seg.tier = tier,
      stopWords.has(text.slice(s, end).toLowerCase()) || (seg.notAStopWord = true)
    );

    // For each new split:
    // extract words of size > 4 and add it to output.
    // extract overlapping pairs and triplets and quadruplets and add it to output.
    i = 0;
    let w1, w2, w3;
    for (let e = words.length - 2; i < e; ++i) {
      (w1 = words[i]).notAStopWord && (
        w2 = words[i + 1],
        w1.span > 4 && (
          output.push(w1),
          w2.span > 4 && w2.notAStopWord && !w2.precededByDelimiter && (
            output.push(seg = new Segment(w1.start, w2.end)),
            seg.tier = tier
          )
        ),
        (w3 = words[i + 2]).notAStopWord && !w2.precededByDelimiter && !w3.precededByDelimiter && (
          output.push(seg = new Segment(w1.start, w3.end)),
          seg.tier = tier
        )
      );
    }
    words.length > 1 && (w1 = words[words.length - 2]).notAStopWord && (
      w1.span > 4 && (
        output.push(seg = new Segment(w1.start, w1.end)),
        seg.tier = tier,
        (w2 = words[words.length - 1]).span > 4 && w2.notAStopWord && !w2.precededByDelimiter && (
          output.push(seg = new Segment(w1.start, w2.end)),
          seg.tier = tier
        )
      )
    );
    words.length > 0 && (
      (w1 = words[words.length - 1]).notAStopWord && w1.span > 4 && (
        output.push(seg = new Segment(w1.start, w1.end)),
        seg.tier = tier
      )
    );
  }

  return output;
}

/**
 * @const stopWords
 * @private
 * @description
 * Small set of clause-connector words used by {@link subsegment} during
 * word/pair/triplet extraction. This is **not** an IR-style stopword list:
 * its only role is to prevent connectors like "and", "or", "but" from
 * dominating pair/triplet candidates without contributing concept content.
 *
 * The 4-character minimum on standalone-word emission in {@link subsegment}
 * handles most short function words ("a", "the", "of", "in", …) implicitly,
 * which is why this set can stay small.
 *
 * @type {Set<string>}
 */
const stopWords = new Set([
  // clause connectors
  "and", "or", "but", "nor", "yet", "so", "although",
  "if", "then", "else",
  "which", "that", "because", "however", "therefore", "thus",
  // pronouns / demonstratives
  "this", "these", "those", "they", "them", "their", "theirs",
  "it", "its", "itself",
  // discourse / temporal openers
  "when", "where", "while", "after", "before",
  "some", "such", "more", "most", "many", "other", "another",
  // short prepositions / auxiliaries (only useful as triplet middle)
  "of", "in", "on", "to", "for", "by", "as", "at",
  "is", "are", "was", "were", "be", "been", "being",
  "a", "an", "the",
  // relational words / longer prepositions (only useful as triplet middle)
  "between", "among", "across", "through", "within",
  "against", "during", "despite", "under", "over",
  "into", "onto", "upon", "from", "with", "without",
  // bridge verbs (useful as triplet middle)
  "use", "uses", "used", "using",
  "have", "has", "had", "having",
  "make", "makes", "made", "making",
  "take", "takes", "took", "taking",
  "give", "gives", "gave", "giving",
]);

/**
 * @ignore
 */
export default segmentText;