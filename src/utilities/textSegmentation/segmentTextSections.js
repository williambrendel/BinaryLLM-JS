"use strict";

/**
 * @file segmentTextSections.js
 * @brief Hierarchical grouping of {@link Segment} instances into
 * {@link Section} trees, with header-level-driven nesting.
 *
 * Calls {@link segmentText} to produce sentence-level segments (some of
 * which may be {@link Header} instances), then groups them into a tree of
 * {@link Section} nodes. The tree mirrors the document's heading outline:
 *
 * - A {@link Header} of level `N` opens a new {@link Section}.
 * - Subsequent segments of any kind nest inside that section until a header
 *   of level `≤ N` closes it.
 * - A header of level `< N` walks back up the tree until it finds a parent
 *   whose level can contain it; same-level headers become siblings.
 *
 * Body segments (non-headers) attach to the current open header section as
 * direct children, alongside any nested child sections. Use
 * {@link Section#content} to filter out the header, or
 * `section.filter(x => typeof x[0] === "number")` to get only body segments.
 *
 * Body that appears before the first header is preserved as top-level
 * sections at the root, with paragraph breaks (`\n\n+`) splitting them.
 *
 * Use {@link Section#flatten} on the result if you need a flat depth-first
 * traversal — the tree is the canonical structure, flat is the projection.
 *
 * **Exported surface:**
 * - Default export: `segmentTextSections(text)` → `Section[]`
 * - `segmentTextSections.Segment` — re-exported {@link Segment} class.
 * - `segmentTextSections.Section` — re-exported {@link Section} class.
 * - `segmentTextSections.Header`  — re-exported {@link Header} class.
 * - `segmentTextSections.segmentText` — re-exported {@link segmentText} function.
 *
 * @see {@link segmentText} for sentence-level segmentation.
 * @see {@link Section} for paragraph geometry methods and `.flatten()`.
 * @see {@link Header} for heading metadata (level, title).
 */

import segmentText from "./segmentText.js";
import Section from "./Section.js";
import Header from "./Header.js";

/**
 * @function segmentTextSections
 * @description Segments `text` via {@link segmentText} and assembles the
 * results into a hierarchical tree of {@link Section} nodes. The tree
 * mirrors the document's structure along two axes: heading depth and
 * paragraph breaks.
 *
 * Two kinds of {@link Section} appear in the output:
 *
 * - **Header sections** — a `Section` whose first child is a {@link Header}.
 *   Created whenever a header segment is encountered. Exposes `.header` and
 *   `.level`, and may contain nested header sections (deeper levels) and/or
 *   paragraph sections (its body content).
 * - **Paragraph sections** — a `Section` containing only body
 *   {@link Segment} instances. Created on every blank-line break (`\n\n+`)
 *   between body segments. Has no header; `.header` and `.level` return
 *   `undefined`. Within a paragraph section, segments separated by single
 *   newlines or sentence punctuation alone (no blank line) accumulate as
 *   siblings.
 *
 * **Algorithm:**
 * 1. A stack of `{ section, level }` frames tracks currently-open header
 *    sections, ordered shallowest → deepest. Only header sections push
 *    onto the stack — paragraph sections do not, because they cannot
 *    contain other sections.
 * 2. A `currentParagraph` slot tracks the most recently opened paragraph
 *    section, into which subsequent body segments accumulate. The slot
 *    resets to `null` whenever a {@link Header} is encountered, so the
 *    next body segment after a header always starts a fresh paragraph.
 * 3. On a {@link Header} at level `L`:
 *    - Pop frames where `frame.level >= L`.
 *    - Build a new header section, attach it to the current parent
 *      (stack top, or root), and push `{ section, level: L }` as the
 *      new top frame.
 *    - Reset `currentParagraph` to `null`.
 * 4. On a body segment:
 *    - If `segment.hasBlankLineBefore` is set, OR no current paragraph
 *      exists, build a new paragraph section, attach it to the current
 *      parent (stack top, or root), and remember it as `currentParagraph`.
 *    - Otherwise, append to `currentParagraph`.
 *
 * **Single-child unwrap.** When the document parses to exactly one
 * top-level section, that section is returned directly instead of wrapped
 * in the root. Callers needing a uniform shape can re-wrap if necessary;
 * the unwrap saves a level of indirection in the common case where a
 * document has a single top-level header.
 *
 * @param {string|*} text - Input text. Forwarded to {@link segmentText}.
 *   Falsy values return an empty {@link Section}.
 *
 * @returns {Section} A {@link Section} representing the document tree.
 *   For multi-section documents this is the root container holding all
 *   top-level sections. For single-section documents this is that section
 *   itself (see "single-child unwrap" above). For empty input this is an
 *   empty `Section` (length 0).
 *
 * @example
 * // Single header with two body sentences in one paragraph
 * segmentTextSections("# Title\n\nFirst sentence. Second sentence.");
 * // → Section [
 * //     Header(0, 7, level=1),
 * //     Section [                 // paragraph
 * //       Segment(9, 23),         // "First sentence"
 * //       Segment(25, 40)         // "Second sentence"
 * //     ]
 * //   ]
 *
 * @example
 * // Header with two body paragraphs
 * segmentTextSections("# Title\n\nFirst paragraph.\n\nSecond paragraph.");
 * // → Section [
 * //     Header(level=1, "Title"),
 * //     Section [ Segment("First paragraph") ],
 * //     Section [ Segment("Second paragraph") ]
 * //   ]
 *
 * @example
 * // Nested headers — level-2 inside level-1
 * segmentTextSections("# Top\n\nIntro.\n\n## Sub\n\nBody.");
 * // → Section [
 * //     Header(level=1, "Top"),
 * //     Section [ Segment("Intro") ],
 * //     Section [
 * //       Header(level=2, "Sub"),
 * //       Section [ Segment("Body") ]
 * //     ]
 * //   ]
 *
 * @example
 * // Higher-level header pops back up to root
 * segmentTextSections("## A\n\n### B\n\n# C\n\nbody.");
 * // → Section [           ← root, returned because length > 1
 * //     Section [ Header(level=2, "A"), Section [ Header(level=3, "B") ] ],
 * //     Section [ Header(level=1, "C"), Section [ Segment("body") ] ]
 * //   ]
 *
 * @example
 * // Body before any header — preserved as top-level paragraph sections
 * segmentTextSections("Intro paragraph.\n\n# Title\n\nBody.");
 * // → Section [           ← root, returned because length > 1
 * //     Section [ Segment("Intro paragraph") ],   // headerless paragraph
 * //     Section [ Header(level=1, "Title"), Section [ Segment("Body") ] ]
 * //   ]
 *
 * @example
 * // Single-segment document — unwrapped to the paragraph section
 * segmentTextSections("Hello.");
 * // → Section [ Segment(0, 5) ]   ← the paragraph itself, not wrapped in root
 *
 * @example
 * // Walk the tree depth-first
 * const tree = segmentTextSections(text);
 * for (const node of tree.flatten()) {
 *   if (node instanceof Header) console.log(node.level, node.extractTitle(text));
 * }
 *
 * @example
 * // Distinguish header sections from paragraph sections
 * const isHeaderSection = s => s instanceof Section && s.header !== undefined;
 * const isParagraphSection = s => s instanceof Section && s.header === undefined;
 */
export const segmentTextSections = text => {
  const root = new Section();

  const segments = segmentText(text, true);
  if (!segments.length) return root;

  const stack = []; // entries: { section, level }
  let currentParagraph = null;

  for (let i = 0, l = segments.length, segment; i !== l; ++i) {
    segment = segments[i];

    if (segment instanceof Header) {
      // Pop frames that cannot contain a header at this level.
      // A frame can contain `segment` only if its level is strictly less.
      const level = segment.level;
      while (stack.length && stack[stack.length - 1].level >= level) {
        stack.pop();
      }

      // Reset on header.
      currentParagraph = null;

      // Build the new section with the header as its first element.
      const section = new Section();
      section.push(segment);

      // Attach to current parent — top of stack, or root if empty.
      (stack.length && stack[stack.length - 1].section || root).push(section);

      // Push new section to stack.
      stack.push({ section, level });

      continue;
    }

    if (segment.hasBlankLineBefore || !currentParagraph) {
      // Create a new section.
      currentParagraph = new Section();

      // Push the current segment to it.
      currentParagraph.push(segment);

      // Attach to current parent — top of stack, or root if empty.
      (stack.length && stack[stack.length - 1].section || root).push(currentParagraph);

      continue;
    }

    // Push to current paragraph section.
    currentParagraph.push(segment);
  }

  // Return sections or the single section child.
  return root.length === 1 && (root[0] instanceof Section) && root[0] || root;
};

/**
 * @ignore
 */
export default segmentTextSections;