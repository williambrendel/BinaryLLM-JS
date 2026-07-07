"use strict";

import { interval, intersects, contains } from "./interval";
import Segment from "./Segment";

/**
 * @file Section.js
 * @brief Paragraph-level container holding an ordered collection of
 * {@link Segment} instances, with range geometry and text extraction.
 *
 * `Section` extends `Array` so instances are directly iterable, spreadable,
 * and passable wherever arrays are expected. Each element is a {@link Segment}
 * or a nested `Section`.
 *
 * When produced by {@link segmentMarkdownTextSections}, `this[0]` may be a
 * {@link Header} instance. The `.header`, `.level`, and `.content` getters
 * expose heading metadata and degrade gracefully on plain sections.
 *
 * `interval` is exported as `Section.interval` so {@link Segment} can import
 * it without a circular require — `Section` has no dependency on `Segment`.
 */

/**
 * @class Section
 * @extends Array
 * @description Ordered collection of {@link Segment} instances representing
 * one logical paragraph of text. Extends `Array` so instances are iterable
 * and can be passed directly as arrays.
 *
 * Construction — use `new Section()` then `.push(segment)`, or spread
 * segments as arguments: `new Section(seg1, seg2)`. Do **not** pass a single
 * `Segment` (a `Uint32Array`) as the only constructor argument — `Array`'s
 * constructor treats a single non-numeric object as `[value]`, which happens
 * to work but is an undocumented side effect. Prefer explicit `.push()`.
 *
 * When `this[0]` is a {@link Header}, the section exposes structured heading
 * getters. These are safe to call on any `Section` — they return `undefined`
 * or `[]` when no header is present.
 *
 * @example
 * const section = new Section();
 * section.push(new Segment(0, 11));
 * section.push(new Segment(13, 24));
 * section.start;         // → 0
 * section.end;           // → 24
 * section.extract(text); // → full paragraph text
 */
export class Section extends Array {
  constructor(...args) {
    super(...args);
  }

  /**
   * @property {number|undefined} start - Start offset of the first element.
   * Handles both Segment (reads `[0]`) and nested Section (reads `.start`) as
   * the first element via the `typeof first[0] === "number"` fallback.
   */
  get start() {
    const first = this[0];
    if (!first) return undefined;
    return typeof first[0] === "number" ? first[0] : first.start;
  }

  /**
   * @property {number|undefined} end - End offset of the last element.
   * Handles both Segment (reads `[1]`) and nested Section (reads `.end`) as
   * the last element via the `typeof last[1] === "number"` fallback.
   */
  get end() {
    const last = this[this.length - 1];
    if (!last) return undefined;
    return typeof last[1] === "number" ? last[1] : last.end;
  }

  /** @property {number} span - Total character span of this section. */
  get span() { return this.end - this.start; }

  /** @property {Segment|Section|undefined} first - First element. */
  get first() { return this[0]; }

  /** @property {Segment|Section|undefined} last - Last element. */
  get last() { return this[this.length - 1]; }

  /**
   * @property {Header|undefined} header
   * The first element if it is a {@link Header} (`constructor.isHeader === true`
   * or has an own `level` property), otherwise `undefined`.
   * Safe to call on plain sections — returns `undefined`.
   */
  get header() {
    return (
      this.hasOwnProperty("inheritedHeader") && this.inheritedHeader || (
        (this[0]?.hasOwnProperty("level") || (this[0]?.constructor.isHeader)) && this[0]
      ) || undefined
    );
  }

  /**
   * @property {number|undefined} level
   * Heading depth from the {@link Header}. Returns `undefined` on plain sections.
   */
  get level() { return this[0]?.level; }

  /**
   * @property {Array} content
   * All elements after the header when one is present, otherwise all elements.
   * Returns a plain `Array` (not a `Section`) in document order.
   *
   * The returned array carries two additional non-enumerable getters:
   * - `.start` — start offset of the first element (Segment or nested Section).
   * - `.end`   — end offset of the last element.
   *
   * These mirror the span of all direct body content, making it easy to get
   * the body byte range without filtering manually.
   *
   * Filter to separate body segments from child sections using structural typing:
   * ```js
   * section.content.filter(x => typeof x[0] === "number") // body segments
   * section.content.filter(x => typeof x[0] !== "number") // child sections
   * ```
   */

  get content() {
    const h = this[0]?.hasOwnProperty("level") || this[0]?.constructor.isHeader;
    return h && Object.defineProperty(this.slice(1), "header", {
      get() { return null; }
    }) || this;
  }

  /**
   * @method toString
   * @description Serializes the segment as a plain `[start, end] children level` string.
   * 
   * @returns {number[]}
   */
  toString() {
    return `Section [${this.start}, ${this.end}] children: ${this.length} level: ${this.level}`;
  }

  /**
   * @method extract
   * @description Extracts the full paragraph text from the source string,
   * spanning from the start of the first element to the end of the last.
   * @param {string} text - The original source text.
   * @returns {string}
   */
  extract(text) {
    return text.slice(this.start, this.end);
  }

  /**
   * @method intersectsWith
   * @description Returns `true` if this section overlaps with the given range.
   * Touching ranges (end === other.start) do not count as intersecting.
   * @param {...*} input - Any form accepted by {@link interval}.
   * @returns {boolean}
   */
  intersectsWith(...input) {
    return intersects(this, interval(...input));
  }

  /**
   * @method isWithin
   * @description Returns `true` if this section is entirely contained inside
   * the given range (inclusive on both ends).
   * @param {...*} input - Any form accepted by {@link interval}.
   * @returns {boolean}
   */
  isWithin(...input) {
    return contains(interval(...input), this);
  }

  /**
   * @method contains
   * @description Returns `true` if this section fully contains the given range.
   * @param {...*} input - Any form accepted by {@link interval}.
   * @returns {boolean}
   */
  contains(...input) {
    return contains(this, interval(...input));
  }

  /**
   * @method flatten
   * @description Returns a flat array containing this section followed by all
   * of its descendants in depth-first document order.
   *
   * The first element is always `this` section itself. Then for each element
   * in the section:
   * - If the element has a `.flatten()` method (i.e. it is a nested `Section`),
   *   its flattened subtree is spread in — so all nested sections and their
   *   segments appear recursively.
   * - Otherwise the element is emitted as a {@link Segment}. If it is already
   *   a `Segment` instance it is used as-is; if not (e.g. a plain `[s,e]`
   *   array) it is wrapped with `new Segment(child)`.
   *
   * This is useful for index building, vectorization pipelines, or any pass
   * that needs to visit every node in the section tree without manual recursion.
   *
   * @returns {Array<Section|Segment>} Flat array starting with `this`, followed
   *   by all descendant sections and segments in depth-first order.
   *
   * @example
   * // Flat section
   * const s = new Section();
   * s.push(new Segment(0, 5));
   * s.push(new Segment(7, 12));
   * s.flatten(); // → [s, Segment(0,5), Segment(7,12)]
   *
   * @example
   * // Nested sections
   * const parent = new Section();
   * const child  = new Section();
   * child.push(new Segment(10, 20));
   * parent.push(new Segment(0, 5));
   * parent.push(child);
   * parent.flatten();
   * // → [parent, Segment(0,5), child, Segment(10,20)]
   *
   * @example
   * // Collect all segments from a markdown tree
   * const allSegments = sections
   *   .flatMap(s => s.flatten())
   *   .filter(x => x instanceof Segment);
   */
  flatten() {
    let output = [this];
    for (let i = 0, l = this.length, child; i !== l; ++i) {
      (child = this[i]) && (
        typeof child.flatten === "function" && output.push(...child.flatten())
        || (
          output.push(child instanceof Segment && child || new Segment(child))
        )
      );
    }

    return output;
  }

  /**
   * @property {Section} directContent
   * @description Returns a new {@link Section} containing only this section's
   * **direct** {@link Segment} children — including any leading {@link Header}.
   * Nested {@link Section} children (paragraph sections, child header sections)
   * are excluded.
   *
   * The result is a freshly constructed `Section`, not a reference into the
   * original tree. This means:
   * - The result has no nested sections, so it's flat and embeddable as-is.
   * - The result's `.start`/`.end` cover only the kept segments, which may be
   *   narrower than the original section's range if paragraph children were
   *   dropped.
   * - `result.header` is preserved when the original had a {@link Header} as
   *   its first child (since `Header extends Segment`, headers pass through
   *   the `instanceof Segment` filter).
   *
   * Useful for producing self-contained "leaf chunks" of a section's own
   * direct content, without descending into nested subsections. Pair with
   * {@link Section#contentSections} to walk a tree and collect every such
   * chunk that actually contains body content.
   *
   * @returns {Section} A new `Section` holding direct {@link Segment} children
   *   of `this`, in original order. Returns an empty `Section` if `this` has
   *   no direct segment children.
   *
   * @example
   *   // Header section under the new model: [Header, paragraphSection]
   *   const section = new Section();
   *   section.push(new Header(0, 8, 1));
   *   section.push(paragraphSection);   // contains a Segment
   *   section.directContent;
   *   // → Section [Header(0, 8)]   ← paragraphSection dropped, Header kept
   *
   * @example
   *   // Paragraph section: just segments, no header
   *   const para = new Section();
   *   para.push(new Segment(10, 20));
   *   para.push(new Segment(22, 30));
   *   para.directContent;
   *   // → Section [Segment(10, 20), Segment(22, 30)]   ← unchanged shape
   *
   * @example
   *   // Structural-only section (only nested sections, no direct segments)
   *   const top = new Section();
   *   top.push(new Header(0, 5, 1));
   *   top.push(childHeaderSection);
   *   top.directContent;
   *   // → Section [Header(0, 5)]   ← only the header survives
   *
   * @see {@link Section#contentSections}
   */
  get directContent() {
    const section = new Section();
    for (let i = 0, l = this.length, child; i !== l; ++i) {
      ((child = this[i]) instanceof Segment) && section.push(child);
    }
    return section;
  }

  /**
   * @method contentSections
   * @description Walks this section's subtree depth-first and returns the
   * `directContent` of every {@link Section} that carries actual body content
   * — that is, at least one non-{@link Header} {@link Segment} as a direct
   * child. Each returned chunk carries an `.ancestors` property: the array
   * of ancestor {@link Header} instances on the path from `this` down to
   * (but not including) the chunk itself.
   *
   * Sections whose only direct segment is a {@link Header} (typical of
   * structural-only sections that exist purely to nest other sections) are
   * filtered out. Sections with no direct segments are also dropped.
   *
   * The returned `Section`s are **freshly constructed** by
   * {@link Section#directContent} — they are not references into the original
   * tree. Each result is a flat `[Header?, Segment, Segment, ...]` chunk
   * suitable for embedding, vectorization, or any pipeline that wants a
   * self-contained leaf-style unit of text per section.
   *
   * The "has content" check uses `length - !!header`, which evaluates to the
   * number of non-header segments in the direct content. A section with only
   * a header (`length === 1`, header set) yields `0` (excluded); a header
   * with one body segment yields `1` (included); a paragraph section with
   * two segments yields `2` (included).
   *
   * **Breadcrumb semantics.** The `.ancestors` array on each chunk lists
   * {@link Header} instances from outermost to innermost. The chunk's own
   * header (if any) is **not** in `.ancestors` — it lives on `.header` of
   * the chunk itself. To render a full title path including the chunk's
   * own title, append `chunk.header` to `chunk.ancestors`.
   *
   * The same `ancestors` array is shared by reference among all chunks at
   * the same depth — do not mutate it. The implementation creates a new
   * array only when descending past a header section.
   *
   * @returns {Section[]} Array of fresh `Section` instances in document
   *   order. Each result has an additional `.ancestors` property of type
   *   `Header[]`.
   *
   * @example
   *   // Document:
   *   // # Article
   *   //   ## Methods
   *   //     "We did X. Then Y."
   *   //   ## Results
   *   //     "It worked."
   *   //
   *   tree.contentSections();
   *   // → [
   *   //     Section[Header("Methods"), Segment("We did X"), Segment("Then Y")]
   *   //       .ancestors = [Header("Article")],
   *   //     Section[Header("Results"), Segment("It worked")]
   *   //       .ancestors = [Header("Article")]
   *   //   ]
   *
   * @example
   *   // RAG indexing with breadcrumb context
   *   for (const chunk of tree.contentSections()) {
   *     const trail = chunk.ancestors
   *       .map(h => h.extractTitle(text))
   *       .join(" > ");
   *     const ownTitle = chunk.header?.extractTitle(text) ?? "";
   *     const breadcrumb = ownTitle ? `${trail} > ${ownTitle}` : trail;
   *
   *     const body = chunk
   *       .filter(c => !(c instanceof Header))
   *       .map(c => c.extract(text))
   *       .join(" ");
   *
   *     await embed(`${breadcrumb}\n\n${body}`);
   *   }
   *
   * @see {@link Section#directContent}
   * @see {@link Section#flatten}
   */
  contentSections() {
    const out = [];
    const walk = (node, ancestors, inheritedHeader, paragraphIndex) => {
      if (!(node instanceof Section)) return;

      const dc = node.directContent;
      const ownHeader = dc.header;

      (dc.length - !!ownHeader) && (
        (!ownHeader && inheritedHeader) && (
          dc.inheritedHeader = inheritedHeader,
          paragraphIndex !== undefined && (dc.paragraph = paragraphIndex)
        ),
        dc.ancestors = ancestors,
        out.push(dc)
      );

      const nextAncestors = node.header && [...ancestors, node.header] || ancestors;

      // Count paragraph (headerless Section) children
      let paraCount = 0;
      for (let i = 0, l = node.length; i !== l; ++i) {
        (node[i] instanceof Section && !node[i].header) && ++paraCount;
      }
      const isMultiPara = paraCount > 1;

      let paraIdx = 0;
      for (let i = 0, l = node.length, child; i !== l; ++i) {
        if (!((child = node[i]) instanceof Section)) continue;

        const childAncestors = child.header ? nextAncestors : ancestors;
        const childInherited = child.header ? inheritedHeader : (node.header ?? inheritedHeader);
        const childParaIdx = (!child.header && isMultiPara) ? paraIdx++ : undefined;

        walk(child, childAncestors, childInherited, childParaIdx);
      }
    };
    walk(this, [], undefined, undefined);
    return out;
  }

  /**
   * @method printPartition
   * @description Renders this section's tree as an indented, human-readable
   * outline. Each node sits on its own line, indented by its heading depth.
   * Useful for debugging and quick visual verification of parser output.
   *
   * @returns {string} A newline-separated indented rendering of the subtree.
   *
   * @example
   *   const tree = segmentTextSections("# Top\n\nIntro.\n\n## Sub\n\nBody.");
   *   console.log(tree.printPartition());
   *   //   Section [...]
   *   //     Header [0,5]
   *   //     Section [...]
   *   //     Segment [7,13]
   *   //       Section [...]
   *   //       Header [15,21]
   *   //       Segment [23,28]
   */
  printPartition() {
    return this.flatten().reduce((out, s) => (
      (s instanceof Section) && (out.level = s.level - 1),
      out.txt += `\n${"  ".repeat(out.level)}${s}`,
      s.level && (out.level = s.level),
      out
    ), {
      level: 0,
      txt: ""
    }).txt
  }
}

Section.isSection = true;

/**
 * @function Section.create
 * @description Factory method. Equivalent to `new Section(...args)`.
 * @param {...*} args
 * @returns {Section}
 */
export const create = Section.create = (...args) => new Section(...args);

/**
 * @ignore
 */
export default Object.freeze(Section);