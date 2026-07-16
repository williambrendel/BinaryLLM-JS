"use strict";

/**
 * @file kind.js
 * @brief Position role of a part plus the length bounds shared by the
 * extractor and the decomposer.
 *
 * Port of `core/parts/part.hpp`. A Part is one basis element in the F-dim
 * part space. Its identity is the pair `(kind, value)`; the same character
 * sequence in different positions is a distinct basis element — `"-ing"`,
 * `"ing-"` and `"-ing-"` are three parts.
 *
 * **Exported surface:**
 * - `Kind` — frozen enum of position roles (numeric values match the C++
 *   `enum class Kind`).
 * - `kMinPartLength` / `kMaxPartLength` — inclusive length bounds for
 *   Start/Mid/End enumeration and matching.
 * - `kInvalidPartId` — sentinel ID for "not in dictionary."
 * - `kindToString(kind)` → `string`
 * - `parseKind(s)` → `number | undefined`
 *
 * @example
 * kindToString(Kind.Start)   // → "start"
 * parseKind("end")           // → Kind.End
 */

/**
 * @enum {number}
 * @readonly
 * @description Position role of a part. Numeric values match the C++
 * `enum class Kind`:
 * - `Start` — anchored to word start: `"un"`, `"re"`, `"trans"`, and single
 *   chars at the word start (`"c"`).
 * - `End` — anchored to word end: `"ing"`, `"ed"`, `"tion"`, and single chars at
 *   the word end (`"t"`).
 * - `Mid` — appears in word interior: `"port"`, `"graph"`, `"rate"`, and interior
 *   single chars (`"a"`).
 * - `Whole` — a whole word, including single-letter words: `"the"`, `"am"`,
 *   `"i"`, `"a"` (lowercased).
 * - `Delimiter` — punctuation or whitespace token, kept literally.
 *
 * There is no separate "letter" kind: a single character is simply a length-1
 * `Start`/`Mid`/`End` fragment (or a `Whole` when it is the word). Position is
 * carried by the kind, not by `##` markers baked into the value — the `##`
 * convention exists only in the serialized text form.
 *
 * @example
 * Kind.Start        // → 0
 * Kind.Delimiter    // → 4
 * kindToString(Kind.Mid)   // → "mid"
 */
export const Kind = Object.freeze({
  Start: 0, // anchored to word start: "un", "re", "trans", "c"
  End: 1, // anchored to word end: "ing", "ed", "tion", "t"
  Mid: 2, // appears in word interior: "port", "graph", "rate", "a"
  Whole: 3, // a whole word incl. single letters: "the", "am", "i", "a" (lowercased)
  Delimiter: 4, // punctuation or whitespace token, kept literally
});

/**
 * @constant {number}
 * @description Minimum (inclusive) length for Start/Mid/End enumeration and
 * matching. The extractor generates candidates at every `L` in
 * `[kMinPartLength, kMaxPartLength]` and the decomposer searches the same
 * range — these MUST stay in sync. Length 1 is handled by the seeded single-char
 * `Start`/`Mid`/`End` atoms (the coverage backstop), not by learned enumeration.
 *
 * @example
 * for (let L = kMinPartLength; L <= kMaxPartLength; L++) {
 *   // L ranges 2..7 inclusive
 * }
 */
export const kMinPartLength = 2;

/**
 * @constant {number}
 * @description Maximum (inclusive) length for Start/Mid/End enumeration and
 * matching. Paired with {@link kMinPartLength}.
 *
 * @example
 * kMaxPartLength   // → 7
 */
export const kMaxPartLength = 7;

/**
 * @constant {number}
 * @description Sentinel ID for "not in dictionary." Matches C++
 * `static_cast<uint32_t>(-1)`.
 *
 * @example
 * kInvalidPartId               // → 0xffffffff
 * id === kInvalidPartId        // → true when a part is absent from the dictionary
 */
export const kInvalidPartId = 0xffffffff;

const KIND_TO_STRING = Object.freeze({
  [Kind.Start]: "start",
  [Kind.End]: "end",
  [Kind.Mid]: "mid",
  [Kind.Whole]: "whole",
  [Kind.Delimiter]: "delim",
});

const STRING_TO_KIND = Object.freeze({
  start: Kind.Start,
  end: Kind.End,
  mid: Kind.Mid,
  whole: Kind.Whole,
  delim: Kind.Delimiter,
});

/**
 * @function positionalKind
 * @description Maps a word-boundary position to the fragment {@link Kind} for a
 * substring (typically a single character): word-start → `Start`, word-end →
 * `End`, interior → `Mid`. A lone fragment that is both start and end (`atStart
 * && atEnd`) resolves to `Start` — the "start-wins" convention shared by the
 * decomposer's fill and the BPE peel's leftover-char fill.
 *
 * @param {boolean} atStart - The fragment sits at the word start.
 * @param {boolean} atEnd - The fragment sits at the word end.
 * @returns {number} `Kind.Start`, `Kind.Mid`, or `Kind.End`.
 *
 * @example
 * positionalKind(true, false)   // → Kind.Start
 * positionalKind(false, false)  // → Kind.Mid
 * positionalKind(false, true)   // → Kind.End
 * positionalKind(true, true)    // → Kind.Start  (lone char, start-wins)
 */
export const positionalKind = (atStart, atEnd) =>
  atEnd && !atStart ? Kind.End : !atStart && !atEnd ? Kind.Mid : Kind.Start;

/**
 * @function kindToString
 * @description Stringifies a {@link Kind} for serialization and debug.
 * Unrecognized values fall back to `"?"`.
 *
 * @param {number} kind - A {@link Kind} enum value.
 * @returns {string} The lowercase string form, or `"?"` if unrecognized.
 *
 * @example
 * kindToString(Kind.Start)       // → "start"
 * kindToString(Kind.Delimiter)   // → "delim"
 * kindToString(99)               // → "?"  (unrecognized)
 */
export const kindToString = (kind) => KIND_TO_STRING[kind] ?? "?";

/**
 * @function parseKind
 * @description Parses a {@link Kind} from its string form — the inverse of
 * {@link kindToString}.
 *
 * @param {string} s - A lowercase kind string (`"start"`, `"end"`, `"mid"`,
 *   `"whole"`, `"delim"`).
 * @returns {number|undefined} The {@link Kind}, or `undefined` if unrecognized.
 *
 * @example
 * parseKind("end")     // → Kind.End
 * parseKind("delim")   // → Kind.Delimiter
 * parseKind("nope")    // → undefined  (unrecognized)
 */
export const parseKind = (s) => STRING_TO_KIND[s];

/**
 * @ignore
 */
export default Kind;
