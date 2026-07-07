"use strict";

// ============================================================================
// core/parts/kind.js
//
// Port of core/parts/part.hpp — the position role of a part and the length
// bounds shared by the extractor and the decomposer.
//
// A Part is one basis element in the F-dim part space. Its identity is the
// pair (kind, value); the same character sequence in different positions is a
// distinct basis element ("-ing", "ing-" and "-ing-" are three parts).
// ============================================================================

/**
 * Position role of a part. Numeric values match the C++ `enum class Kind`.
 * @readonly
 * @enum {number}
 */
export const Kind = Object.freeze({
  Start: 0, // anchored to word start: "un", "re", "trans"
  End: 1, // anchored to word end: "ing", "ed", "tion"
  Mid: 2, // appears in word interior: "port", "graph", "rate"
  Letter: 3, // positional letter singleton: "a##", "##a##", "##a"
  Whole: 4, // short whole-word atom: "the", "am", "i" (lowercased)
  Delimiter: 5, // punctuation or whitespace token, kept literally
});

// Length bounds for Start/Mid/End enumeration and matching. The extractor
// generates candidates at every L in [kMinPartLength, kMaxPartLength] and the
// decomposer searches the same range — these MUST stay in sync. Length 1 is
// reserved for the positional Letter singleton path (the coverage backstop).
export const kMinPartLength = 2;
export const kMaxPartLength = 7;

// Sentinel ID for "not in dictionary." Matches C++ static_cast<uint32_t>(-1).
export const kInvalidPartId = 0xffffffff;

const KIND_TO_STRING = Object.freeze({
  [Kind.Start]: "start",
  [Kind.End]: "end",
  [Kind.Mid]: "mid",
  [Kind.Letter]: "letter",
  [Kind.Whole]: "whole",
  [Kind.Delimiter]: "delim",
});

const STRING_TO_KIND = Object.freeze({
  start: Kind.Start,
  end: Kind.End,
  mid: Kind.Mid,
  letter: Kind.Letter,
  whole: Kind.Whole,
  delim: Kind.Delimiter,
});

/**
 * Stringify a Kind for serialization and debug.
 * @param {number} kind
 * @returns {string}
 */
export const kindToString = (kind) => KIND_TO_STRING[kind] ?? "?";

/**
 * Parse a Kind from its string form.
 * @param {string} s
 * @returns {number|undefined} the Kind, or undefined if unrecognized.
 */
export const parseKind = (s) => STRING_TO_KIND[s];

/**
 * @ignore
 */
export default Kind;
