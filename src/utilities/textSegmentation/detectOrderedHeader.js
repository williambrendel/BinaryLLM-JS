"use strict";

/**
 * @file detectOrderedHeader.js
 * @brief Detects ordered list / outline headers and their nesting depth.
 *
 * Recognizes prefixes like:
 *   `1.`, `A.`, `a.`, `iv.`, `IX.`
 *   `1.2.a.xvii`, `1. 2. a. xvii`
 *   `1)`, `A)`, `1.2)`, `(1)`, `(A)`
 *   `1-2-a`, `§1.2`, `1/2/a`
 *
 * Each "token" is a number, single letter, or roman numeral.
 * Tokens are separated by one of `.`, `)`, `-`, `/`, `:` followed by optional
 * whitespace. Level = number of tokens. Optional trailing closer like `.` or
 * `)` after the last token is allowed.
 *
 * @example
 *   detectOrderedHeader("1. Introduction")        // → { level: 1, prefix: "1.",            titleOffset: 3  }
 *   detectOrderedHeader("1.2.a.xvii Foo bar")     // → { level: 4, prefix: "1.2.a.xvii",    titleOffset: 11 }
 *   detectOrderedHeader("1. 2. a. xvii Foo")      // → { level: 4, prefix: "1. 2. a. xvii", titleOffset: 14 }
 *   detectOrderedHeader("(A) Hello")              // → { level: 1, prefix: "(A)",           titleOffset: 4  }
 *   detectOrderedHeader("Just a sentence.")       // → null
 */

// One token: digits, roman numeral (multi-char), or single letter.
// Order matters for the alternation: longer/more-specific first so the
// regex engine matches "iv" as roman before backtracking to single 'i'.
const TOKEN = /(?:\d+|[IVXLCDM]{2,7}|[ivxlcdm]{2,7}|[A-Za-z])/.source;

// Separator between tokens: one of . ) - / : — followed by optional whitespace.
// Must NOT be followed by another separator (keeps `..` from matching).
const SEP = /[.)\-/:](?!\s*[.)\-/:])\s*/.source;

// Optional opening bracket / paren / section sign at the very start.
const OPEN = /^\s*[(§]?\s*/.source;

// Optional closer after the last token: `.`, `)`, `:` or whitespace.
const CLOSE = /\s*[.):]?\s+/.source;

// Full prefix: one or more "TOKEN SEP" groups, optionally a trailing TOKEN
// without a separator (covers cases like `(A)` where `)` is the closer not a sep).
const PREFIX = new RegExp(
  OPEN +
  `(?:${TOKEN}${SEP})+` +     // at least one TOKEN-then-SEP
  `(?:${TOKEN})?` +            // optional final lone token
  CLOSE
);

// Counting regex: match TOKEN only when followed by a SEP, end-of-prefix, or
// closing punct — guarantees we don't double-count chars inside multi-letter
// tokens like "xvii".
const TOKEN_COUNT_RE = new RegExp(`${TOKEN}(?=[.)\\-/:\\s]|$)`, "g");

/**
 * @function detectOrderedHeader
 * @description Tests whether a text begins with an ordered-list / outline
 * prefix and returns its level, the matched prefix, and the offset where
 * the title text begins (with leading whitespace already consumed).
 *
 * Level is the number of ordering tokens in the prefix:
 *   "1."          → 1
 *   "1.2."        → 2
 *   "1.2.a."      → 3
 *   "1.2.a.xvii." → 4
 *
 * Whitespace between tokens is permitted (`"1. 2. a."` ≡ `"1.2.a."`).
 * Wrapping `()` or leading `§` are allowed and stripped from `prefix`.
 *
 * @param {string} text - Candidate header text.
 * 
 * @returns {{ level: number, prefix: string, titleOffset: number } | null}
 *   `titleOffset` is the index in `text` where the title begins (after the
 *   prefix and any whitespace). Returns `null` if no prefix is detected.
 */
export const detectOrderedHeader = text => {
  if (typeof text !== "string" || !text) return null;

  const m = PREFIX.exec(text);
  if (!m || m.index !== 0) return null;

  const matched = m[0];

  // Pull the inner ordering portion out of any surrounding ()/§/whitespace
  // so we can count tokens cleanly.
  const inner = matched.replace(/^\s*[(§]?\s*/, "").replace(/\s*[.):]?\s*$/, "");
  const tokens = inner.match(TOKEN_COUNT_RE);
  if (!tokens || !tokens.length) return null;

  return {
    level:       tokens.length,
    prefix:      matched.replace(/\s+$/, ""),
    titleOffset: matched.length,
  };
};

/**
 * @ignore
 */
export default detectOrderedHeader;