"use strict";

// ============================================================================
// core/parts/tokenize.js
//
// Port of core/parts/tokenize.cpp — the stream tokenizer. Splits raw text
// into a sequence of Word and Delimiter tokens in stream order, applying the
// same connector, number-prefix, and run-normalization rules as the C++.
//
// Works on byte-strings (see byteString.js): input is converted with
// toByteString so classification and lowercasing are byte-wise, matching
// std::string. Token `value`s are byte-strings; call fromByteString to
// display them.
// ============================================================================

import { toByteString } from "./byteString.js";

/**
 * Token category emitted by {@link tokenizeStream}.
 * @readonly
 * @enum {number}
 */
export const StreamTokenType = Object.freeze({
  Word: 0, // value is the lowercased word (byte-string)
  Delimiter: 1, // value is the normalized delimiter (byte-string)
});

// ---- Byte-code constants -------------------------------------------------
const TAB = 9;
const LF = 10;
const CR = 13;
const SP = 32;
const DOLLAR = 36; // $
const AMP = 38; // &
const APOS = 39; // '
const PLUS = 43; // +
const COMMA = 44; // ,
const DASH = 45; // -
const DOT = 46; // .

// ---- Character classification (ASCII-only by design) ---------------------
const isLetter = (c) => (c >= 97 && c <= 122) || (c >= 65 && c <= 90);
const isDigit = (c) => c >= 48 && c <= 57;
const isWordChar = (c) => isLetter(c) || isDigit(c);
const isSpace = (c) => c === SP;
const isTab = (c) => c === TAB;
const isNewline = (c) => c === LF || c === CR;
const isWhitespace = (c) => isSpace(c) || isTab(c) || isNewline(c);

const toLowerAscii = (c) => (c >= 65 && c <= 90 ? c + 32 : c);

const CAT_WORD = 0;
const CAT_WHITESPACE = 1;
const CAT_PUNCTUATION = 2;

const classify = (c) => {
  if (isWordChar(c)) return CAT_WORD;
  if (isWhitespace(c)) return CAT_WHITESPACE;
  return CAT_PUNCTUATION;
};

/**
 * Lowercase ASCII letters in a byte-string; other bytes unchanged.
 * @param {string} s byte-string.
 * @returns {string}
 */
export const asciiLowercase = (s) => {
  let out = "";
  for (let i = 0; i < s.length; i++) out += String.fromCharCode(toLowerAscii(s.charCodeAt(i)));
  return out;
};

// Number of leading sign/currency prefix chars at position p that begin a
// numeric literal ([+-]?\$? immediately followed by a digit, and not itself
// preceded by an alphanumeric). 0 means "not a number prefix here".
const numberPrefixLen = (raw, p) => {
  const n = raw.length;
  if (p >= n) return 0;
  if (p > 0 && isWordChar(raw.charCodeAt(p - 1))) return 0;
  let k = 0;
  if (p + k < n) {
    const c = raw.charCodeAt(p + k);
    if (c === DASH || c === PLUS) k++;
  }
  if (p + k < n && raw.charCodeAt(p + k) === DOLLAR) k++;
  if (k > 0 && p + k < n && isDigit(raw.charCodeAt(p + k))) return k;
  return 0;
};

const classifyWhitespaceRun = (run) => {
  if (run.length === 0) return "";
  let newlines = 0;
  let otherWs = 0;
  for (let i = 0; i < run.length; i++) {
    const c = run.charCodeAt(i);
    if (c === LF) newlines++;
    else if (c === CR) {
      /* ignored */
    } else if (isSpace(c) || isTab(c)) otherWs++;
  }
  if (newlines >= 2) return "\n\n";
  if (newlines === 1) return "\n";
  if (otherWs === 1 && run.length === 1 && run.charCodeAt(0) === SP) return " ";
  return "\t";
};

const classifyPunctuationRun = (run) => {
  if (run.length === 0) return "";
  if (run.length >= 3) {
    let allDots = true;
    for (let i = 0; i < run.length; i++) {
      if (run.charCodeAt(i) !== DOT) {
        allDots = false;
        break;
      }
    }
    if (allDots) return "...";
  }
  if (run.length >= 2) {
    let allDashes = true;
    for (let i = 0; i < run.length; i++) {
      if (run.charCodeAt(i) !== DASH) {
        allDashes = false;
        break;
      }
    }
    if (allDashes) return "--";
  }
  return run;
};

/**
 * Tokenize raw text into Word / Delimiter tokens.
 *
 * See tokenize.cpp for the full rule set: `-`/`&`/`'`/`,`/`.` in-word
 * connectors, acronym and decimal handling, leading `[+-]?\$?` number
 * prefixes, and whitespace/punctuation run normalization.
 *
 * @param {string|Uint8Array} raw UTF-8 text or raw bytes.
 * @returns {Array<{type: number, value: string}>} tokens (value is a
 *   byte-string).
 */
export const tokenizeStream = (raw) => {
  const s = toByteString(raw);
  const out = [];
  const n = s.length;
  if (n === 0) return out;

  let i = 0;
  let inQuote = false;

  while (i < n) {
    const startC = s.charCodeAt(i);
    const numPrefix = numberPrefixLen(s, i);

    if (numPrefix > 0 || classify(startC) === CAT_WORD) {
      // Word run, possibly leading with a number prefix.
      let j = numPrefix > 0 ? i + numPrefix : i + 1;
      while (j < n) {
        const c = s.charCodeAt(j);
        const left = s.charCodeAt(j - 1);
        const hasRight = j + 1 < n;
        const right = hasRight ? s.charCodeAt(j + 1) : 0;

        if (isWordChar(c)) {
          j++;
          continue;
        }
        if (c === DASH || c === AMP) {
          if (hasRight && isWordChar(right)) {
            j++;
            continue;
          }
          break;
        }
        if (c === APOS) {
          if (hasRight && isWordChar(right)) {
            j++;
            continue;
          }
          if (!inQuote) {
            j++;
            break;
          }
          break;
        }
        if (c === COMMA) {
          if (isDigit(left) && hasRight && isDigit(right)) {
            j++;
            continue;
          }
          break;
        }
        if (c === DOT) {
          // Decimal: digit on both sides.
          if (isDigit(left) && hasRight && isDigit(right)) {
            j++;
            continue;
          }
          // Acronym: right alnum, left isolated letter.
          if (hasRight && isWordChar(right) && isLetter(left)) {
            const leftIsolated = j === i + 1 ? true : s.charCodeAt(j - 2) === DOT;
            if (leftIsolated) {
              j++;
              continue;
            }
          }
          break;
        }
        break;
      }
      out.push({ type: StreamTokenType.Word, value: asciiLowercase(s.slice(i, j)) });
      i = j;
    } else {
      // Whitespace or Punctuation: maximal same-category run.
      const cat = classify(startC);
      let j = i + 1;
      while (j < n && classify(s.charCodeAt(j)) === cat) {
        if (cat === CAT_PUNCTUATION && numberPrefixLen(s, j) > 0) {
          // Break at a number prefix, but preserve em-dash priority: don't
          // break if every char from i..j is a dash (the `--` classifier
          // wants the full run).
          let allDashesSoFar = s.charCodeAt(j) === DASH;
          if (allDashesSoFar) {
            for (let k = i; k < j; k++) {
              if (s.charCodeAt(k) !== DASH) {
                allDashesSoFar = false;
                break;
              }
            }
          }
          if (!allDashesSoFar) break;
        }
        j++;
      }
      const run = s.slice(i, j);
      if (cat === CAT_WHITESPACE) {
        out.push({ type: StreamTokenType.Delimiter, value: classifyWhitespaceRun(run) });
      } else {
        out.push({ type: StreamTokenType.Delimiter, value: classifyPunctuationRun(run) });
        for (let k = 0; k < run.length; k++) {
          if (run.charCodeAt(k) === APOS) inQuote = !inQuote;
        }
      }
      i = j;
    }
  }

  return out;
};

/**
 * @ignore
 */
export default tokenizeStream;