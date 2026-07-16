"use strict";

/**
 * @file tokenize.js
 * @brief Stream tokenizer — splits raw text into Word and Delimiter tokens.
 *
 * Port of `core/parts/tokenize.cpp`. Splits raw text into a sequence of Word
 * and Delimiter tokens in stream order, applying the same connector,
 * number-prefix, and run-normalization rules as the C++ implementation.
 *
 * Works on byte-strings (see `byteString.js`): input is converted with
 * {@link toByteString} so classification and lowercasing are byte-wise,
 * matching `std::string`. Token `value`s are byte-strings; call
 * `fromByteString` to display them.
 *
 * **Exported surface:**
 * - `StreamTokenType` — enum tagging each emitted token (`Word` / `Delimiter`).
 * - `asciiLowercase(s)` → byte-string with ASCII letters lowercased.
 * - `tokenizeStream(raw)` → `Array<{ type, value }>` in stream order.
 *
 * @see {@link tokenizeStream}
 * @see {@link StreamTokenType}
 */

import { toByteString } from "./byteString.js";

/**
 * @constant StreamTokenType
 * @description Token category emitted by {@link tokenizeStream}. A frozen
 * enum: `Word` tokens carry the lowercased word (byte-string) and `Delimiter`
 * tokens carry the normalized whitespace or punctuation run (byte-string).
 * @readonly
 * @enum {number}
 *
 * @example
 * StreamTokenType.Word        // → 0
 * StreamTokenType.Delimiter   // → 1
 *
 * @example
 * // Inspecting a token's category:
 * const [tok] = tokenizeStream("hi");
 * tok.type === StreamTokenType.Word   // → true
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
const PLUS = 43; // +
const COMMA = 44; // ,
const DASH = 45; // -
const DOT = 46; // .

// ---- Character classification (ASCII-only by design) ---------------------

/**
 * @description Test whether a byte code is an ASCII letter (`A`–`Z` or `a`–`z`).
 * @param {number} c - Byte code to test.
 * @returns {boolean} `true` if `c` is an ASCII letter.
 */
const isLetter = (c) => (c >= 97 && c <= 122) || (c >= 65 && c <= 90);

/**
 * @description Test whether a byte code is an ASCII decimal digit (`0`–`9`).
 * @param {number} c - Byte code to test.
 * @returns {boolean} `true` if `c` is an ASCII digit.
 */
const isDigit = (c) => c >= 48 && c <= 57;

/**
 * @description Test whether a byte code is a word character (ASCII letter or digit).
 * @param {number} c - Byte code to test.
 * @returns {boolean} `true` if `c` is a letter or digit.
 */
const isWordChar = (c) => isLetter(c) || isDigit(c);

/**
 * @description Test whether a byte code is a space (`SP`, 0x20).
 * @param {number} c - Byte code to test.
 * @returns {boolean} `true` if `c` is a space.
 */
const isSpace = (c) => c === SP;

/**
 * @description Test whether a byte code is a horizontal tab (`TAB`, 0x09).
 * @param {number} c - Byte code to test.
 * @returns {boolean} `true` if `c` is a tab.
 */
const isTab = (c) => c === TAB;

/**
 * @description Test whether a byte code is a newline byte (`LF` or `CR`).
 * @param {number} c - Byte code to test.
 * @returns {boolean} `true` if `c` is `LF` (0x0A) or `CR` (0x0D).
 */
const isNewline = (c) => c === LF || c === CR;

/**
 * @description Test whether a byte code is any whitespace (space, tab, or newline).
 * @param {number} c - Byte code to test.
 * @returns {boolean} `true` if `c` is a space, tab, `LF`, or `CR`.
 */
const isWhitespace = (c) => isSpace(c) || isTab(c) || isNewline(c);

/**
 * @description Lowercase a single ASCII byte code: uppercase letters (`A`–`Z`)
 * are shifted by 32; all other byte codes are returned unchanged.
 * @param {number} c - Byte code to lowercase.
 * @returns {number} The lowercased byte code, or `c` if it is not `A`–`Z`.
 */
const toLowerAscii = (c) => (c >= 65 && c <= 90 ? c + 32 : c);

const CAT_WORD = 0;
const CAT_WHITESPACE = 1;
const CAT_PUNCTUATION = 2;

/**
 * @description Classify a byte code into one of the three tokenizer categories.
 * Word characters (letters/digits) yield `CAT_WORD`, whitespace bytes yield
 * `CAT_WHITESPACE`, and everything else (including non-ASCII bytes) falls
 * through to `CAT_PUNCTUATION`.
 * @param {number} c - Byte code to classify.
 * @returns {number} `CAT_WORD`, `CAT_WHITESPACE`, or `CAT_PUNCTUATION`.
 */
const classify = (c) => {
  if (isWordChar(c)) return CAT_WORD;
  if (isWhitespace(c)) return CAT_WHITESPACE;
  return CAT_PUNCTUATION;
};

/**
 * @function asciiLowercase
 * @description Lowercase the ASCII letters (`A`–`Z`) in a byte-string; every
 * other byte is left unchanged. Operates byte-wise, so non-ASCII bytes and
 * digits/punctuation pass through untouched.
 *
 * @param {string} s - Byte-string to lowercase.
 * @returns {string} A new byte-string with ASCII uppercase letters lowered.
 *
 * @example
 * asciiLowercase("Hello")     // → "hello"
 * asciiLowercase("ABC-123")   // → "abc-123"  (digits and dash unchanged)
 * asciiLowercase("already")   // → "already"
 */
export const asciiLowercase = (s) => {
  let out = "";
  for (let i = 0; i < s.length; i++) out += String.fromCharCode(toLowerAscii(s.charCodeAt(i)));
  return out;
};

/**
 * @description Compute the length of the sign/currency prefix that begins a
 * numeric literal at position `p`. A prefix is an optional leading `-`/`+`
 * followed by an optional `$`, matching `[+-]?\$?`, that is immediately
 * followed by a digit and is not itself preceded by a word character. The
 * return value is the number of prefix bytes (0, 1, or 2); `0` means there is
 * no number prefix at `p`.
 *
 * @param {string} raw - Byte-string being scanned.
 * @param {number} p - Index at which to test for a number prefix.
 * @returns {number} Count of leading prefix bytes (`0`–`2`); `0` when `p` does
 *   not start a numeric literal.
 *
 * @example
 * numberPrefixLen("$3", 0)     // → 1   ("$" before a digit)
 * numberPrefixLen("-$5", 0)    // → 2   ("-$" before a digit)
 * numberPrefixLen("a-3", 1)    // → 0   (preceded by word char "a")
 * numberPrefixLen("+x", 0)     // → 0   (no digit follows)
 */
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

/**
 * @description Normalize a maximal whitespace run into a single canonical
 * delimiter value. Only `LF` bytes count as newlines (`CR` bytes are ignored);
 * spaces and tabs count as "other" whitespace. The result is chosen by
 * priority: two or more newlines collapse to `"\n\n"` (paragraph break), a
 * single newline to `"\n"`, a lone single space to `" "`, and anything else
 * (tabs, multiple spaces, mixed space/tab runs) to `"\t"`.
 *
 * @param {string} run - Byte-string containing only whitespace bytes.
 * @returns {string} The normalized delimiter: `""`, `"\n\n"`, `"\n"`, `" "`,
 *   or `"\t"`.
 *
 * @example
 * classifyWhitespaceRun(" ")        // → " "
 * classifyWhitespaceRun("   ")      // → "\t"   (more than one space)
 * classifyWhitespaceRun("\n\n\n")   // → "\n\n" (two or more newlines)
 * classifyWhitespaceRun("\n")       // → "\n"
 */
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

/**
 * @description Normalize a maximal punctuation run into a canonical delimiter
 * value. A run of three or more `.` bytes collapses to an ellipsis `"..."`; a
 * run of two or more `-` bytes collapses to an em-dash `"--"`. Any other run
 * (including mixed punctuation) is returned unchanged.
 *
 * @param {string} run - Byte-string containing the punctuation run.
 * @returns {string} `"..."` for all-dot runs of length ≥ 3, `"--"` for
 *   all-dash runs of length ≥ 2, otherwise `run` itself (`""` if empty).
 *
 * @example
 * classifyPunctuationRun("....")   // → "..."
 * classifyPunctuationRun("---")    // → "--"
 * classifyPunctuationRun("?!")     // → "?!"  (unchanged)
 */
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
 * @function tokenizeStream
 * @description Tokenize raw text into `Word` / `Delimiter` tokens in stream
 * order. The hyphen (`-`) is the only in-word letter connector; word runs also
 * keep in-number `,`/`.` (thousands and decimals), acronym dots (`U.S.A`), and
 * a leading `[+-]?\$?` number prefix. The apostrophe (`'`) and ampersand (`&`)
 * are NOT connectors — they split off as their own delimiter tokens.
 * Whitespace and punctuation runs are collapsed into normalized delimiter
 * values. Each token's `type` is a {@link StreamTokenType} value and its
 * `value` is a byte-string.
 *
 * @param {string|Uint8Array} raw - UTF-8 text or raw bytes to tokenize.
 * @returns {Array<{type: number, value: string}>} Tokens in stream order;
 *   each `value` is a byte-string (empty input yields `[]`).
 *
 * @example
 * tokenizeStream("Hi there")
 * // → [
 * //     { type: 0, value: "hi" },      // Word
 * //     { type: 1, value: " " },       // Delimiter
 * //     { type: 0, value: "there" }    // Word
 * //   ]
 *
 * @example
 * // Number prefix ($) and thousands comma stay inside one Word.
 * tokenizeStream("$3,000")
 * // → [ { type: 0, value: "$3,000" } ]
 *
 * @example
 * // Empty input yields no tokens.
 * tokenizeStream("")   // → []
 */
export const tokenizeStream = (raw) => {
  const s = toByteString(raw);
  const out = [];
  const n = s.length;
  if (n === 0) return out;

  let i = 0;

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
        // Hyphen is an in-word connector between two word chars ("co-op").
        // The apostrophe and ampersand are NOT: they split off as delimiters.
        if (c === DASH) {
          if (hasRight && isWordChar(right)) {
            j++;
            continue;
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