"use strict";

/**
 * @file byteString.js
 * @brief UTF-8 ⇄ latin1 "byte-string" conversion for byte-exact string ops.
 *
 * The C++ pipeline operates on `std::string`, i.e. raw bytes: tokenization,
 * substring, length, and lexicographic ordering are all byte-wise. To
 * reproduce its output byte-for-byte — including how multi-byte UTF-8
 * delimiters (curly quotes, em-dashes in the French/Spanish corpora) are
 * split and ordered — this port works in "byte-strings": ordinary JS strings
 * whose every char code is a single byte in `[0, 255]`.
 *
 * Because all code units are `<= 0xFF`, JS string `<` (compare by UTF-16 code
 * unit) coincides exactly with C++ `std::string` `operator<` (unsigned byte
 * compare), and `.length` / `.substr` / indexing coincide with byte offsets.
 *
 * **Exported surface:**
 * - `toByteString(input)` → maps UTF-8 text or raw bytes to that representation.
 * - `fromByteString(s)`   → maps a byte-string back to a normal UTF-8 JS string
 *   for display.
 *
 * Uses the portable Web `{TextEncoder, TextDecoder}` (present in browsers and
 * Node), so nothing here depends on the Node Buffer API.
 *
 * @example
 * toByteString("café").length         // → 5  ("é" is 2 UTF-8 bytes)
 * fromByteString(toByteString("café")) // → "café"  (round-trips)
 */

const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8");

// String.fromCharCode.apply blows the argument stack on large arrays; chunk it.
const CHUNK = 0x8000;

/**
 * @function toByteString
 * @description Convert input to a byte-string (one char per byte).
 *
 * When `input` is a JS string it is UTF-8 encoded first, so multi-byte
 * characters expand: each resulting byte becomes one char code in `[0, 255]`.
 * When `input` is already a `Uint8Array` of raw bytes, those bytes are mapped
 * directly to char codes. Encoding is done in chunks because
 * `String.fromCharCode.apply` overflows the argument stack on large arrays.
 *
 * @param {string|Uint8Array} input - UTF-8 text, or raw bytes.
 * @returns {string} A byte-string with every char code in `[0, 255]`.
 *
 * @example
 * toByteString("AB")                       // → "AB"  (ASCII, unchanged)
 * toByteString("café").length              // → 5     ("é" → 2 bytes: 0xC3 0xA9)
 * [...toByteString("é")].map(c => c.charCodeAt(0))  // → [195, 169]
 * toByteString(new Uint8Array([65, 66]))   // → "AB"  (raw bytes → char codes)
 */
export const toByteString = (input) => {
  const bytes = typeof input === "string" ? encoder.encode(input) : input;
  let out = "";
  for (let i = 0; i < bytes.length; i += CHUNK) {
    out += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return out;
};

/**
 * @function fromByteString
 * @description Convert a byte-string back to a normal UTF-8 JS string.
 *
 * Each char code is masked to a single byte (`& 0xff`) and collected into a
 * `Uint8Array`, which is then decoded as UTF-8. This is the inverse of
 * {@link toByteString}: multi-byte sequences are reassembled into their
 * original characters.
 *
 * @param {string} s - A byte-string (char codes in `[0, 255]`).
 * @returns {string} The decoded UTF-8 text.
 *
 * @example
 * fromByteString("AB")                              // → "AB"
 * fromByteString(String.fromCharCode(195, 169))     // → "é"  (2 bytes → 1 char)
 * fromByteString(toByteString("café"))              // → "café"  (round-trips)
 */
export const fromByteString = (s) => {
  const bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i) & 0xff;
  return decoder.decode(bytes);
};

/**
 * @ignore
 */
export default toByteString;