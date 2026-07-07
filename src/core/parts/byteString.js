"use strict";

// ============================================================================
// core/parts/byteString.js
//
// The C++ pipeline operates on std::string, i.e. raw bytes: tokenization,
// substring, length, and lexicographic ordering are all byte-wise. To
// reproduce its output byte-for-byte — including how multi-byte UTF-8
// delimiters (curly quotes, em-dashes in the French/Spanish corpora) are
// split and ordered — this port works in "byte-strings": ordinary JS strings
// whose every char code is a single byte in [0,255].
//
// Because all code units are <= 0xFF, JS string `<` (compare by UTF-16 code
// unit) coincides exactly with C++ std::string `operator<` (unsigned byte
// compare), and `.length`/`.substr`/indexing coincide with byte offsets.
//
// `toByteString` maps input to that representation; `fromByteString` maps a
// byte-string back to a normal UTF-8 JS string for display. Uses the
// portable Web {TextEncoder,TextDecoder} (present in browsers and Node), so
// nothing here depends on the Node Buffer API.
// ============================================================================

const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8");

// String.fromCharCode.apply blows the argument stack on large arrays; chunk it.
const CHUNK = 0x8000;

/**
 * Convert input to a byte-string (one char per byte).
 * @param {string|Uint8Array} input UTF-8 text, or raw bytes.
 * @returns {string} a byte-string with char codes in [0,255].
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
 * Convert a byte-string back to a normal UTF-8 JS string.
 * @param {string} s a byte-string (char codes in [0,255]).
 * @returns {string} decoded UTF-8 text.
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