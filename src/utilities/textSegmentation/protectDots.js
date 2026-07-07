/**
 * @file dotProtection.js
 * @brief
 * Utilities for safely segmenting text on sentence punctuation (. ! ?) without
 * false positives on dot-containing patterns such as decimal numbers, acronyms,
 * outline headers, filenames, URLs, emails, and common abbreviations.
 *
 * The approach is a placeholder-substitution pipeline:
 *   1. {@link protectDots} scans the input text and replaces every dot-bearing
 *      pattern with a unique placeholder token, returning both the protected
 *      text and a dictionary mapping tokens back to their originals.
 *   2. The caller performs sentence segmentation on the protected text using
 *      whatever splitting strategy they prefer (the remaining "." characters
 *      are now genuine sentence terminators).
 *   3. {@link restore} swaps the placeholders in each segment back to their
 *      original strings.
 *
 * Tokens use NULL control characters (\x00) as delimiters to virtually
 * guarantee no collision with real text content.
 *
 * The default pattern set ({@link DOT_PROTECTION_PATTERNS}) is tuned for
 * scientific and technical prose. Patterns are applied in order of specificity
 * so that longer matches (e.g. "1.2.3") are protected before shorter ones
 * (e.g. "1.") could partially consume them.
 *
 * @example
 *   import protectDots, { restore } from './protectDots';
 *   const { protectedText, dictionary } = protectDots("Dr. Smith found pi = 3.14. Done.");
 *   const segments = protectedText.split(/(?<=[.!?])\s+/);
 *   const restored = segments.map(s => restore(s, dictionary));
 */
"use strict";

/**
 * @function protectDots
 * @description
 * Protects dot-containing patterns in text by replacing them with unique
 * placeholder tokens, so the text can be safely split on sentence punctuation
 * without false positives on decimals, acronyms, headers, etc.
 *
 * The returned dictionary maps each placeholder back to the original string,
 * allowing restoration after segmentation.
 *
 * Patterns are protected in order of specificity (longest/most specific first)
 * to avoid partial matches mangling later rules.
 *
 * Protected patterns:
 *  - URLs (http://..., https://..., www....)
 *  - Email addresses (user@domain.tld)
 *  - File paths / filenames with extensions (e.g. file.txt, foo.tar.gz)
 *  - Multi-level outline numbering (1.2.3, A.1.b)
 *  - Decimal numbers (3.14, .5, 1,234.56)
 *  - Acronyms / initials (U.S.A., A.B.C.)
 *  - Single-token outline headers at line start (1., A., a., vii.)
 *  - Common honorifics / abbreviations (Dr., Mr., Mrs., Ms., Prof., vs., etc., e.g., i.e., cf., Fig., Eq.)
 *
 * @param {string} text - The input text to protect.
 * 
 * @returns {{ protectedText: string, dictionary: Object<string, string> }}
 *          An object containing the modified text with placeholders, and a
 *          dictionary mapping each placeholder token to its original string.
 *
 * @example
 *   const { protectedText, dictionary } = protectDots("Dr. Smith found pi = 3.14. See Fig. 2.");
 *   // protectedText: "«TOK_0» Smith found pi = «TOK_1». See «TOK_2» 2."
 *   // dictionary: { "«TOK_0»": "Dr.", "«TOK_1»": "3.14", "«TOK_2»": "Fig." }
 */
export const protectDots = text => {
  // Coerce non-string input
  if (text == null) return { protectedText: "", dictionary: {} };
  typeof text === "string" || (text = `${text}`);

  // Init.
  const dictionary = {};
  let counter = 0;

  let result = text;
  for (const pattern of DOT_PROTECTION_PATTERNS) {
    result = result.replace(pattern, (match) => {
      // Preserve any leading whitespace/newline captured by the outline-header pattern
      const leadingWs = match.match(/^\s*/)[0];
      const core = match.slice(leadingWs.length);
      const token = makeToken(counter++);
      dictionary[token] = core;
      return leadingWs + token;
    });
  }

  return { protectedText: result, dictionary };
}

/**
 * @constant
 * @type {RegExp[]}
 * @description
 * Default regex patterns for dot-protection, applied in order. Each pattern
 * matches a substring containing one or more "." that should NOT be treated
 * as a sentence boundary. Order matters: earlier (more specific) patterns
 * take precedence over later ones.
 */
const DOT_PROTECTION_PATTERNS = [
  // URLs
  /\b(?:https?|ftp):\/\/[^\s]+|\bwww\.[^\s]+/gi,

  // Emails
  /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g,

  // Acronyms / initials — BEFORE multi-level so "U.S.A." wins over "U.S.A"
  /\b(?:[A-Za-z]\.){2,}/g,

  // Multi-level outline numbering (1.2.3, A.1.b) — 2+ dot groups
  /\b(?:[A-Za-z]+|\d+)(?:\.(?:[A-Za-z]+|\d+)){2,}\b/g,

  // Decimal numbers
  /\b\d{1,3}(?:,\d{3})+\.\d+\b|\b\d+\.\d+\b|(?<!\w)\.\d+\b/g,

  // Filenames
  /\b[\w-]+(?:\.[A-Za-z0-9]{1,8}){1,2}\b/g,

  // Outline headers at line start
  /(?:^|\n)\s*(?:\d+|[A-Za-z]|[ivxlcdmIVXLCDM]+)\.(?=\s)/g,

  // Honorifics
  /\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|Mt|Fig|Eq|Ref|Vol|No|pp|p|vs|etc|cf|al|approx|Inc|Ltd|Co|Corp)\.|\b(?:e\.g|i\.e|et\sal)\./g,
];

/**
 * @function makeToken
 * @private
 * @description
 * Builds a placeholder token using NULL control characters as delimiters,
 * which virtually guarantees no collision with real text content.
 *
 * @param {number} val - A unique numeric id for this token within the document.
 * @returns {string} A placeholder token of the form "\x00TOK_<val>\x00".
 */
const makeToken = val => `\x00TOK_${val}\x00`;

/**
 * @function restore
 * @description
 * Reverses {@link protectDots} by replacing every placeholder token in the
 * text with its original string from the dictionary.
 *
 * @param {string} text - Text containing placeholder tokens.
 * @param {Object<string, string>} dictionary - Mapping from token to original string,
 *        as returned by {@link protectDots}.
 * @returns {string} The text with all placeholders restored.
 *
 * @example
 *   const { protectedText, dictionary } = protectDots(input);
 *   const segments = protectedText.split(/[.!?]\s+/);
 *   const restored = segments.map(s => restore(s, dictionary));
 */
export const restore = (text, dictionary) => {
  let result = text;
  for (const [token, original] of Object.entries(dictionary)) {
    result = result.replaceAll(token, original);
  }
  return result;
}

/**
 * @ignore
 */
export default protectDots;