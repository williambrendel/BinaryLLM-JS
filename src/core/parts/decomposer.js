"use strict";

/**
 * @file decomposer.js
 * @brief Greedy decomposition of words and delimiters into dictionary part IDs.
 *
 * Port of `core/parts/decomposer.cpp` — a single "peel" algorithm shared by
 * three exported entry points:
 *
 * - `decomposeWord`      — word → left-to-right sequence of part IDs.
 * - `findSingletonRuns`  — the contiguous uncovered ranges the same peel leaves
 *   behind, which the extractor's peel loop turns into new candidates.
 * - `decomposeDelimiter` — delimiter token → part IDs, per-byte on miss.
 *
 * **Algorithm.** Try a Whole-word fast path first; otherwise enumerate every
 * registered Start/End/Mid candidate substring, order them by `(dict ID, pos)`,
 * then greedily claim non-overlapping ranges — at most one Start and one End,
 * while Mid may fire repeatedly. Any position still uncovered is filled with a
 * positional Letter singleton whose spelling encodes where it sits:
 * `"x##"` (start), `"##y##"` (mid), or `"##z"` (end).
 *
 * All strings are byte-strings (see `byteString.js`).
 *
 * **Exported surface:**
 * - `decomposeWord(dict, word)`      → `number[]`
 * - `findSingletonRuns(dict, word)`  → `Array<{ start, end, touchesWordStart, touchesWordEnd }>`
 * - `decomposeDelimiter(dict, delim)` → `number[]`
 *
 * @see {@link decomposeWord}
 * @see {@link findSingletonRuns}
 */

import { Kind, kMinPartLength, kMaxPartLength, kInvalidPartId } from "./kind.js";

// ---- Letter singleton (## position marker) -------------------------------
const emitLetter = (dict, c, atStart, atEnd) => {
  let s;
  if (atEnd && !atStart) s = "##" + c;
  else if (!atStart && !atEnd) s = "##" + c + "##";
  else s = c + "##"; // atStart (with or without atEnd) defaults to start encoding
  return dict.lookup(Kind.Letter, s);
};

// ---- Candidate enumeration -----------------------------------------------
// For every L in [kMin, kMax] and every valid position, ask the dict whether
// the substring is a registered Start/End/Mid. Collect the hits.
const enumerateCandidates = (dict, word) => {
  const n = word.length;
  const out = [];
  for (let L = kMinPartLength; L <= kMaxPartLength; L++) {
    if (L > n) break;
    // Start at pos 0.
    {
      const id = dict.lookup(Kind.Start, word.slice(0, L));
      if (id !== kInvalidPartId) out.push({ pos: 0, L, kind: Kind.Start, id });
    }
    // End at pos n-L.
    {
      const id = dict.lookup(Kind.End, word.slice(n - L, n));
      if (id !== kInvalidPartId) out.push({ pos: n - L, L, kind: Kind.End, id });
    }
    // Mid at every valid position.
    for (let pos = 0; pos + L <= n; pos++) {
      const id = dict.lookup(Kind.Mid, word.slice(pos, pos + L));
      if (id !== kInvalidPartId) out.push({ pos, L, kind: Kind.Mid, id });
    }
  }
  return out;
};

// Sort by (id asc, pos asc): dict ID order == insertion order == whatever
// priority the dict's builder/reorderer shaped.
const sortCandidatesById = (cands) => {
  cands.sort((a, b) => (a.id !== b.id ? a.id - b.id : a.pos - b.pos));
};

const rangeIsFree = (claimed, begin, end) => {
  for (let i = begin; i < end; i++) if (claimed[i]) return false;
  return true;
};

const claimRange = (claimed, begin, end) => {
  for (let i = begin; i < end; i++) claimed[i] = true;
};

// Apply the greedy peel to a pre-sorted candidate list.
const peel = (n, candidates) => {
  const claimed = new Array(n).fill(false);
  const emitted = [];
  let hasStartClaim = false;
  let hasEndClaim = false;
  for (const c of candidates) {
    if (c.kind === Kind.Start && hasStartClaim) continue;
    if (c.kind === Kind.End && hasEndClaim) continue;
    if (!rangeIsFree(claimed, c.pos, c.pos + c.L)) continue;
    claimRange(claimed, c.pos, c.pos + c.L);
    emitted.push([c.pos, c.id]);
    if (c.kind === Kind.Start) hasStartClaim = true;
    if (c.kind === Kind.End) hasEndClaim = true;
  }
  return { emitted, claimed, hasStartClaim, hasEndClaim };
};

/**
 * @function decomposeWord
 * @description Decomposes a single (lowercased, byte-string) word into part IDs,
 * in left-to-right stream order.
 *
 * If the word is registered as a Whole entry, it takes the fast path and returns
 * that single ID. Otherwise the greedy peel claims the best Start/End/Mid
 * candidates and every remaining position is filled with a positional Letter
 * singleton, so the returned sequence always covers the whole word.
 *
 * @param {import("./dictionary.js").PartDictionary} dict
 *   Part dictionary used to look up candidate substrings and Letter singletons.
 * @param {string} word - The word to decompose, as a byte-string.
 * @returns {number[]} Part IDs in left-to-right order. Empty for an empty word.
 *
 * @example
 * // Whole fast path: "the" is registered as a Whole entry.
 * decomposeWord(dict, "the")   // → [<Whole id for "the">]  (single id)
 *
 * @example
 * // No candidates: every position falls back to a positional Letter singleton.
 * decomposeWord(dict, "xyz")
 * // → [ dict.lookup(Kind.Letter, "x##"),    // start
 * //     dict.lookup(Kind.Letter, "##y##"),  // mid
 * //     dict.lookup(Kind.Letter, "##z") ]   // end
 */
export const decomposeWord = (dict, word) => {
  const n = word.length;
  if (n === 0) return [];

  // Whole-word fast path.
  if (dict.hasWhole(word)) {
    const id = dict.lookup(Kind.Whole, word);
    if (id !== kInvalidPartId) return [id];
  }

  const cands = enumerateCandidates(dict, word);
  sortCandidatesById(cands);
  const pr = peel(n, cands);

  // Letter singleton fill.
  for (let p = 0; p < n; p++) {
    if (pr.claimed[p]) continue;
    const atStart = p === 0 && !pr.hasStartClaim;
    const atEnd = p === n - 1 && !pr.hasEndClaim;
    const id = emitLetter(dict, word[p], atStart, atEnd);
    if (id !== kInvalidPartId) pr.emitted.push([p, id]);
  }

  pr.emitted.sort((a, b) => a[0] - b[0]);
  return pr.emitted.map((e) => e[1]);
};

/**
 * @function findSingletonRuns
 * @description Returns the contiguous uncovered (Letter-singleton) ranges left
 * behind by the same peel that {@link decomposeWord} uses — consumed by the
 * extractor's peel loop. Each run's `start` is inclusive and `end` is exclusive.
 *
 * A Whole-word match claims the entire word, so it yields no runs. When nothing
 * is claimed the whole word is returned as a single run touching both ends.
 *
 * @param {import("./dictionary.js").PartDictionary} dict
 *   Part dictionary used to look up candidate substrings.
 * @param {string} word - The word to scan, as a byte-string.
 * @returns {Array<{start: number, end: number, touchesWordStart: boolean, touchesWordEnd: boolean}>}
 *   One entry per uncovered run. `touchesWordStart`/`touchesWordEnd` flag runs
 *   that reach the word's first/last position. Empty when the word is empty or
 *   fully claimed (e.g. a Whole match).
 *
 * @example
 * // No candidates claim any position, so the whole word is one run.
 * findSingletonRuns(dict, "abc")
 * // → [{ start: 0, end: 3, touchesWordStart: true, touchesWordEnd: true }]
 *
 * @example
 * // "the" is a Whole entry — fully claimed, so no runs remain.
 * findSingletonRuns(dict, "the")   // → []
 */
export const findSingletonRuns = (dict, word) => {
  const runs = [];
  const n = word.length;
  if (n === 0) return runs;

  if (dict.hasWhole(word)) {
    const id = dict.lookup(Kind.Whole, word);
    if (id !== kInvalidPartId) return runs;
  }

  const cands = enumerateCandidates(dict, word);
  sortCandidatesById(cands);
  const pr = peel(n, cands);

  let p = 0;
  while (p < n) {
    if (pr.claimed[p]) {
      p++;
      continue;
    }
    const runStart = p;
    while (p < n && !pr.claimed[p]) p++;
    const runEnd = p;
    runs.push({
      start: runStart,
      end: runEnd,
      touchesWordStart: runStart === 0,
      touchesWordEnd: runEnd === n,
    });
  }
  return runs;
};

/**
 * @function decomposeDelimiter
 * @description Decomposes a delimiter token into part IDs. If the whole token is
 * a registered Delimiter it emits a single ID; otherwise it falls back to
 * per-byte emission, looking up each byte as its own Delimiter and skipping any
 * that are unknown.
 *
 * @param {import("./dictionary.js").PartDictionary} dict
 *   Part dictionary used to look up Delimiter entries.
 * @param {string} delim - The delimiter token, as a byte-string.
 * @returns {number[]} Part IDs for the token. Empty for an empty token.
 *
 * @example
 * // "-" is a registered Delimiter — emitted as a single id.
 * decomposeDelimiter(dict, "-")   // → [dict.lookup(Kind.Delimiter, "-")]
 *
 * @example
 * // Unknown multi-byte token — falls back to per-byte lookup of "-" and "-".
 * decomposeDelimiter(dict, "--")
 * // → [dict.lookup(Kind.Delimiter, "-"), dict.lookup(Kind.Delimiter, "-")]
 */
export const decomposeDelimiter = (dict, delim) => {
  const out = [];
  if (delim.length === 0) return out;

  if (dict.hasDelimiter(delim)) {
    const id = dict.lookup(Kind.Delimiter, delim);
    if (id !== kInvalidPartId) {
      out.push(id);
      return out;
    }
  }

  for (let i = 0; i < delim.length; i++) {
    const s = delim[i];
    if (dict.hasDelimiter(s)) {
      const id = dict.lookup(Kind.Delimiter, s);
      if (id !== kInvalidPartId) out.push(id);
    }
  }
  return out;
};

/**
 * @ignore
 */
export default decomposeWord;