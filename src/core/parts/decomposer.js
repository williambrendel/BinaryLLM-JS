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
 * length-1 fragment atom whose kind encodes where it sits: `Start "x"`,
 * `Mid "y"`, or `End "z"` (bare value — no `##` markers).
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

import { Kind, kMinPartLength, kMaxPartLength, kInvalidPartId, positionalKind } from "./kind.js";

// ---- Single-char fragment fill (length-1 Start/Mid/End atom) --------------
/**
 * @function emitLetter
 * @description Looks up the length-1 fragment part ID for a single character `c`,
 * choosing the kind that encodes where the character sits within its word:
 * {@link Kind.Start} at the word start, {@link Kind.End} at the word end,
 * {@link Kind.Mid} in the interior. When both `atStart` and `atEnd` are set (a
 * lone character), the Start kind is used (start-wins, via {@link positionalKind}).
 * The value looked up is the bare character — there is no `##`-marked "letter".
 *
 * @param {import("./dictionary.js").PartDictionary} dict
 *   Part dictionary used to look up the fragment entry.
 * @param {string} c - The single character (byte) to encode.
 * @param {boolean} atStart - Whether the character sits at the word's start.
 * @param {boolean} atEnd - Whether the character sits at the word's end.
 * @returns {number} The fragment part ID, or `kInvalidPartId` if unregistered.
 *
 * @example
 * emitLetter(dict, "y", false, false)  // → dict.lookup(Kind.Mid, "y")
 * emitLetter(dict, "x", true,  false)  // → dict.lookup(Kind.Start, "x")
 * emitLetter(dict, "z", false, true)   // → dict.lookup(Kind.End, "z")
 */
const emitLetter = (dict, c, atStart, atEnd) => {
  return dict.lookup(positionalKind(atStart, atEnd), c);
};

// ---- Candidate enumeration -----------------------------------------------
/**
 * @function enumerateCandidates
 * @description Enumerates every registered Start/End/Mid substring of `word`
 * that could participate in the greedy peel. For each length `L` in
 * `[kMinPartLength, kMaxPartLength]` (capped at the word length) it probes three
 * placements against the dictionary:
 *
 * - **Start** — the prefix `word[0..L)`, looked up as `Kind.Start`.
 * - **End** — the suffix `word[n-L..n)`, looked up as `Kind.End`.
 * - **Mid** — every substring `word[pos..pos+L)` for all valid `pos`, looked up
 *   as `Kind.Mid`.
 *
 * Only lookups that resolve to a valid part ID are kept. The result is an
 * unordered flat list of candidate records; {@link sortCandidatesById} imposes
 * the peel priority and {@link peel} consumes it.
 *
 * @param {import("./dictionary.js").PartDictionary} dict
 *   Part dictionary used to look up Start/End/Mid substrings.
 * @param {string} word - The word to scan, as a byte-string.
 * @returns {Array<{pos: number, L: number, kind: number, id: number}>}
 *   One record per registered candidate: `pos` is the inclusive start offset,
 *   `L` the length, `kind` the {@link Kind}, and `id` the resolved part ID.
 */
export const enumerateCandidates = (dict, word) => {
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

/**
 * @function sortCandidatesById
 * @description Sorts a candidate list in place by ascending part ID, breaking
 * ties by ascending position. Dictionary ID order mirrors insertion order,
 * which is whatever priority the dict's builder/reorderer shaped — so lower IDs
 * are peeled first, giving the greedy claim in {@link peel} its precedence.
 *
 * @param {Array<{pos: number, L: number, kind: number, id: number}>} cands
 *   Candidate records (as produced by {@link enumerateCandidates}); mutated in place.
 * @returns {void}
 */
const sortCandidatesById = (cands) => {
  cands.sort((a, b) => (a.id !== b.id ? a.id - b.id : a.pos - b.pos));
};

/**
 * @function rangeIsFree
 * @description Reports whether the half-open span `[begin, end)` is entirely
 * unclaimed in the `claimed` bitmap — i.e. no position within it has been taken
 * by a previously peeled candidate.
 *
 * @param {boolean[]} claimed - Per-position occupancy flags.
 * @param {number} begin - Inclusive start of the span to test.
 * @param {number} end - Exclusive end of the span to test.
 * @returns {boolean} `true` if every position in `[begin, end)` is free.
 */
const rangeIsFree = (claimed, begin, end) => {
  for (let i = begin; i < end; i++) if (claimed[i]) return false;
  return true;
};

/**
 * @function claimRange
 * @description Marks every position in the half-open span `[begin, end)` as
 * claimed in the `claimed` bitmap. Mutates `claimed` in place.
 *
 * @param {boolean[]} claimed - Per-position occupancy flags (mutated in place).
 * @param {number} begin - Inclusive start of the span to claim.
 * @param {number} end - Exclusive end of the span to claim.
 * @returns {void}
 */
const claimRange = (claimed, begin, end) => {
  for (let i = begin; i < end; i++) claimed[i] = true;
};

/**
 * @function peel
 * @description Applies the greedy peel to a pre-sorted candidate list — the
 * shared core of {@link decomposeWord} and {@link findSingletonRuns}.
 *
 * Walking the candidates in priority order (see {@link sortCandidatesById}), it
 * claims the first candidate whose span is still free, subject to two rules:
 * at most one Start and one End may ever be claimed (tracked by `hasStartClaim`
 * / `hasEndClaim`), whereas Mid candidates may fire any number of times.
 * Candidates that overlap an already-claimed span, or that would exceed the
 * single-Start / single-End budget, are skipped. Positions left uncovered are
 * not filled here — the callers turn them into length-1 fragment atoms or runs.
 *
 * @param {number} n - Length of the word being decomposed.
 * @param {Array<{pos: number, L: number, kind: number, id: number}>} candidates
 *   Candidate records, pre-sorted by {@link sortCandidatesById}.
 * @returns {{emitted: Array<[number, number]>, claimed: boolean[], hasStartClaim: boolean, hasEndClaim: boolean}}
 *   `emitted` is a list of `[pos, id]` pairs for each claimed candidate (in
 *   claim order, not position order); `claimed` is the final per-position
 *   occupancy bitmap; `hasStartClaim`/`hasEndClaim` record whether a Start/End
 *   candidate was consumed.
 */
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
 * candidates and every remaining position is filled with a length-1 fragment
 * atom, so the returned sequence always covers the whole word.
 *
 * @param {import("./dictionary.js").PartDictionary} dict
 *   Part dictionary used to look up candidate substrings and length-1 fragments.
 * @param {string} word - The word to decompose, as a byte-string.
 * @returns {number[]} Part IDs in left-to-right order. Empty for an empty word.
 *
 * @example
 * // Whole fast path: "the" is registered as a Whole entry.
 * decomposeWord(dict, "the")   // → [<Whole id for "the">]  (single id)
 *
 * @example
 * // No candidates: every position falls back to a length-1 fragment atom.
 * decomposeWord(dict, "xyz")
 * // → [ dict.lookup(Kind.Start, "x"),   // start
 * //     dict.lookup(Kind.Mid, "y"),     // mid
 * //     dict.lookup(Kind.End, "z") ]    // end
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

  // Single-char fragment fill (length-1 Start/Mid/End atom).
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
 * @description Returns the contiguous uncovered (fragment-fill) ranges left
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