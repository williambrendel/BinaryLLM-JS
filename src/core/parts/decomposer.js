"use strict";

// ============================================================================
// core/parts/decomposer.js
//
// Port of core/parts/decomposer.cpp — one peel algorithm, two callers:
//
//   decomposeWord        — public; word -> sequence of part IDs
//   findSingletonRuns    — internal; the length->=2 uncovered ranges the
//                          extractor's peel loop turns into new candidates
//
// Algorithm: Whole fast path; else enumerate every Start/End/Mid candidate,
// order by (dict ID, pos), greedily claim non-overlapping ranges (one Start
// and one End max; Mid may fire repeatedly), then fill any uncovered
// position with a positional Letter singleton ("##" markers).
//
// All strings are byte-strings (see byteString.js).
// ============================================================================

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
 * Decompose a single (lowercased, byte-string) word into part IDs, in
 * left-to-right stream order.
 * @param {import("./dictionary.js").PartDictionary} dict
 * @param {string} word byte-string.
 * @returns {number[]}
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
 * The uncovered (Letter-singleton) ranges left by the same peel — used by the
 * extractor's peel loop. start inclusive, end exclusive.
 * @param {import("./dictionary.js").PartDictionary} dict
 * @param {string} word byte-string.
 * @returns {Array<{start: number, end: number, touchesWordStart: boolean, touchesWordEnd: boolean}>}
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
 * Decompose a delimiter token into part IDs. Falls back to per-byte emission
 * for unseen delimiter strings.
 * @param {import("./dictionary.js").PartDictionary} dict
 * @param {string} delim byte-string.
 * @returns {number[]}
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