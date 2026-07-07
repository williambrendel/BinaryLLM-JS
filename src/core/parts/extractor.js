"use strict";

// ============================================================================
// core/parts/extractor.js
//
// Port of core/parts/extractor.cpp — PartExtractor builds a PartDictionary
// from observed words and delimiters via the iterative peel algorithm.
//
//   const ex = new PartExtractor();
//   for (const tok of tokenizeStream(raw)) {
//     if (tok.type === StreamTokenType.Word) ex.addWord(tok.value);
//     else ex.addDelimiter(tok.value);
//   }
//   const dict = ex.finalize();
//
// finalize():
//   1. Seed: whole atoms (length-bound + frequency cascade prune), letter/
//      digit/connector singletons + positional Letter atoms, observed
//      delimiters (freq-sorted) + connector delimiters.
//   2. Peel loop: re-decompose every training word with the current dict,
//      pool the length->=2 singleton runs, and for L in {7..2} promote the
//      cascade-pruned top Start/Mid/End substrings. Stop when the run pool is
//      empty or an iteration adds nothing.
//
// All strings are byte-strings (see byteString.js).
// ============================================================================

import { Kind, kMinPartLength, kMaxPartLength, kInvalidPartId } from "./kind.js";
import { PartDictionary } from "./dictionary.js";
import { findSingletonRuns } from "./decomposer.js";
import { adaptivePruneCount } from "../math/adaptiveThreshold.js";

// In-word connectors — must stay in sync with tokenize.js's rules. Seeded as
// single-char Whole atoms, positional Letter atoms, and Delimiter atoms.
const CONNECTORS = ["-", "'", "&", ",", ".", "$"];

/**
 * @typedef {Object} ExtractorConfig
 * @property {number} [wholeMaxLen=16] Length cap on whole-atom candidates.
 * @property {boolean} [addLetterSingletons=true] Seed the coverage backstop.
 * @property {number} [maxPeelIterations=20] Worst-case fuse on the peel loop.
 */

// Sort a Map<value,freq> into [value,freq] pairs by (freq desc, value asc).
// value asc is byte-wise (JS `<` on byte-strings == C++ std::string operator<).
const sortByFreqDesc = (freqMap) => {
  const v = [...freqMap.entries()];
  v.sort((a, b) => {
    if (a[1] !== b[1]) return b[1] - a[1];
    return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0;
  });
  return v;
};

const prunedCount = (sorted) => adaptivePruneCount(sorted.map((e) => e[1]));

// Add the first k of `sorted` under `kind`, skipping any already present.
const addTopK = (dict, kind, sorted, k) => {
  let added = 0;
  const lim = Math.min(k, sorted.length);
  for (let i = 0; i < lim; i++) {
    if (dict.lookup(kind, sorted[i][0]) === kInvalidPartId) {
      dict.add(kind, sorted[i][0]);
      added++;
    }
  }
  return added;
};

const addLetterPositions = (dict, s) => {
  dict.add(Kind.Letter, s + "##");
  dict.add(Kind.Letter, "##" + s + "##");
  dict.add(Kind.Letter, "##" + s);
};

const bump = (bin, key, weight) => bin.set(key, (bin.get(key) || 0) + weight);

// One peel iteration: pool the current length->=2 singleton runs, then sweep
// L from kMax..kMin promoting cascade-pruned Start/Mid/End winners.
const runPeelIteration = (dict, allWordFreq) => {
  const pool = [];
  for (const [word, freq] of allWordFreq) {
    for (const r of findSingletonRuns(dict, word)) {
      const len = r.end - r.start;
      if (len < 2) continue;
      pool.push({
        substring: word.slice(r.start, r.end),
        touchesStart: r.touchesWordStart,
        touchesEnd: r.touchesWordEnd,
        weight: freq,
      });
    }
  }
  const totalRuns = pool.length;
  if (totalRuns === 0) return { partsAdded: 0, totalRuns: 0 };

  let partsAdded = 0;
  for (let L = kMaxPartLength; L >= kMinPartLength; L--) {
    const startBin = new Map();
    const midBin = new Map();
    const endBin = new Map();

    for (const e of pool) {
      const n = e.substring.length;
      if (L > n) continue;
      for (let i = 0; i + L <= n; i++) bump(midBin, e.substring.slice(i, i + L), e.weight);
      if (e.touchesStart) bump(startBin, e.substring.slice(0, L), e.weight);
      if (e.touchesEnd) bump(endBin, e.substring.slice(n - L, n), e.weight);
    }

    for (const [bin, kind] of [
      [startBin, Kind.Start],
      [midBin, Kind.Mid],
      [endBin, Kind.End],
    ]) {
      const sorted = sortByFreqDesc(bin);
      partsAdded += addTopK(dict, kind, sorted, prunedCount(sorted));
    }
  }
  return { partsAdded, totalRuns };
};

/**
 * Builds a PartDictionary from observed words and delimiters.
 */
export class PartExtractor {
  /** @param {ExtractorConfig} [cfg] */
  constructor(cfg = {}) {
    this._cfg = {
      wholeMaxLen: cfg.wholeMaxLen ?? 16,
      addLetterSingletons: cfg.addLetterSingletons ?? true,
      maxPeelIterations: cfg.maxPeelIterations ?? 20,
    };
    /** @type {Map<string, number>} */
    this._allWordFreq = new Map();
    /** @type {Map<string, number>} */
    this._delimFreq = new Map();
    this._wordCount = 0;
    this._lastPeelIters = 0;
  }

  /**
   * Feed a word (assumed already lowercased ASCII byte-string).
   * @param {string} word
   */
  addWord(word) {
    this._wordCount++;
    if (word.length === 0) return;
    this._allWordFreq.set(word, (this._allWordFreq.get(word) || 0) + 1);
  }

  /**
   * Feed a delimiter token (kept literally).
   * @param {string} delim
   */
  addDelimiter(delim) {
    if (delim.length === 0) return;
    this._delimFreq.set(delim, (this._delimFreq.get(delim) || 0) + 1);
  }

  /** @returns {number} */
  observedWordCount() {
    return this._wordCount;
  }
  /** @returns {number} */
  distinctWordCount() {
    return this._allWordFreq.size;
  }
  /** @returns {number} peel iterations the most recent finalize() ran. */
  lastPeelIterations() {
    return this._lastPeelIters;
  }

  /**
   * Build the final dictionary.
   * @returns {PartDictionary}
   */
  finalize() {
    const dict = new PartDictionary();
    const cfg = this._cfg;
    this._lastPeelIters = 0;

    // --- Seed: whole atoms (length-bound + frequency cascade prune) ---
    {
      const wholes = [];
      for (const [v, f] of this._allWordFreq) {
        if (v.length >= 2 && v.length <= cfg.wholeMaxLen) wholes.push([v, f]);
      }
      wholes.sort((a, b) => {
        if (a[1] !== b[1]) return b[1] - a[1];
        return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0;
      });
      const k = adaptivePruneCount(wholes.map((e) => e[1]));
      const lim = Math.min(k, wholes.length);
      for (let i = 0; i < lim; i++) dict.add(Kind.Whole, wholes[i][0]);
    }

    // --- Seed: single-char Whole atoms + positional Letter atoms ---
    if (cfg.addLetterSingletons) {
      for (let c = 97; c <= 122; c++) dict.add(Kind.Whole, String.fromCharCode(c));
      for (let c = 48; c <= 57; c++) dict.add(Kind.Whole, String.fromCharCode(c));
      for (const c of CONNECTORS) dict.add(Kind.Whole, c);

      for (let c = 97; c <= 122; c++) addLetterPositions(dict, String.fromCharCode(c));
      for (let c = 48; c <= 57; c++) addLetterPositions(dict, String.fromCharCode(c));
      for (const c of CONNECTORS) addLetterPositions(dict, c);
    }

    // --- Seed: observed delimiters (freq order) + connector delimiters ---
    for (const [v] of sortByFreqDesc(this._delimFreq)) dict.add(Kind.Delimiter, v);
    for (const c of CONNECTORS) dict.add(Kind.Delimiter, c);

    // --- Peel loop ---
    for (let iter = 0; iter < cfg.maxPeelIterations; iter++) {
      const { partsAdded, totalRuns } = runPeelIteration(dict, this._allWordFreq);
      this._lastPeelIters++;
      if (totalRuns === 0) break;
      if (partsAdded === 0) break;
    }

    return dict;
  }
}

/**
 * @ignore
 */
export default PartExtractor;