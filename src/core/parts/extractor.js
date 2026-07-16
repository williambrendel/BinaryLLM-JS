"use strict";

/**
 * @file extractor.js
 * @brief Builds a {@link PartDictionary} from observed words and delimiters
 * via the iterative peel algorithm.
 *
 * Port of `core/parts/extractor.cpp`. A {@link PartExtractor} accumulates word
 * and delimiter frequencies as tokens are fed in, then {@link PartExtractor#finalize}
 * produces the trained dictionary in two phases:
 *
 * 1. **Seed** — whole atoms (length-bound + frequency cascade prune), single-char
 *    letter/digit/connector singletons plus length-1 Start/Mid/End atoms, and observed
 *    delimiters (frequency-sorted) plus connector delimiters.
 * 2. **Peel loop** — re-decompose every training word with the current dictionary,
 *    pool the length-≥2 singleton runs, and for each `L` in `{7..2}` promote the
 *    cascade-pruned top Start/Mid/End substrings. The loop stops when the run pool
 *    is empty or an iteration adds nothing new.
 *
 * All strings are byte-strings (see `byteString.js`).
 *
 * **Exported surface:**
 * - `PartExtractor` — the trainer class (also the default export).
 *
 * @see {@link PartDictionary}
 * @see {@link findSingletonRuns}
 */

import { Kind, kMinPartLength, kMaxPartLength, kInvalidPartId } from "./kind.js";
import { PartDictionary, DELIMITERS } from "./dictionary.js";
import { findSingletonRuns } from "./decomposer.js";
import { adaptivePruneCount } from "../math/adaptiveThreshold.js";

// In-word connectors — must stay in sync with tokenize.js's rules. Seeded as
// single-char Whole atoms, length-1 Start/Mid/End atoms, and Delimiter atoms.
const CONNECTORS = ["-", ",", ".", "$"];

/**
 * @typedef {Object} ExtractorConfig
 * @property {number} [wholeMaxLen=16] Length cap on whole-atom candidates.
 * @property {boolean} [addLetterSingletons=true] Seed the coverage backstop.
 * @property {number} [maxPeelIterations=20] Worst-case fuse on the peel loop.
 */

/**
 * @function sortByFreqDesc
 * @description Flattens a frequency map into `[value, freq]` pairs sorted by
 * frequency descending, breaking ties by value ascending. The value comparison
 * is byte-wise (JS `<` on byte-strings matches C++ `std::string::operator<`),
 * so the ordering is deterministic across the port.
 *
 * @param {Map<string, number>} freqMap - Map from value to its observed count.
 * @returns {Array<[string, number]>} The entries sorted by `(freq desc, value asc)`.
 */
const sortByFreqDesc = (freqMap) => {
  const v = [...freqMap.entries()];
  v.sort((a, b) => {
    if (a[1] !== b[1]) return b[1] - a[1];
    return a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0;
  });
  return v;
};

/**
 * @function prunedCount
 * @description Computes how many of the frequency-sorted candidates survive the
 * adaptive cascade prune, by handing their frequencies (already in descending
 * order) to {@link adaptivePruneCount}. The returned count is the cutoff `k`
 * used with {@link addTopK}.
 *
 * @param {Array<[string, number]>} sorted - Candidates sorted by frequency
 *   descending (as produced by {@link sortByFreqDesc}).
 * @returns {number} The number of top candidates to keep.
 */
const prunedCount = (sorted) => adaptivePruneCount(sorted.map((e) => e[1]));

/**
 * @function addTopK
 * @description Adds the first `k` of the frequency-sorted candidates to the
 * dictionary under `kind`, skipping any value already present for that kind, and
 * reports how many were newly inserted. Used to promote the cascade-pruned top
 * substrings during seeding and each peel sweep.
 *
 * @param {PartDictionary} dict - The dictionary being populated.
 * @param {number} kind - The {@link Kind} to insert the values under.
 * @param {Array<[string, number]>} sorted - Candidates sorted by frequency
 *   descending (as produced by {@link sortByFreqDesc}).
 * @param {number} k - Maximum number of leading candidates to consider
 *   (typically {@link prunedCount}); capped at `sorted.length`.
 * @returns {number} The count of parts actually added (new insertions only).
 */
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

/**
 * @function addLetterPositions
 * @description Seeds the three length-1 fragment atoms for a single character:
 * {@link Kind.Start} (`s`), {@link Kind.Mid} (`s`), and {@link Kind.End} (`s`).
 * The value is the bare character — word-boundary position is carried by the
 * kind, mirroring how fragments are matched during decomposition.
 *
 * @param {PartDictionary} dict - The dictionary being populated.
 * @param {string} s - The single-character byte-string to add positional atoms for.
 * @returns {void}
 */
const addLetterPositions = (dict, s) => {
  dict.add(Kind.Start, s);
  dict.add(Kind.Mid, s);
  dict.add(Kind.End, s);
};

/**
 * @function bump
 * @description Adds `weight` to `key`'s running tally in a frequency-accumulator
 * map, initializing the entry to `0` when the key is first seen.
 *
 * @param {Map<string, number>} bin - The accumulator map to update in place.
 * @param {string} key - The key whose tally is incremented.
 * @param {number} weight - The amount to add to the key's tally.
 * @returns {Map<string, number>} The same `bin` map, after the update.
 */
const bump = (bin, key, weight) => bin.set(key, (bin.get(key) || 0) + weight);

/**
 * @function runPeelIteration
 * @description Runs a single iteration of the peel loop against the current
 * dictionary.
 *
 * First it re-decomposes every training word with the dictionary as it stands
 * (via {@link findSingletonRuns}) and pools every length-≥2 singleton run — the
 * spans the dictionary still cannot cover with a real part — recording each run's
 * substring, whether it touches the word's start/end, and its word frequency as a
 * weight.
 *
 * Then, for each substring length `L` from {@link kMaxPartLength} down to
 * {@link kMinPartLength}, it bins every length-`L` window of the pooled runs by
 * frequency: interior windows into the Mid bin, and start-/end-touching prefixes
 * and suffixes into the Start/End bins. Each bin is frequency-sorted and its
 * cascade-pruned top entries ({@link prunedCount}) are promoted into the
 * dictionary as new Start/Mid/End parts via {@link addTopK}.
 *
 * @param {PartDictionary} dict - The dictionary to decompose against and extend
 *   in place with newly promoted parts.
 * @param {Map<string, number>} allWordFreq - Map from training word to its
 *   observed frequency, supplying both the words to decompose and the run weights.
 * @returns {{ partsAdded: number, totalRuns: number }}
 *   `partsAdded` is the number of new parts promoted this iteration;
 *   `totalRuns` is the number of length-≥2 singleton runs pooled. The caller
 *   stops the peel loop when either reaches `0`.
 */
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
 * @class PartExtractor
 * @description Trains a {@link PartDictionary} from a stream of observed words
 * and delimiters. Feed tokens one at a time with {@link PartExtractor#addWord}
 * and {@link PartExtractor#addDelimiter} (typically driven by `tokenizeStream`),
 * then call {@link PartExtractor#finalize} to run the seed + peel algorithm and
 * obtain the trained dictionary.
 *
 * The extractor only accumulates frequency counts as tokens arrive; all of the
 * work happens in `finalize()`. `finalize()` is pure with respect to the
 * accumulated state, so calling it twice on the same extractor yields
 * byte-identical dictionaries.
 *
 * @example
 * // Typical flow: construct, feed tokens from tokenizeStream, finalize.
 * import { PartExtractor } from "./extractor.js";
 * import { tokenizeStream, StreamTokenType } from "./tokenize.js";
 *
 * const ex = new PartExtractor();
 * for (const tok of tokenizeStream(rawBytes)) {
 *   if (tok.type === StreamTokenType.Word) ex.addWord(tok.value);
 *   else ex.addDelimiter(tok.value);
 * }
 * const dict = ex.finalize();   // → trained PartDictionary
 * dict.size();                  // → number of parts in the dictionary
 */
export class PartExtractor {
  /**
   * @description Constructs an empty extractor with no observed tokens.
   * @param {ExtractorConfig} [cfg={}] - Optional configuration overrides.
   *   Unset fields fall back to their defaults (`wholeMaxLen: 16`,
   *   `addLetterSingletons: true`, `maxPeelIterations: 20`).
   *
   * @example
   * const ex = new PartExtractor();                  // all defaults
   * const capped = new PartExtractor({ wholeMaxLen: 12 });
   */
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
   * @method addWord
   * @description Feeds one word into the extractor, incrementing both the total
   * observed-word count and the word's frequency. The word is assumed to be an
   * already-lowercased ASCII byte-string. Empty words still bump the observed
   * count but are not recorded in the frequency map.
   *
   * @param {string} word - The word token (lowercased ASCII byte-string).
   * @returns {void}
   *
   * @example
   * const ex = new PartExtractor();
   * ex.addWord("the");
   * ex.addWord("the");
   * ex.observedWordCount();   // → 2
   * ex.distinctWordCount();   // → 1
   */
  addWord(word) {
    this._wordCount++;
    if (word.length === 0) return;
    this._allWordFreq.set(word, (this._allWordFreq.get(word) || 0) + 1);
  }

  /**
   * @method addDelimiter
   * @description Feeds one delimiter token into the extractor, incrementing its
   * frequency. The delimiter is kept literally (byte-string, not normalized).
   * Empty delimiters are ignored. Delimiter frequencies do not affect the
   * observed- or distinct-word counts.
   *
   * @param {string} delim - The delimiter token (kept literally).
   * @returns {void}
   *
   * @example
   * const ex = new PartExtractor();
   * ex.addDelimiter(" ");
   * ex.addDelimiter(" ");
   * ex.addDelimiter("\n");
   * // Observed delimiters are later seeded into the dictionary by finalize().
   */
  addDelimiter(delim) {
    if (delim.length === 0) return;
    this._delimFreq.set(delim, (this._delimFreq.get(delim) || 0) + 1);
  }

  /**
   * @method observedWordCount
   * @description Total number of words fed via {@link PartExtractor#addWord},
   * counting duplicates and empty words.
   * @returns {number} The running observed-word count.
   *
   * @example
   * const ex = new PartExtractor();
   * ex.addWord("a"); ex.addWord("a"); ex.addWord("b");
   * ex.observedWordCount();   // → 3
   */
  observedWordCount() {
    return this._wordCount;
  }
  /**
   * @method distinctWordCount
   * @description Number of unique non-empty words observed so far.
   * @returns {number} The count of distinct words in the frequency map.
   *
   * @example
   * const ex = new PartExtractor();
   * ex.addWord("a"); ex.addWord("a"); ex.addWord("b");
   * ex.distinctWordCount();   // → 2
   */
  distinctWordCount() {
    return this._allWordFreq.size;
  }
  /**
   * @method lastPeelIterations
   * @description Number of peel-loop iterations executed by the most recent
   * {@link PartExtractor#finalize} call. Returns `0` before `finalize()` has
   * ever run.
   * @returns {number} Peel iterations from the last `finalize()`.
   *
   * @example
   * const ex = new PartExtractor();
   * ex.addWord("running");
   * ex.finalize();
   * ex.lastPeelIterations();   // → number of peel passes that ran
   */
  lastPeelIterations() {
    return this._lastPeelIters;
  }

  /**
   * @method finalize
   * @description Runs the seed + peel algorithm over the accumulated word and
   * delimiter frequencies and returns the trained {@link PartDictionary}.
   *
   * Phases:
   * 1. **Seed** — length-bounded, cascade-pruned Whole atoms; single-char Whole
   *    atoms and length-1 Start/Mid/End atoms for `a`–`z`, `0`–`9`, and the in-word
   *    connectors (when `addLetterSingletons` is enabled); observed Delimiter
   *    atoms in frequency order plus the connector delimiters.
   * 2. **Peel** — up to `maxPeelIterations` passes that re-decompose every
   *    training word with the current dictionary, pool the length-≥2 singleton
   *    runs, and promote the cascade-pruned top Start/Mid/End substrings for
   *    each `L` from `kMaxPartLength` down to `kMinPartLength`. Stops early when
   *    a pass finds no runs or adds no new parts.
   *
   * Does not mutate the extractor's accumulated state (aside from recording the
   * peel-iteration count), so repeated calls produce byte-identical dictionaries.
   *
   * @returns {PartDictionary} The trained dictionary.
   *
   * @example
   * const ex = new PartExtractor();
   * for (const tok of tokenizeStream(rawBytes)) {
   *   if (tok.type === StreamTokenType.Word) ex.addWord(tok.value);
   *   else ex.addDelimiter(tok.value);
   * }
   * const dict = ex.finalize();   // → trained PartDictionary
   * saveDictText(dict);           // → serialized text form of the dictionary
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

    // --- Seed: single-char Whole atoms + length-1 Start/Mid/End atoms ---
    if (cfg.addLetterSingletons) {
      for (let c = 97; c <= 122; c++) dict.add(Kind.Whole, String.fromCharCode(c));
      for (let c = 48; c <= 57; c++) dict.add(Kind.Whole, String.fromCharCode(c));
      for (const c of CONNECTORS) dict.add(Kind.Whole, c);

      for (let c = 97; c <= 122; c++) addLetterPositions(dict, String.fromCharCode(c));
      for (let c = 48; c <= 57; c++) addLetterPositions(dict, String.fromCharCode(c));
      for (const c of CONNECTORS) addLetterPositions(dict, c);
    }

    // --- Seed: fixed manual delimiter set (not learned from the corpus) ---
    for (const d of DELIMITERS) dict.add(Kind.Delimiter, d);

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