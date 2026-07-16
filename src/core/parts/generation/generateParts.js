"use strict";

/**
 * @file generateParts.js
 * @brief Orchestrates the four-track part generator into one dictionary + tags.
 *
 * Tracks (all populate one deduped {@link PartDictionary}):
 *   1. {@link prefilterWholes}     — whole-word atoms.
 *   2. {@link generativeAffixes}   — frequency-head Start/Mid/End (obvious morphemes).
 *   3. {@link discriminativeTree}  — balanced-split word-separating parts.
 *   4. atoms — {@link augmentWithAtoms} seeds letters/connectors/delimiters.
 *
 * The length-2 **bigram backstop** is *not* baked in here — it is a fixed,
 * corpus-independent layer added at encode time by
 * {@link module:core/parts/encoding/bigramBackstop}.
 *
 * Emits a `tracks` map (`"kind|value" → "w"|"d"|"g"`) so the encoder can prefer
 * discriminative parts on the boundaries.
 */

import { Kind } from "../kind.js";
import { PartDictionary, augmentWithAtoms } from "../dictionary.js";
import { prefilterWholes } from "./prefilterWholes.js";
import { generativeAffixes } from "./generativeAffixes.js";
import { discriminativeTree } from "./discriminativeTree.js";

const TRACKED = [Kind.Whole, Kind.Start, Kind.Mid, Kind.End];

/**
 * @typedef {Object} GenerateResult
 * @property {PartDictionary} dict - The combined dictionary (learned + atoms).
 * @property {Object<string, string>} tracks - `"kind|value" → "w"|"d"|"g"` tags.
 * @property {{words:number, wholes:number, generative:number, discriminative:number, sharedGenDisc:number}} stats
 */

/**
 * @function generateParts
 * @description Builds the four-track dictionary from a word-frequency map.
 *
 * @param {Map<string, number>} freqMap - Word → token count (see
 *   {@link module:utilities/corpus/wordFreq}).
 * @param {Object} [opts] - Forwarded to the track functions (`lMax`, `lMin`,
 *   `shortWholeMax`, `elbowSig`, `minGain`, `maxTree`).
 * @returns {GenerateResult}
 *
 * @example
 * const { dict, tracks, stats } = generateParts(wordFreq([buf]));
 * saveDictText(dict);            // serialize the learned parts
 * bpeEncode(dict, "running", tracks);
 */
export const generateParts = (freqMap, opts = {}) => {
  const words = [...freqMap.keys()];
  const freq = Float64Array.from(words.map((w) => freqMap.get(w)));
  const dict = new PartDictionary();
  const removed = new Uint8Array(words.length);

  const wholes = prefilterWholes(words, freq, dict, removed, opts);
  const alive = [];
  for (let wi = 0; wi < words.length; wi++) if (!removed[wi]) alive.push(wi);

  const genSet = new Set(), treeSet = new Set();
  const generative = generativeAffixes(words, freq, alive, dict, genSet, opts);
  const discriminative = discriminativeTree(words, freq, alive, dict, treeSet, opts);

  augmentWithAtoms(dict);

  const tracks = {};
  let sharedGenDisc = 0;
  for (const p of dict.allParts()) {
    if (!TRACKED.includes(p.kind) || p.value.length < 2) continue;
    const key = p.kind + "|" + p.value;
    tracks[key] = p.kind === Kind.Whole ? "w" : treeSet.has(key) ? "d" : genSet.has(key) ? "g" : "?";
  }
  for (const k of genSet) if (treeSet.has(k)) sharedGenDisc++;

  dict.tracks = tracks; // carry provenance on the dict so IO / the encoder can default to it
  return { dict, tracks, stats: { words: words.length, wholes, generative, discriminative, sharedGenDisc } };
};

export default generateParts;
