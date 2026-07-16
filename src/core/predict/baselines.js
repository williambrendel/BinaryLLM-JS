"use strict";

/**
 * @file baselines.js
 * @brief Context-free (unigram) and single-context (bigram/n-gram) baselines.
 *
 * These are the bars the pool tree must beat on held-out log-loss (spec §5.4,
 * §8.3): unigram is the calibration floor; the bigram/trigram baselines bound how
 * much a *local* context already explains, so a tree's gain over them is the
 * marginal value of the learned structure. Each baseline is a {@link
 * module:core/predict/metrics~RoutedModel} — it routes a record to a count
 * bucket, so {@link module:core/predict/metrics.evaluate} scores it identically
 * to a tree.
 */

import { NO_LABEL } from "../signatures/signatureDataset.js";

/**
 * @function histogram
 * @description Builds a single label-count bucket over labeled records.
 * @param {Object[]} records - Signature records.
 * @param {(r: Object) => number} labelOf - True-label extractor.
 * @returns {import("./readout.js").Leaf} `{ counts, total }`.
 */
const histogram = (records, labelOf) => {
  const counts = new Map();
  let total = 0;
  for (const r of records) {
    const w = labelOf(r);
    if (w === NO_LABEL || w === undefined) continue;
    counts.set(w, (counts.get(w) || 0) + 1);
    total++;
  }
  return { counts, total };
};

/**
 * @function buildUnigram
 * @description The context-free baseline: every record routes to the global label
 * histogram. This is the calibration floor the gate is defined against.
 * @param {Object[]} records - Training records.
 * @param {(r: Object) => number} labelOf - True-label extractor.
 * @returns {import("./metrics.js").RoutedModel} `route` returns the global leaf.
 *
 * @example
 * const uni = buildUnigram(train, (r) => r.nextWord);
 * uni.route(anyRecord) === uni.unigram;   // → true
 */
export const buildUnigram = (records, labelOf) => {
  const leaf = histogram(records, labelOf);
  return { route: () => leaf, unigram: leaf };
};

/**
 * @function buildContextGram
 * @description A single-context baseline (bigram / trigram / …): records are
 * bucketed by a context key `keyOf(r)` (e.g. the previous word), each bucket
 * keeping its own label histogram; an unseen context backs off to the global
 * unigram. Generic over the key so the same code gives the GPT bigram
 * (`keyOf = r => r.curWord`) or any conditioning context.
 * @param {Object[]} records - Training records.
 * @param {(r: Object) => number} labelOf - True-label extractor.
 * @param {(r: Object) => (number|string)} keyOf - Context key extractor.
 * @returns {import("./metrics.js").RoutedModel} `route` returns the context leaf
 *   (or the unigram leaf on an unseen context).
 *
 * @example
 * // GPT bigram: P(next | current word)
 * const bi = buildContextGram(train, (r) => r.nextWord, (r) => r.curWord);
 */
export const buildContextGram = (records, labelOf, keyOf) => {
  const unigram = histogram(records, labelOf);
  const leaves = new Map();
  for (const r of records) {
    const w = labelOf(r);
    if (w === NO_LABEL || w === undefined) continue;
    const k = keyOf(r);
    let leaf = leaves.get(k);
    if (!leaf) leaves.set(k, (leaf = { counts: new Map(), total: 0 }));
    leaf.counts.set(w, (leaf.counts.get(w) || 0) + 1);
    leaf.total++;
  }
  return { route: (r) => leaves.get(keyOf(r)) || unigram, unigram };
};

export default buildUnigram;
