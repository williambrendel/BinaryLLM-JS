"use strict";

/**
 * @file metrics.js
 * @brief Held-out log-loss / accuracy and the calibration gate.
 *
 * The evaluation contract from the spec (pool-discovery.tex §4.6, §5.4, §10.1):
 * **log-loss is the decision metric**; accuracy is diagnostic only (it is blind
 * to the "argmax sits on the majority word while the distribution is
 * miscalibrated" failure that produced the earlier false "architecture is broken"
 * conclusion). Any model that routes a record to a {@link module:core/predict/readout~Leaf}
 * is scored the same way, so baselines and trees are directly comparable.
 */

import { leafLogProb, flooredLogProb, leafArgmax } from "./readout.js";
import { NO_LABEL } from "../signatures/signatureDataset.js";

/**
 * @typedef {Object} RoutedModel
 * @property {(record: Object) => import("./readout.js").Leaf} route - Maps a
 *   record to its count bucket (leaf / context / global histogram).
 * @property {import("./readout.js").Leaf} unigram - The global unigram histogram
 *   (used for the interpolation floor).
 */

/**
 * @function evaluate
 * @description Mean held-out log-loss (raw and floored) and top-1 accuracy of a
 * routed model. Records whose label is {@link NO_LABEL} are skipped.
 *
 * @param {RoutedModel} model - Anything exposing `route` + `unigram`.
 * @param {Object[]} records - Held-out signature records.
 * @param {Object} cfg
 * @param {number} cfg.V - Global vocabulary size.
 * @param {(record: Object) => number} cfg.labelOf - Extracts the true label
 *   (`curWord` for the BERT/cloze head, `nextWord` for the GPT/next-word head).
 * @param {number} [cfg.alpha=0.1] - Laplace smoothing pseudo-count.
 * @param {number} [cfg.beta=0.1] - Interpolation-floor weight.
 * @returns {{ll:number, llFloored:number, acc:number, n:number}} Means over the
 *   `n` labeled records — `ll` (nats) is the decision metric.
 *
 * @example
 * const uni = buildUnigram(train, (r) => r.nextWord);
 * evaluate(uni, held, { V, labelOf: (r) => r.nextWord });  // → { ll, llFloored, acc, n }
 */
export const evaluate = (model, records, { V, labelOf, alpha = 0.1, beta = 0.1 }) => {
  let ll = 0, llf = 0, correct = 0, n = 0;
  for (const r of records) {
    const w = labelOf(r);
    if (w === NO_LABEL || w === undefined) continue;
    const leaf = model.route(r);
    ll -= leafLogProb(leaf, w, V, alpha);
    llf -= flooredLogProb(leaf, model.unigram, w, V, alpha, beta);
    if (leafArgmax(leaf) === w) correct++;
    n++;
  }
  return n ? { ll: ll / n, llFloored: llf / n, acc: correct / n, n } : { ll: 0, llFloored: 0, acc: 0, n: 0 };
};

/**
 * @function calibrationGate
 * @description The hard gate (spec §4.6): a model is valid only if its held-out
 * log-loss does not exceed the unigram log-loss. A model above unigram is broken
 * (it assigns the truth less mass than the context-free prior on average), and no
 * accuracy-based conclusion is valid until it passes. A small tolerance absorbs
 * float noise on the exact-equality (unigram-vs-unigram) case.
 *
 * @param {number} modelLL - The model's mean held-out log-loss.
 * @param {number} unigramLL - The unigram mean held-out log-loss.
 * @param {number} [eps=1e-9] - Numerical tolerance.
 * @returns {boolean} `true` if the model passes (`modelLL ≤ unigramLL`).
 *
 * @example
 * calibrationGate(6.02, 6.10);   // → true   (beats the prior)
 * calibrationGate(6.20, 6.10);   // → false  (broken — fix before trusting accuracy)
 */
export const calibrationGate = (modelLL, unigramLL, eps = 1e-9) => modelLL <= unigramLL + eps;

export default evaluate;
