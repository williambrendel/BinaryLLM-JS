"use strict";

/**
 * @file readout.js
 * @brief Smoothed leaf / unigram distributions and the interpolation floor.
 *
 * Port of the spec's readout contract (pool-discovery.tex §4.1, §4.3, §5.3): a
 * model stores label **counts**, never probabilities, and derives probabilities
 * at decode with Laplace/Dirichlet smoothing so every log is finite:
 *
 *   P(w | leaf) = (N(w | leaf) + α) / (m_leaf + α·V)
 *
 * with `α` small (default 0.1) and `V` the **global tokenizer vocabulary size**
 * (fixed across builds and inference). A "leaf" is any routed bucket of counts —
 * a tree leaf, a bigram context, or the global unigram histogram — so the same
 * readout scores every model uniformly.
 */

/**
 * @typedef {Object} Leaf
 * @property {Map<number, number>} counts - label id → training count `N(w|leaf)`.
 * @property {number} total - `m_leaf` = Σ counts.
 */

/**
 * @function leafProb
 * @description Smoothed `P(w | leaf) = (N(w)+α)/(m+αV)`. Always `> 0`.
 * @param {Leaf} leaf - The routed count bucket.
 * @param {number} w - Label id.
 * @param {number} V - Global vocabulary size.
 * @param {number} alpha - Smoothing pseudo-count.
 * @returns {number} Probability in (0, 1).
 */
export const leafProb = (leaf, w, V, alpha) => ((leaf.counts.get(w) || 0) + alpha) / (leaf.total + alpha * V);

/**
 * @function leafLogProb
 * @description `log P(w | leaf)` (natural log — spec reports nats).
 * @param {Leaf} leaf - The routed count bucket.
 * @param {number} w - Label id.
 * @param {number} V - Global vocabulary size.
 * @param {number} alpha - Smoothing pseudo-count.
 * @returns {number} A finite (negative) log-probability.
 */
export const leafLogProb = (leaf, w, V, alpha) => Math.log(leafProb(leaf, w, V, alpha));

/**
 * @function flooredLogProb
 * @description Interpolation-floored log prob (spec §4.3): the convex mix
 * `(1-β)·P(w|leaf) + β·P(w|unigram)`, guaranteeing at least `β·P_unigram(w)` mass
 * on every word — a thin-leaf blow-up guard that stays normalized.
 * @param {Leaf} leaf - The routed count bucket.
 * @param {Leaf} uni - The global unigram histogram.
 * @param {number} w - Label id.
 * @param {number} V - Global vocabulary size.
 * @param {number} alpha - Smoothing pseudo-count.
 * @param {number} beta - Floor weight (default caller-supplied, spec 0.1).
 * @returns {number} A finite floored log-probability.
 */
export const flooredLogProb = (leaf, uni, w, V, alpha, beta) =>
  Math.log((1 - beta) * leafProb(leaf, w, V, alpha) + beta * leafProb(uni, w, V, alpha));

/**
 * @function leafArgmax
 * @description The majority label of a leaf (argmax over counts) — the top-1
 * prediction. Smoothing is monotone, so this equals argmax over `P(w|leaf)`;
 * unseen labels never win a non-empty leaf. Returns `undefined` for an empty leaf.
 * @param {Leaf} leaf - The routed count bucket.
 * @returns {number|undefined} The most frequent label id.
 */
export const leafArgmax = (leaf) => {
  let best, bestC = -1;
  for (const [w, c] of leaf.counts) if (c > bestC) { bestC = c; best = w; }
  return best;
};

export default leafLogProb;
