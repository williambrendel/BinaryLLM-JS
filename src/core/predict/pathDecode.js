"use strict";

/**
 * @file pathDecode.js
 * @brief The path decoder — the readout over a test signature's root→leaf path.
 *
 * Adapts the spec's decoder (§4.2). Along the path the sample takes, each visited
 * child node contributes its smoothed distribution `P_k` as a support-weighted
 * correction relative to its PARENT `P_{k-1}` (`P_0` = unigram). This is the sound
 * telescoping the spec describes — nested splits partition what the previous left,
 * so corrections chain instead of each re-referencing the global prior (which
 * would make the prior coefficient `1−Σw_k` go negative on deep paths and invert):
 *
 *   w_k       = m_k / (m_k + κ)                              (support weight)
 *   score(w)  = log P_0(w) + Σ_k w_k·[log P_k(w) − log P_{k-1}(w)]
 *   P̂(w)      = softmax_w score(w)
 *   P_out(w)  = (1−β)·P̂(w) + β·P_0(w)                        (unigram floor)
 *
 * If every `w_k=1` the sum telescopes exactly to `log P_d(w)` (the leaf); support
 * weights shrink each step toward its parent. The prior coefficient is `1−w_1 > 0`,
 * so common words are never inverted. All smoothing is `(N(w)+α)/(m+αV)`. Where the
 * leaf readout sees only the final histogram, this sees the whole path + backoff —
 * the §5.5 discriminator between "leaf-weak" and "signal-poor".
 */

import { NO_LABEL } from "../signatures/signatureDataset.js";

/**
 * @function makePathContext
 * @description Precomputes the per-vocabulary unigram prior (`P` and `log P`) and a
 * reusable scratch buffer, so decoding a path is two O(V) sweeps.
 * @param {import("./readout.js").Leaf} unigram - Global label histogram.
 * @param {number} V - Vocabulary size.
 * @param {Object} [opts]
 * @param {number} [opts.alpha=0.1] - Smoothing pseudo-count.
 * @param {number} [opts.kappa=50] - Support-weight pseudo-count.
 * @param {number} [opts.beta=0.1] - Unigram-floor weight.
 * @returns {Object} Reusable decode context.
 */
export const makePathContext = (unigram, V, { alpha = 0.1, kappa = 50, beta = 0.1 } = {}) => {
  const uniLogP = new Float64Array(V), uniP = new Float64Array(V);
  const denom = unigram.total + alpha * V;
  for (let w = 0; w < V; w++) { const p = ((unigram.counts.get(w) || 0) + alpha) / denom; uniP[w] = p; uniLogP[w] = Math.log(p); }
  return { V, alpha, kappa, beta, uniLogP, uniP, score: new Float64Array(V) };
};

/**
 * @function decodePath
 * @description Decodes one path to `P_out`, returning the true word's log-prob and
 * the argmax. `score(w) = γ·logP(w) + C + delta(w)` where `γ = 1 − Σw_k`, `C`
 * collects the "absent" mass of each node, and `delta(w)` the per-word corrections
 * for words present on the path — so only path words touch the scratch buffer.
 * @param {import("./readout.js").Leaf[]} path - Child histograms root→leaf.
 * @param {number} trueW - True label id.
 * @param {Object} ctx - From {@link makePathContext}.
 * @returns {{logProb:number, argmax:number}}
 */
export const decodePath = (path, trueW, ctx) => {
  const { V, alpha, kappa, beta, uniLogP, uniP, score } = ctx;
  const d = path.length;
  const wk = new Float64Array(d), base = new Float64Array(d), dnm = new Float64Array(d);
  for (let k = 0; k < d; k++) { const m = path[k].total; wk[k] = m / (m + kappa); dnm[k] = m + alpha * V; base[k] = Math.log(alpha / dnm[k]); }

  // score(w) = logP_0(w) + Σ_k w_k[logP_k(w) − logP_{k-1}(w)], telescoping through parents.
  let Z = 0;
  for (let w = 0; w < V; w++) {
    let s = uniLogP[w], prev = uniLogP[w];
    for (let k = 0; k < d; k++) {
      const cnt = path[k].counts.get(w);
      const lpk = cnt !== undefined ? Math.log((cnt + alpha) / dnm[k]) : base[k];
      s += wk[k] * (lpk - prev);
      prev = lpk;
    }
    score[w] = s; Z += Math.exp(s);
  }
  let bestW = 0, bestP = -1;
  for (let w = 0; w < V; w++) { const po = (1 - beta) * Math.exp(score[w]) / Z + beta * uniP[w]; if (po > bestP) { bestP = po; bestW = w; } }
  const poTrue = (1 - beta) * Math.exp(score[trueW]) / Z + beta * uniP[trueW];
  return { logProb: Math.log(poTrue), argmax: bestW };
};

/**
 * @function evaluatePath
 * @description Held-out log-loss + accuracy of a tree's path decoder.
 * @param {{routePath:(r:Object)=>import("./readout.js").Leaf[], unigram:import("./readout.js").Leaf}} tree
 * @param {Object[]} records - Held-out records.
 * @param {Object} cfg
 * @param {number} cfg.V - Vocabulary size.
 * @param {(r: Object) => number} cfg.labelOf - True-label extractor.
 * @param {number} [cfg.alpha=0.1]
 * @param {number} [cfg.kappa=50]
 * @param {number} [cfg.beta=0.1]
 * @returns {{ll:number, acc:number, n:number}}
 */
export const evaluatePath = (tree, records, { V, labelOf, alpha = 0.1, kappa = 50, beta = 0.1 }) => {
  const ctx = makePathContext(tree.unigram, V, { alpha, kappa, beta });
  let ll = 0, correct = 0, n = 0;
  for (const r of records) {
    const w = labelOf(r);
    if (w === NO_LABEL || w === undefined) continue;
    const { logProb, argmax } = decodePath(tree.routePath(r), w, ctx);
    ll -= logProb; if (argmax === w) correct++; n++;
  }
  return n ? { ll: ll / n, acc: correct / n, n } : { ll: 0, acc: 0, n: 0 };
};

export default evaluatePath;
