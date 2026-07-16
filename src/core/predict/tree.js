"use strict";

/**
 * @file tree.js
 * @brief Capped, best-first CART tree whose splits are grown parts (coverage-fired).
 *
 * A single per-node CART tree: at each node {@link growPart} grows one part `(p, t)`
 * and the node splits into samples that fire it (`|x∩p|/|p| ≥ t`) vs the rest.
 * Growth is best-first by size-weighted gain under a `K`-leaf cap; `s_min`
 * regularizes. Every node carries the label histogram of the samples reaching it,
 * so the tree supports two readouts:
 *   - `route(r)` → the LEAF histogram (leaf readout).
 *   - `routePath(r)` → the ordered CHILD histograms root→leaf (path decoder, §4.2).
 * The tree is a {@link module:core/predict/metrics~RoutedModel} for the leaf readout.
 */

import { NO_LABEL } from "../signatures/signatureDataset.js";
import { growPart } from "./growPart.js";

/**
 * @function buildCartTree
 * @description Grows a depth-limited CART tree of coverage-fired grown-part splits
 * (the spec's `build`). Each node grows one part via {@link growPart} and recurses
 * on Left `{popcount(x∧p) ≥ k*}` / Right. Records with a {@link NO_LABEL} label are
 * dropped from training. Every leaf records WHY it stopped (`terminalReasons`).
 *
 * @param {Object[]} records - Training records.
 * @param {Object} opts
 * @param {(r: Object) => number[]} opts.featuresOf - Sorted feature-bit array.
 * @param {(r: Object) => number} opts.labelOf - True-label extractor.
 * @param {number} opts.sMin - Minimum samples per child (regularizer).
 * @param {number} [opts.D=4] - Max part width per split (bit accretion cap).
 * @param {number} [opts.Dmax=12] - Tree depth cap.
 * @param {number} [opts.alpha=1] - Class-weight exponent `w_l ∝ n_l^{−α}`.
 * @param {number} [opts.lambda=1e-9] - MDL floor: stop bit accretion at marginal gain ≤ λ.
 * @param {number} [opts.M=48] - Candidate bits per accretion step.
 * @returns {{route:Function, routePath:Function, unigram:Object, leaves:number, root:Object, terminalReasons:Object}}
 */
export const buildCartTree = (records, { featuresOf, labelOf, sMin, D = 4, Dmax = 12, alpha = 0, lambda = 1e-9, M = 48, gainType = "ig" }) => {
  const featArr = [], featSet = [], labels = [];
  for (const r of records) {
    const lab = labelOf(r);
    if (lab === NO_LABEL || lab === undefined) continue;
    const bits = featuresOf(r);
    featArr.push(bits); featSet.push(new Set(bits)); labels.push(lab);
  }
  const N = labels.length;
  const growOpts = { D, sMin, alpha, lambda, M, gainType };
  const histOf = (idx) => {
    const counts = new Map();
    for (const j of idx) counts.set(labels[j], (counts.get(labels[j]) || 0) + 1);
    return { counts, total: idx.length };
  };
  const unigram = histOf([...Array(N).keys()]);

  // Recursive depth-limited growth. Leaves tag their terminal reason for §5 analysis.
  const reasons = { pure: 0, smin: 0, depth: 0, nosplit: 0 };
  let leaves = 0;
  const build = (idx, depth) => {
    const hist = histOf(idx);
    if (hist.counts.size <= 1) { reasons.pure++; leaves++; return { hist }; }
    if (idx.length < 2 * sMin) { reasons.smin++; leaves++; return { hist }; }
    if (depth >= Dmax) { reasons.depth++; leaves++; return { hist }; }
    const s = growPart(idx, featArr, featSet, labels, growOpts);
    if (!s) { reasons.nosplit++; leaves++; return { hist }; }
    const node = { hist, part: s.bits, k: s.k, threshold: s.threshold, gain: s.gain, fireSize: s.fire.length, notSize: s.notFire.length };
    node.present = build(s.fire, depth + 1);
    node.absent = build(s.notFire, depth + 1);
    return node;
  };
  const root = build([...Array(N).keys()], 0);

  // fire iff coverage(x, part) = popcount(x ∩ part) ≥ k.
  const fires = (bits, node) => {
    let inter = 0;
    for (const b of node.part) if (bits.has(b)) inter++;
    return inter >= node.k;
  };
  const route = (r) => {
    const bits = new Set(featuresOf(r));
    let node = root;
    while (node.part !== undefined) node = fires(bits, node) ? node.present : node.absent;
    return node.hist;
  };
  // The child histograms visited root→leaf (each = P(w | node, branch taken)).
  const routePath = (r) => {
    const bits = new Set(featuresOf(r)), path = [];
    let node = root;
    while (node.part !== undefined) { node = fires(bits, node) ? node.present : node.absent; path.push(node.hist); }
    return path;
  };

  return { route, routePath, unigram, leaves, root, terminalReasons: reasons };
};

export default buildCartTree;
