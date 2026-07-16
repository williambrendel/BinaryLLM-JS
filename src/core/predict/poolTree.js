"use strict";

/**
 * @file poolTree.js
 * @brief The second-pass readout: one tree over a FIXED pool of parts.
 *
 * Pool discovery's second pass (spec): a forest discovers a pool of parts; here a
 * single tree uses that pool as its fixed set of splitters — no new parts are
 * grown. Each part `p` is a binary feature `fires(x,p) = popcount(x∩p) ≥ k_p` (its
 * discovered threshold), and at each node we pick the pool part whose fire/not
 * split maximizes the per-class binary-entropy gain (same objective as growth):
 *
 *   w_l ∝ n_l^{−α},  f_l = a_l/n_l,  Gain = H_b(Σ w_l f_l) − Σ w_l H_b(f_l)
 *
 * Firing per (sample, part) is fixed, so it is precomputed once; each node then
 * costs only O(Σ |fired parts|) to score every candidate. Routing a held-out
 * record recomputes coverage on the fly, so the tree is a routed model scored
 * uniformly by {@link module:core/predict/metrics} and the path decoder.
 */

import { NO_LABEL } from "../signatures/signatureDataset.js";

const Hb = (p) => (p <= 0 || p >= 1 ? 0 : -p * Math.log(p) - (1 - p) * Math.log(1 - p));

/**
 * @function buildPoolTree
 * @description Grows a depth-limited tree whose splitters are a fixed pool of parts.
 * @param {Object[]} records - Training records.
 * @param {Object} opts
 * @param {(r:Object)=>number[]} opts.featuresOf - Sorted feature-bit array.
 * @param {(r:Object)=>number} opts.labelOf - True-label extractor.
 * @param {{bits:number[], k:number}[]} opts.pool - Fixed candidate parts.
 * @param {number} opts.sMin - Minimum samples per child.
 * @param {number} [opts.Dmax=12] - Depth cap.
 * @param {number} [opts.alpha=1] - Class-weight exponent.
 * @returns {{route:Function, routePath:Function, unigram:Object, leaves:number, root:Object, terminalReasons:Object}}
 */
export const buildPoolTree = (records, { featuresOf, labelOf, pool, sMin, Dmax = 12, alpha = 0 }) => {
  const featSet = [], labels = [];
  for (const r of records) {
    const lab = labelOf(r);
    if (lab === NO_LABEL || lab === undefined) continue;
    featSet.push(new Set(featuresOf(r))); labels.push(lab);
  }
  const N = labels.length;
  const parts = pool.map((p) => ({ bits: p.bits, set: p.bits instanceof Set ? p.bits : new Set(p.bits), k: p.k }));

  // Precompute, per sample, which pool parts it fires (sparse).
  const sampleFires = new Array(N);
  for (let i = 0; i < N; i++) {
    const bits = featSet[i], fired = [];
    for (let pi = 0; pi < parts.length; pi++) {
      const { set, k } = parts[pi]; let inter = 0;
      for (const b of set) if (bits.has(b) && ++inter >= k) break;
      if (inter >= k) fired.push(pi);
    }
    sampleFires[i] = Int32Array.from(fired);
  }

  const histOf = (idx) => { const counts = new Map(); for (const i of idx) counts.set(labels[i], (counts.get(labels[i]) || 0) + 1); return { counts, total: idx.length }; };
  const unigram = histOf([...Array(N).keys()]);

  const reasons = { pure: 0, smin: 0, depth: 0, nosplit: 0 };
  let leaves = 0;
  const build = (idx, depth) => {
    const hist = histOf(idx);
    if (hist.counts.size <= 1) { reasons.pure++; leaves++; return { hist }; }
    if (idx.length < 2 * sMin) { reasons.smin++; leaves++; return { hist }; }
    if (depth >= Dmax) { reasons.depth++; leaves++; return { hist }; }

    // Per-class weighted node mass/entropy for INFORMATION GAIN (ω_c = n_c^{−α}) —
    // same criterion as the grower, so the pool tree isn't biased toward bisects.
    const nCount = hist.counts;
    const omega = new Map(); let Wnode = 0, Snode = 0;
    for (const [c, n] of nCount) { const w = alpha === 0 ? 1 : n ** -alpha; omega.set(c, w); const nw = n * w; Wnode += nw; Snode += nw * Math.log(nw); }
    const Hnode = Math.log(Wnode) - Snode / Wnode;

    // Left (fire) class-counts per candidate part, over this node only.
    const leftHist = new Map(); const leftTot = new Map();
    for (const i of idx) { const c = labels[i]; for (const pi of sampleFires[i]) { let m = leftHist.get(pi); if (!m) leftHist.set(pi, (m = new Map())); m.set(c, (m.get(c) || 0) + 1); leftTot.set(pi, (leftTot.get(pi) || 0) + 1); } }

    let bestP = -1, bestG = 0;
    for (const [pi, lt] of leftTot) {
      if (lt < sMin || idx.length - lt < sMin) continue;
      let WL = 0, termFire = 0, termNotAdj = 0;
      for (const [c, a] of leftHist.get(pi)) { const w = omega.get(c); const aw = a * w, nw = nCount.get(c) * w; WL += aw; termFire += aw * Math.log(aw); const rem = nw - aw; termNotAdj += nw * Math.log(nw) - (rem > 0 ? rem * Math.log(rem) : 0); }
      const WR = Wnode - WL; if (WL <= 0 || WR <= 0) continue;
      const ig = Hnode - (WL * Math.log(WL) + WR * Math.log(WR) - termFire - (Snode - termNotAdj)) / Wnode;
      if (ig > bestG) { bestG = ig; bestP = pi; }
    }
    if (bestP < 0) { reasons.nosplit++; leaves++; return { hist }; }

    const fire = [], notFire = [];
    for (const i of idx) { (sampleFires[i].includes(bestP) ? fire : notFire).push(i); }
    const node = { hist, part: parts[bestP].bits, k: parts[bestP].k, gain: bestG, fireSize: fire.length, notSize: notFire.length };
    node.present = build(fire, depth + 1);
    node.absent = build(notFire, depth + 1);
    return node;
  };
  const root = build([...Array(N).keys()], 0);

  const fires = (bits, node) => { let inter = 0; for (const b of node.part) if (bits.has(b)) inter++; return inter >= node.k; };
  const route = (r) => { const bits = new Set(featuresOf(r)); let node = root; while (node.part !== undefined) node = fires(bits, node) ? node.present : node.absent; return node.hist; };
  const routePath = (r) => { const bits = new Set(featuresOf(r)), path = []; let node = root; while (node.part !== undefined) { node = fires(bits, node) ? node.present : node.absent; path.push(node.hist); } return path; };

  return { route, routePath, unigram, leaves, root, terminalReasons: reasons };
};

export default buildPoolTree;
