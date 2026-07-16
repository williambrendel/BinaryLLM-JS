"use strict";

/**
 * @file negSet.js
 * @brief §2 — the confusable negative set and its FROZEN statistics.
 *
 * The mask plays TWO roles, now decoupled:
 *   SELECT mask (Mglob, mild): de-saturates the neg-selection content-fraction score. Load-bearing —
 *     common bits make |P⁺∧y|/|content| saturate at 1 so confusables look like positives.
 *   NODE mask (maskNodes, default = SELECT): the affinity node universe P⁺ = ⋁A ∖ D_τ^nodes. In a
 *     PMI-difference affinity a universal common bit self-cancels (PMI⁺≈PMI⁻≈log(1/f)) and has u≈0, so
 *     it decays under −ρx on its own — the NODE mask may be relaxed (even ∅) to keep every moderately-
 *     common bit's pairwise signal, and common bits should simply not get selected. Pass maskNodes=∅ to
 *     test that self-elimination.
 *
 * Two-pass (§ closed): re-runs the SELECT mask locally over {pos ∪ Pass-1 neg}; empirically a no-op at
 * any mild θ_loc and destructive when aggressive — retained as documented evidence, off by default.
 */

export const PAIRK = 100000;
export const pairKey = (a, b) => (a < b ? a * PAIRK + b : b * PAIRK + a);

const freezeStats = (Neg, pset) => {
  const nMinus = new Map(), jMinus = new Map();
  for (const y of Neg) {
    const bits = []; for (const b of y) if (pset.has(b)) bits.push(b); bits.sort((x, z) => x - z);
    for (const b of bits) nMinus.set(b, (nMinus.get(b) || 0) + 1);
    for (let i = 0; i < bits.length; i++) for (let j = i + 1; j < bits.length; j++) { const k = pairKey(bits[i], bits[j]); jMinus.set(k, (jMinus.get(k) || 0) + 1); }
  }
  return { nMinus, jMinus };
};

/**
 * @function buildNegSet
 * @param {Array<number[]>} A @param {Array<number[]>} negPool @param {Iterable<number>} Mglob - SELECT mask.
 * @param {object} [opts] @param {number} [opts.minSize]
 * @param {Iterable<number>} [opts.maskNodes] - NODE mask (default = Mglob). Pass ∅ to keep common bits as nodes.
 * @param {boolean} [opts.twoPass=false] @param {number} [opts.candMult=8] @param {number} [opts.localTheta=0.5]
 * @param {number} [opts.delta=0.05] - δ band (always positive; skips saturated near-duplicates).
 * @returns {{pPlus:number[], pset:Set<number>, Dtau:Set<number>, DtauSel:Set<number>, DtauLoc:number,
 *   Neg:Array<number[]>, negN:number, candN:number, skipped:number, scoreCutPre:number,
 *   scoreCutPost:number, nMinus:Map, jMinus:Map}}
 */
export const buildNegSet = (A, negPool, Mglob, opts = {}) => {
  const { minSize = A.length, twoPass = false, candMult = 8, localTheta = 0.5, delta = 0.05, maskNodes } = opts;
  const Msel = Mglob instanceof Set ? Mglob : new Set(Mglob || []);
  const Mnod = maskNodes === undefined ? Msel : (maskNodes instanceof Set ? maskNodes : new Set(maskNodes || []));

  const raw = new Set(); for (const y of A) for (const b of y) raw.add(b);

  // ── SELECTION mask (mild): drives neg scoring only ──
  const DtauSel = new Set(); for (const b of raw) if (Msel.has(b)) DtauSel.add(b);
  let psetSel = new Set(); for (const b of raw) if (!DtauSel.has(b)) psetSel.add(b);

  let pool = negPool, DtauLoc = 0;
  if (twoPass) {                                                                  // (closed) local re-mask on SELECT scale
    const movG = (y) => { let c = 0; for (const b of y) if (psetSel.has(b)) c++; return c; };
    const cand = negPool.map((y) => [y, movG(y)]).filter(([, c]) => c > 0)
      .sort((p, q) => q[1] - p[1] || q[0].length - p[0].length).slice(0, candMult * minSize).map(([y]) => y);
    const local = A.concat(cand), nLoc = Math.max(1, local.length);
    const lf = new Map(); for (const y of local) for (const b of y) if (psetSel.has(b)) lf.set(b, (lf.get(b) || 0) + 1);
    for (const [b, c] of lf) if (c / nLoc > localTheta) { DtauSel.add(b); DtauLoc++; }
    psetSel = new Set(); for (const b of raw) if (!DtauSel.has(b)) psetSel.add(b);
    pool = cand;
  }

  // selection: content-fraction + δ band (SELECT mask). Two paths, IDENTICAL output:
  //   (a) scan (default): score each pool context directly, O(Σ|y|).
  //   (b) inverted index (opts.negIndex, only when !twoPass): maskedOv = rawOv − dtauOv via posting lists —
  //       integer overlaps ⇒ identical scores ⇒ identical stable sort. `negLen` = precomputed |y| per pool ctx.
  let Neg, start, scoreCutPre, scoreCutPost;
  const cut = 1 - delta;
  if (opts.negIndex && !twoPass) {
    const idx = opts.negIndex, negLen = opts.negLen, nN = pool.length;
    const rawOv = new Int32Array(nN), dtauOv = new Int32Array(nN);
    for (const b of raw) { const p = idx.get(b); if (p) for (let k = 0; k < p.length; k++) rawOv[p[k]]++; }      // |y∩⋁A|
    for (const b of DtauSel) { const p = idx.get(b); if (p) for (let k = 0; k < p.length; k++) dtauOv[p[k]]++; }  // |y∩D_τ|
    const scr = new Float64Array(nN);
    for (let i = 0; i < nN; i++) { const cl = negLen[i] - dtauOv[i]; scr[i] = cl > 0 ? (rawOv[i] - dtauOv[i]) / cl : 0; }  // = maskedOv/contentLen
    const order = Array.from({ length: nN }, (_, i) => i).sort((a, b) => scr[b] - scr[a] || negLen[b] - negLen[a]);  // stable
    start = 0; while (start < nN && scr[order[start]] >= cut) start++;
    if (start + minSize > nN) start = Math.max(0, nN - minSize);
    const win = order.slice(start, start + minSize);
    Neg = win.map((i) => pool[i]);
    scoreCutPre = nN ? scr[order[0]] : 0; scoreCutPost = win.length ? scr[win[win.length - 1]] : 0;
  } else {
    const contentLen = (y) => { let c = 0; for (const b of y) if (!DtauSel.has(b)) c++; return c; };
    const maskedOv = (y) => { let c = 0; for (const b of y) if (psetSel.has(b)) c++; return c; };
    const score = (y) => { const cl = contentLen(y); return cl > 0 ? maskedOv(y) / cl : 0; };
    const scored = pool.map((y) => [y, score(y)]).sort((p, q) => q[1] - p[1] || q[0].length - p[0].length);
    start = 0; while (start < scored.length && scored[start][1] >= cut) start++;
    if (start + minSize > scored.length) start = Math.max(0, scored.length - minSize);
    const window = scored.slice(start, start + minSize); Neg = window.map(([y]) => y);
    scoreCutPre = scored.length ? scored[0][1] : 0; scoreCutPost = window.length ? window[window.length - 1][1] : 0;
  }

  // ── NODE universe (may be relaxed vs SELECT): affinity nodes + frozen stats ──
  const Dtau = new Set(); for (const b of raw) if (Mnod.has(b)) Dtau.add(b);
  const pset = new Set(); for (const b of raw) if (!Dtau.has(b)) pset.add(b);
  const pPlus = [...pset].sort((x, z) => x - z);
  const { nMinus, jMinus } = freezeStats(Neg, pset);
  return { pPlus, pset, Dtau, DtauSel, DtauLoc, Neg, negN: Neg.length, candN: pool.length, skipped: start,
    scoreCutPre, scoreCutPost, nMinus, jMinus };
};

export default buildNegSet;
