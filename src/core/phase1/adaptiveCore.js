"use strict";

/**
 * @file adaptiveCore.js
 * @brief §8 [CANDIDATE] — the FP-aware, concentration-driven core extractor: an alternative to the AdaBoost
 * `boost` loop. Round 0 derives a dynamic FP ceiling from the word's own concentration and decides whether to
 * ship a single strong core or spawn disjoint cores:
 *
 *   r50      = size of the top-x* prefix that first reaches 50% recall (concentration proxy)
 *   dynFPMAX = min(FPCAP, FPSLOPE·r50)                    predicted ensemble-FP plateau
 *   STRONG   : if dominant recall ≥ STRONG AND its FP ≤ dynFPMAX → ship ONE core, done
 *   else     : grow a weight-first core to VIABLE weighted-recall, HARD-PEEL its bits (disjoint), reweight the
 *              uncovered positives, iterate until union coverage ≥ TR → a small ENSEMBLE of disjoint cores.
 *
 * Each core is then fitted to a standard `{Qbits, m, alpha}` part (mfit gate + recall-vote α) so the SAME
 * α-sum head + θ-tuning (`fit.js`) sits on top — making it a drop-in for `boost` under `fitClass({core:"adaptive"})`.
 */

import buildAffinity from "./affinity.js";
import replicate from "./replicator.js";
import mfit from "./mfit.js";
import andCount from "../math/sparse/andCount.js";

/**
 * @function adaptiveCore
 * @param {Array<number[]>} A - positive contexts (train).
 * @param {object} neg - buildNegSet output.
 * @param {object} [opts] STRONG=0.7 · VIABLE=0.55 · TR=0.85 · FPCAP=0.25 · FPSLOPE=0.015 · rho=80.
 * @returns {{G:Array<{Qbits:number[],m:number,alpha:number,recall:number,precision:number}>,
 *   stoppedStrong:boolean, r50:number, dynFP:number}}
 */
export const adaptiveCore = (A, neg, opts = {}) => {
  const { rho = 80, solver = "exp", STRONG = 0.7, VIABLE = 0.55, TR = 0.85, FPCAP = 0.25, FPSLOPE = 0.015, maxRounds = 12, suppPatience } = opts;
  const nA = A.length;
  const wUnif = new Float64Array(nA).fill(1 / nA);
  const rate = (b) => { if (!b.length) return 0; const s = [...b].sort((x, y) => x - y); let c = 0; for (const y of A) if (andCount(y, s) >= 1) c++; return c / nA; };
  const fpr = (b) => { if (!b.length || !neg.Neg.length) return 0; const s = [...b].sort((x, y) => x - y); let c = 0; for (const y of neg.Neg) if (andCount(y, s) >= 1) c++; return c / neg.Neg.length; };

  const peeled = new Set(), covered = new Uint8Array(nA), coreBits = [];
  let stoppedStrong = false, r50 = 0, dynFP = 0, cov = 0;

  for (let round = 0; round < maxRounds; round++) {
    let W = 0; const wt = new Float64Array(nA); for (let i = 0; i < nA; i++) { wt[i] = covered[i] ? 0.02 : 1; W += wt[i]; } for (let i = 0; i < nA; i++) wt[i] /= W;
    const { u, edges } = buildAffinity(A, wt, neg, { commonSet: neg.DtauSel, commonCoef: 1, exclude: peeled, excludeCoef: 0 });
    const r = replicate(u, edges, { solver, rho, ...(suppPatience ? { suppPatience } : {}) });
    const byW = r.Q.map((p, i) => [neg.pPlus[p], r.wPart[i]]).sort((a, b) => b[1] - a[1]).map((e) => e[0]);
    if (!byW.length) break;

    // bit → the round's covered positive contexts (for weight-first growth)
    const dset = new Set(byW), cmap = new Map();
    for (let i = 0; i < nA; i++) for (const b of A[i]) if (dset.has(b)) { let a = cmap.get(b); if (!a) { a = []; cmap.set(b, a); } a.push(i); }

    if (round === 0) {                                                     // concentration → dynamic FP ceiling
      const p = []; for (const b of byW) { p.push(b); if (rate(p) >= 0.5) break; } r50 = p.length;
      dynFP = Math.min(FPCAP, FPSLOPE * r50);
      const Rdom = rate(byW);
      if (Rdom >= STRONG && fpr(byW) <= dynFP) { coreBits.push(byW.slice()); stoppedStrong = true; break; }   // ship ONE clean core
    }

    // weak: grow weight-first until WEIGHTED recall ≥ VIABLE (buffer), on the uncovered focus
    const wc = new Uint8Array(nA); let wcov = 0; const weak = [];
    for (const b of byW) { weak.push(b); for (const i of (cmap.get(b) || [])) if (!wc[i]) { wc[i] = 1; wcov += wt[i]; } if (wcov >= VIABLE) break; }
    coreBits.push(weak);
    for (const b of weak) peeled.add(b);                                   // HARD-peel ⇒ next core is disjoint
    const ws = new Set(weak); for (let i = 0; i < nA; i++) if (!covered[i]) { for (const b of A[i]) if (ws.has(b)) { covered[i] = 1; cov++; break; } }
    if (cov / nA >= TR) break;
  }

  // fit each core to a standard part: mfit gate + recall-weighted AdaBoost vote α (clipped ≥ 0)
  const G = [];
  for (const c of coreBits) {
    const Qbits = [...new Set(c)].sort((a, b) => a - b);
    if (!Qbits.length) continue;
    const fit = mfit(Qbits, A, wUnif, neg.Neg);
    const m = fit.valid ? fit.m : 1;
    const rec = fit.valid ? fit.recall : (fit.recallAt1 || 0);
    const rc = Math.min(Math.max(rec, 0.501), 0.999);                      // clip so the vote stays positive
    const alpha = 0.5 * Math.log(rc / (1 - rc));
    G.push({ Qbits, m, alpha, recall: rec, precision: fit.precision ?? 0 });
  }
  return { G, stoppedStrong, r50, dynFP };
};

export default adaptiveCore;
