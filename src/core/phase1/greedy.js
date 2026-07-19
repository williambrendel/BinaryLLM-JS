"use strict";

/**
 * @file greedy.js
 * @brief Greedy bit-sort part construction — the CV-validated fast path (matches the replicator at ~13× speed).
 * Greedy by per-bit discriminativeness `score=(r_b+α)/(f_b+α)`; IGNORES bit co-occurrence (the pairwise `M`) — CV
 * showed that co-occurrence buys a different bit representation, not different positives, so it's not load-bearing
 * (see docs/phase1_report.md §15). Peels with the closed-form independence recall/FP running product (verified
 * accurate: FP ~exact, recall within ~0.04), sets an FP-aware DENSITY threshold `t=|Q∩x|/|x|` (recall−λ·FP over the
 * positive AND negative τ distributions — p⁻ as the precision counter), and boosts (reweight+peel) into an ensemble.
 * Deployed gate: `h(x)=[|Q∩x| ≥ t·|x|]`.
 *
 * `selectBy`: "or" (default — sort-by-ratio + OR-recall peel) or "marginal" (the principled twin: forward-greedy on
 * the closed-form `R−λ·FP` marginal; α-free, ties "or"). Everything richer was tried and rejected — see the report.
 */

import andCount from "../math/sparse/andCount.js";
import buildNegSet from "./negSet.js";
import { jaccard } from "../math/sparse/jaccard.js";

const sortAsc = (a) => [...new Set(a)].sort((x, y) => x - y);

/**
 * §1 — per-bit intra/extra-class rates over candidate bits `Pplus`. `w` = optional AdaBoost weights over `A`.
 * @returns {{r:Map<number,number>, f:Map<number,number>}} r_b = weighted recall rate, f_b = FP rate.
 */
export const bitRates = (Pplus, A, Neg, w) => {
  const pset = new Set(Pplus), r = new Map(), f = new Map();
  for (const b of Pplus) { r.set(b, 0); f.set(b, 0); }
  let W = 0; for (let i = 0; i < A.length; i++) W += w ? w[i] : 1;
  for (let i = 0; i < A.length; i++) { const wi = w ? w[i] : 1; for (const b of A[i]) if (pset.has(b)) r.set(b, r.get(b) + wi); }
  for (const b of Pplus) r.set(b, r.get(b) / (W || 1));
  for (const y of Neg) for (const b of y) if (pset.has(b)) f.set(b, f.get(b) + 1);
  for (const b of Pplus) f.set(b, f.get(b) / (Neg.length || 1));
  return { r, f };
};

/**
 * §2 — sort bits by `score_b=(r_b+α)/(f_b+α)` DESC and peel to the recall floor, accumulating the closed-form
 * independence recall/FP running product (O(1)/step). Track the knee (best `rec−λ·fp` past `Rstar`); stop once a
 * marginal bit's recall gain drops below its λ-weighted FP cost (`r_b < λ·f_b`).
 * @returns {{Q:number[], rec:number, fp:number, obj:number, noValidT?:boolean}}
 */
export const buildPartGreedy = (Pplus, r, f, opts = {}) => {
  const { Rstar = 0.6, lambda = 1, alpha = 1e-4 } = opts;
  const sc = (b) => (r.get(b) + alpha) / (f.get(b) + alpha);
  const order = [...Pplus].sort((x, y) => sc(y) - sc(x));
  const Q = []; let rec = 0, fp = 0, best = null;
  for (const b of order) {
    const rb = r.get(b), fb = f.get(b);
    rec = 1 - (1 - rec) * (1 - rb); fp = 1 - (1 - fp) * (1 - fb); Q.push(b);
    if (rec >= Rstar) {
      const obj = rec - lambda * fp;
      if (!best || obj > best.obj) best = { Q: [...Q], rec, fp, obj };
      if (rb < lambda * fb) break;                                     // closed-form knee
    }
  }
  return best ?? { Q: [...Q], rec, fp, obj: rec - lambda * fp, noValidT: true };
};

/**
 * CLOSED-FORM MARGINAL selection (the principled twin of `buildPartGreedy`) — forward-greedy on `R−λ·FP` using the
 * verified independence approximations. Each step picks the bit maximizing the marginal `(1−rec)·r_b − λ·(1−fp)·f_b`
 * (ΔR−λ·ΔFP), updates rec/fp with the running product, and STOPS when the best marginal ≤ 0. α-FREE (a difference,
 * never divides). The effective weight `λ·(1−fp)/(1−rec)` auto-adapts: coverage-greedy early, FP-averse as recall
 * accumulates. CV: ties "or" on accuracy and speed (docs/phase1_report.md §15).
 */
export const buildPartMarginal = (Pplus, r, f, opts = {}) => {
  const { lambda = 1, capBits = 1000 } = opts;
  const rem = [...Pplus], Q = []; let rec = 0, fp = 0;
  while (Q.length < capBits && rem.length) {
    const c1 = 1 - rec, c2 = lambda * (1 - fp);
    let bi = -1, bm = -Infinity;
    for (let i = 0; i < rem.length; i++) { const b = rem[i], m = c1 * r.get(b) - c2 * f.get(b); if (m > bm) { bm = m; bi = i; } }
    if (bm <= 0) break;                                                // no bit improves R−λ·FP under the closed form
    const b = rem[bi]; rem[bi] = rem[rem.length - 1]; rem.pop();
    rec = 1 - (1 - rec) * (1 - r.get(b)); fp = 1 - (1 - fp) * (1 - f.get(b)); Q.push(b);
  }
  return sortAsc(Q);
};

/**
 * §3 — FP-aware DENSITY threshold `t` for `|Q∩x| ≥ t·|x|`, chosen to maximize `recall − λ·FP` over BOTH the positive
 * `τ_x` distribution (recall) AND the NEGATIVE one (p⁻ as the precision counter) — not just the recall quantile. A
 * soft recall floor (Rstar−0.15) keeps the part a weak learner; `t` is free to drop lower where the negatives permit
 * (lifting recall). Falls back to 0.2 when val is too thin (`|A_val| < 150`).
 */
export const setThreshold = (Q, Aval, Neg, Rstar = 0.6, lambda = 1) => {
  if (Aval.length < 150) return 0.2;
  const Qs = sortAsc(Q);
  const pos = Aval.map((x) => (x.length ? andCount(x, Qs) / x.length : 0));
  const neg = Neg.map((x) => (x.length ? andCount(x, Qs) / x.length : 0));
  const cands = [...new Set([0, ...pos, ...neg])].sort((a, b) => a - b);
  const floor = Rstar - 0.15;
  let bt = 0.2, bo = -Infinity;
  for (const t of cands) {
    const rec = pos.filter((v) => v >= t).length / pos.length;
    if (rec < floor) break;                                            // recall monotone-decreasing in t
    const fp = neg.length ? neg.filter((v) => v >= t).length / neg.length : 0;
    const o = rec - lambda * fp;
    if (o > bo) { bo = o; bt = t; }
  }
  return Math.max(0.05, Math.min(0.5, bt));
};

/** density gate: fires iff the on-target overlap fraction clears `t`. `Qs` must be bit-sorted. */
export const gateFires = (x, Qs, t) => andCount(x, Qs) >= t * x.length;

const orRateGate = (parts, pool) => (pool.length ? pool.filter((x) => parts.some((p) => gateFires(x, p.Qs, p.t))).length / pool.length : 0);

/**
 * §5 — AdaBoost-flavored ensemble (reweight covered + hard peel) with the greedy part as the weak learner + §6
 * quantile θ. The boost is genuinely needed: a single grown part reaches ~80% held recall, the ensemble ~85%
 * (+10pp on rare words), because one density gate can't cover all senses (docs/phase1_report.md §15). Head:
 * `S(x)=Σ α_k·[|Q_k∩x| ≥ t_k·|x|] > θ`.
 * @param {object} [opts] @param {"or"|"marginal"} [opts.selectBy="or"] @param {"pPlus"|"union"} [opts.candidates]
 * @returns {{parts:Array<{Qs:number[],t:number,alpha:number}>, theta:number, neg:object, fires:Function}}
 */
export const fitClassGreedy = (A, negPool, Mglob, opts = {}) => {
  const { valFrac = 0.25, Rstar = 0.6, delta = 0.05, maxRounds = 20, epsMin = 1e-6, peel = true, candidates = "pPlus", selectBy = "or" } = opts;
  const lambda = Math.min(3, Math.max(1, 1500 / A.length));
  const nTr = Math.max(1, Math.floor(A.length * (1 - valFrac)));
  const tr = A.slice(0, nTr), val = A.slice(nTr);
  const neg = buildNegSet(tr, negPool, Mglob, { delta });
  // candidate bits P⁺: "pPlus" = the replicator's frequency-selected set, or "union" = the naive ⋁A (all positive bits).
  let Pcur = candidates === "union" ? sortAsc(tr.flatMap((x) => [...x])) : [...neg.pPlus];
  const w = new Float64Array(tr.length).fill(1 / tr.length), parts = [];

  for (let round = 0; round < maxRounds && Pcur.length; round++) {
    const { r, f } = bitRates(Pcur, tr, neg.Neg, w);                   // reweighted intra-rates ⇒ score shifts each round
    const Qs = selectBy === "marginal" ? buildPartMarginal(Pcur, r, f, { lambda }) : sortAsc(buildPartGreedy(Pcur, r, f, { Rstar, lambda }).Q);
    if (!Qs.length) break;
    const t = setThreshold(Qs, val, neg.Neg, Rstar, lambda);          // FP-aware threshold (p⁻ counter)
    let recW = 0, W = 0; for (let i = 0; i < tr.length; i++) { W += w[i]; if (gateFires(tr[i], Qs, t)) recW += w[i]; } recW /= W || 1;
    if (recW < 0.5) break;                                             // weak-learner floor (below-random ⇒ dual bound spent)
    if (parts.some((p) => jaccard(p.Qs, Qs) > 0.6)) break;             // D6: this part duplicates a kept one ⇒ one bundle, stop
    const eps = Math.min(Math.max(1 - recW, epsMin), 1 - epsMin), alpha = 0.5 * Math.log((1 - eps) / eps);
    parts.push({ Qs, t, alpha });
    if (peel) { const qset = new Set(Qs); Pcur = Pcur.filter((b) => !qset.has(b)); }  // HARD PEEL — force the next part onto disjoint bits
    let Wn = 0; for (let i = 0; i < tr.length; i++) { if (gateFires(tr[i], Qs, t)) w[i] *= Math.exp(-alpha); Wn += w[i]; } for (let i = 0; i < tr.length; i++) w[i] /= Wn;
    if (orRateGate(parts, val) >= 0.99) break;                         // val union early-stop
  }

  // §6 ensemble θ = Rstar-quantile of the α-sum over val, then λ-refined vs the negative-S distribution
  const S = (x) => { let s = 0; for (const p of parts) if (gateFires(x, p.Qs, p.t)) s += p.alpha; return s; };
  let theta = 0;
  if (parts.length && val.length) {
    const Sv = val.map(S).sort((a, b) => b - a), Sn = neg.Neg.map(S);
    theta = Sv[Math.min(Sv.length - 1, Math.max(0, Math.floor(Rstar * val.length)))];
    const cands = [...new Set([0, ...Sv, ...Sn])].sort((a, b) => a - b);
    let bObj = -Infinity;                                              // λ-refine on recall − λ·FP
    for (const c of cands) { const vr = Sv.filter((s) => s > c).length / Sv.length, vf = Sn.length ? Sn.filter((s) => s > c).length / Sn.length : 0; if (vr - lambda * vf > bObj) { bObj = vr - lambda * vf; theta = c; } }
  }
  return { parts, theta, neg, fires: (x) => S(x) > theta };
};

export default fitClassGreedy;
