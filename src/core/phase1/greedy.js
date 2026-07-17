"use strict";

/**
 * @file greedy.js
 * @brief [CANDIDATE] Greedy bit-sort part construction — a fast baseline to measure against the replicator
 * (`replicator.js` §4–5). Greedy by per-bit discriminativeness `score=(r_b+α)/(f_b+α)`; IGNORES bit
 * co-occurrence (the pairwise `M`). Peels with the closed-form independence recall/FP running product
 * (verified accurate: FP ~exact, recall within ~0.04), then sets a DENSITY threshold `t=|Q∩x|/|x|` from
 * validation contexts. Deployed gate: `h(x)=[|Q∩x| ≥ t·|x|]`. Do NOT replace the replicator until CV says so.
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
      if (rb < lambda * fb) break;                                   // closed-form knee
    }
  }
  return best ?? { Q: [...Q], rec, fp, obj: rec - lambda * fp, noValidT: true };
};

/**
 * §3 — DENSITY threshold `t` for `|Q∩x| ≥ t·|x|`, tuned on the recall−λ·FP objective using BOTH the positive
 * `τ_x` distribution (recall) AND the NEGATIVE one (p⁻ as the precision counter) — not just the recall quantile.
 * A soft recall floor (Rstar−0.15) keeps the part a weak learner; `t` is free to drop lower where the negatives
 * permit (lifting recall). Falls back to 0.2 when val is too thin (`|A_val| < 150`).
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
    if (rec < floor) break;                                          // recall monotone-decreasing in t
    const fp = neg.length ? neg.filter((v) => v >= t).length / neg.length : 0;
    const obj = rec - lambda * fp;
    if (obj > bo) { bo = obj; bt = t; }
  }
  return Math.max(0.05, Math.min(0.5, bt));
};

/** density gate: fires iff the on-target overlap fraction clears `t`. `Qs` must be bit-sorted. */
export const gateFires = (x, Qs, t) => andCount(x, Qs) >= t * x.length;

const orRateGate = (parts, pool) => (pool.length ? pool.filter((x) => parts.some((p) => gateFires(x, p.Qs, p.t))).length / pool.length : 0);

/**
 * §5 — AdaBoost ensemble with `buildPartGreedy` as the weak learner + §6 quantile θ. Head:
 * `S(x)=Σ α_k·[|Q_k∩x| ≥ t_k·|x|] > θ`.
 * @returns {{parts:Array<{Qs:number[],t:number,alpha:number}>, theta:number, neg:object, fires:Function}}
 */
export const fitClassGreedy = (A, negPool, Mglob, opts = {}) => {
  const { valFrac = 0.25, Rstar = 0.6, delta = 0.05, maxRounds = 20, epsMin = 1e-6, peel = true, candidates = "pPlus" } = opts;
  const lambda = Math.min(3, Math.max(1, 1500 / A.length));
  const nTr = Math.max(1, Math.floor(A.length * (1 - valFrac)));
  const tr = A.slice(0, nTr), val = A.slice(nTr);
  const neg = buildNegSet(tr, negPool, Mglob, { delta });
  // candidate bits P⁺: "pPlus" = the replicator's frequency-selected set, or "union" = the naive ⋁A (all positive bits).
  let Pcur = candidates === "union" ? sortAsc(tr.flatMap((x) => [...x])) : [...neg.pPlus];
  const w = new Float64Array(tr.length).fill(1 / tr.length), parts = [];

  for (let round = 0; round < maxRounds && Pcur.length; round++) {
    const { r, f } = bitRates(Pcur, tr, neg.Neg, w);                  // reweighted intra-rates ⇒ score shifts each round
    const part = buildPartGreedy(Pcur, r, f, { Rstar, lambda });
    const Qs = sortAsc(part.Q); if (!Qs.length) break;
    const t = setThreshold(Qs, val, neg.Neg, Rstar, lambda);         // t tuned on recall−λ·FP (p⁻ counter)
    let recW = 0, W = 0; for (let i = 0; i < tr.length; i++) { W += w[i]; if (gateFires(tr[i], Qs, t)) recW += w[i]; } recW /= W || 1;
    if (recW < 0.5) break;                                            // weak-learner floor (below-random ⇒ dual bound spent)
    if (parts.some((p) => jaccard(p.Qs, Qs) > 0.6)) break;            // D6: this part duplicates a kept one ⇒ one bundle, stop
    const eps = Math.min(Math.max(1 - recW, epsMin), 1 - epsMin), alpha = 0.5 * Math.log((1 - eps) / eps);
    parts.push({ Qs, t, alpha });
    if (peel) { const qset = new Set(Qs); Pcur = Pcur.filter((b) => !qset.has(b)); }  // HARD PEEL — force the next part onto disjoint bits
    let Wn = 0; for (let i = 0; i < tr.length; i++) { if (gateFires(tr[i], Qs, t)) w[i] *= Math.exp(-alpha); Wn += w[i]; } for (let i = 0; i < tr.length; i++) w[i] /= Wn;
    if (orRateGate(parts, val) >= 0.99) break;                        // val union early-stop
  }

  // §6 ensemble θ = Rstar-quantile of the α-sum over val, then λ-refined vs the negative-S distribution
  const S = (x) => { let s = 0; for (const p of parts) if (gateFires(x, p.Qs, p.t)) s += p.alpha; return s; };
  let theta = 0;
  if (parts.length && val.length) {
    const Sv = val.map(S).sort((a, b) => b - a), Sn = neg.Neg.map(S);
    theta = Sv[Math.min(Sv.length - 1, Math.max(0, Math.floor(Rstar * val.length)))];
    const cands = [...new Set([0, ...Sv, ...Sn])].sort((a, b) => a - b);
    let bObj = -Infinity;                                             // λ-refine
    for (const c of cands) { const vr = Sv.filter((s) => s > c).length / Sv.length, vf = Sn.length ? Sn.filter((s) => s > c).length / Sn.length : 0; if (vr - lambda * vf > bObj) { bObj = vr - lambda * vf; theta = c; } }
  }
  return { parts, theta, neg, fires: (x) => S(x) > theta };
};

export default fitClassGreedy;
