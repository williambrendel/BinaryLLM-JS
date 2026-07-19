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
import buildAffinity from "./affinity.js";
import replicate from "./replicator.js";

const sortAsc = (a) => [...new Set(a)].sort((x, y) => x - y);
const samp = (arr, n) => { if (arr.length <= n) return arr; const step = arr.length / n, out = []; for (let i = 0; i < arr.length; i += step) out.push(arr[Math.floor(i)]); return out; };

/**
 * §2.5 — refine a greedy part by running the replicator on ITS bits only (unary already handled by greedy). Builds
 * the affinity over the subset `Q` (reusing frozen neg stats), replicates, returns the dominant sub-clique — the
 * cohesive co-occurring core among the discriminative bits. Cheap: |Q|~hundreds of nodes, not ~10⁴. Falls back to Q
 * if the replicator returns empty. `andObjective` gates edges by g=max(0,u) (u>0 within Q, so ~scaling).
 */
export const refineOnSubset = (Q, A, w, neg, opts = {}) => {
  const { rho = 80, andObjective = false } = opts;
  if (Q.length < 3) return Q;
  const subNeg = { ...neg, pPlus: Q, pset: new Set(Q) };
  const { u, edges } = buildAffinity(A, w, subNeg, { andObjective });
  const r = replicate(u, edges, { solver: "exp", rho, suppPatience: 3 });
  const bits = r.Q.map((p) => Q[p]);
  return bits.length ? sortAsc(bits) : Q;
};

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
 * CLOSED-FORM MARGINAL selection — forward-greedy on `R−λ·FP` using the verified independence approximations. At each
 * step pick the bit maximizing the marginal `(1−rec)·r_b − λ·(1−fp)·f_b` (ΔR−λ·ΔFP), update rec/fp with the running
 * product, and STOP when the best marginal ≤ 0. O(1)/bit/step (no threshold sweep). The effective weight
 * `λ·(1−fp)/(1−rec)` auto-adapts: coverage-greedy early (recurring high-r bits), FP-averse as recall accumulates.
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

/** density-weighted rates: each context contributes 1/|x| (its share of a density gate). f_b^d = neg-density mass. */
export const bitRatesD = (Pplus, A, Neg, w) => {
  const pset = new Set(Pplus), r = new Map(), f = new Map();
  for (const b of Pplus) { r.set(b, 0); f.set(b, 0); }
  let Wp = 0; for (let i = 0; i < A.length; i++) Wp += w ? w[i] : 1;
  for (let i = 0; i < A.length; i++) { const wi = (w ? w[i] : 1) / (A[i].length || 1); for (const b of A[i]) if (pset.has(b)) r.set(b, r.get(b) + wi); }
  for (const b of Pplus) r.set(b, r.get(b) / (Wp || 1));
  for (const y of Neg) { const inv = 1 / (y.length || 1); for (const b of y) if (pset.has(b)) f.set(b, f.get(b) + inv); }
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
  const { Rstar = 0.6, lambda = 1, alpha = 1e-4, rD = null, fD = null, orderBy = "ratio", noKnee = false } = opts;  // rD/fD: density-weighted (option-3); orderBy "diff" vs "ratio"; noKnee: grow past the knee to high recall (single-part test)
  const sc = rD ? ((b) => (rD.get(b) + alpha) / (fD.get(b) + alpha)) : orderBy === "diff" ? ((b) => r.get(b) - lambda * f.get(b)) : ((b) => (r.get(b) + alpha) / (f.get(b) + alpha));
  const order = [...Pplus].sort((x, y) => sc(y) - sc(x));
  const Q = []; let rec = 0, fp = 0, best = null;
  for (const b of order) {
    const rb = r.get(b), fb = f.get(b);
    rec = 1 - (1 - rec) * (1 - rb); fp = 1 - (1 - fp) * (1 - fb); Q.push(b);
    if (rec >= Rstar) {
      const obj = rec - lambda * fp;
      if (!best || obj > best.obj) best = { Q: [...Q], rec, fp, obj };
      if (!noKnee && (rD ? rD.get(b) < lambda * fD.get(b) : rb < lambda * fb)) break;  // knee (skippable for the single-part test)
    }
  }
  return best ?? { Q: [...Q], rec, fp, obj: rec - lambda * fp, noValidT: true };
};

/**
 * MATCHED-METRIC selection — forward-select bits to maximize the DENSITY gate's `recall−λ·FP` (re-tuning the
 * threshold each step), instead of OR-recall. This aligns the selection metric with the deployed density gate, so a
 * bit is admitted only if it improves the gate we actually score on — down-ranking common scaffold bits that help OR
 * but lift the negatives' density. Selects on TRAIN pos + neg (both sampled by the caller for speed).
 */
export const buildPartDensity = (cands, trPos, negNeg, rr, ff, opts = {}) => {
  const { lambda = 1, capBits = 500, alpha = 1e-4, capCand = 2000 } = opts;
  const order = [...cands].sort((a, b) => ((rr.get(b) + alpha) / (ff.get(b) + alpha)) - ((rr.get(a) + alpha) / (ff.get(a) + alpha))).slice(0, capCand);
  const nv = trPos.length, nn = negNeg.length; if (!nv || !nn) return sortAsc(cands.slice(0, 1));
  const setV = trPos.map((x) => new Set(x)), setN = negNeg.map((x) => new Set(x));
  const lenV = trPos.map((x) => x.length || 1), lenN = negNeg.map((x) => x.length || 1);
  const cntV = new Int32Array(nv), cntN = new Int32Array(nn);
  const bestObj = () => {                                              // best recall−λ·FP over the density threshold
    const ds = new Array(nv + nn); for (let i = 0; i < nv; i++) ds[i] = [cntV[i] / lenV[i], 1]; for (let j = 0; j < nn; j++) ds[nv + j] = [cntN[j] / lenN[j], 0];
    ds.sort((a, b) => b[0] - a[0]); let tp = 0, fp = 0, bo = -Infinity;
    for (let k = 0; k < ds.length; k++) { if (ds[k][1]) tp++; else fp++; if (k === ds.length - 1 || ds[k][0] !== ds[k + 1][0]) { const o = tp / nv - lambda * (fp / nn); if (o > bo) bo = o; } }  // eval only at threshold boundaries (fire on ALL ≥ t, not a tie-prefix)
    return bo;
  };
  const Q = []; let cur = bestObj();
  for (const b of order) {
    if (Q.length >= capBits) break;
    for (let i = 0; i < nv; i++) if (setV[i].has(b)) cntV[i]++; for (let j = 0; j < nn; j++) if (setN[j].has(b)) cntN[j]++;
    const o = bestObj();
    if (o > cur + 1e-6) { Q.push(b); cur = o; }
    else { for (let i = 0; i < nv; i++) if (setV[i].has(b)) cntV[i]--; for (let j = 0; j < nn; j++) if (setN[j].has(b)) cntN[j]--; }  // revert
  }
  return Q.length ? sortAsc(Q) : sortAsc(order.slice(0, 1));
};

/**
 * §3 — DENSITY threshold `t` for `|Q∩x| ≥ t·|x|`, tuned on the recall−λ·FP objective using BOTH the positive
 * `τ_x` distribution (recall) AND the NEGATIVE one (p⁻ as the precision counter) — not just the recall quantile.
 * A soft recall floor (Rstar−0.15) keeps the part a weak learner; `t` is free to drop lower where the negatives
 * permit (lifting recall). Falls back to 0.2 when val is too thin (`|A_val| < 150`).
 */
/** threshold objective: "diff" = recall − λ·FP (Youden's J at λ=1); "prod" = recall·(1−FP) (G-mean², param-free);
 * "prodSqrt" = recall·√(1−FP) (softer FP penalty ⇒ recall boost, still param-free). */
export const objScore = (rec, fp, lambda, obj) =>
  obj === "prod" ? rec * (1 - fp) : obj === "prodSqrt" ? rec * Math.sqrt(Math.max(0, 1 - fp)) : rec - lambda * fp;

export const setThreshold = (Q, Aval, Neg, Rstar = 0.6, lambda = 1, obj = "diff") => {
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
    const o = objScore(rec, fp, lambda, obj);
    if (o > bo) { bo = o; bt = t; }
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
  const { valFrac = 0.25, Rstar = 0.6, delta = 0.05, maxRounds = 20, epsMin = 1e-6, peel = true, candidates = "pPlus", thObj = "diff", refine = false, refineAnd = false, selectBy = "or", orderBy = "ratio", noKnee = false } = opts;
  const lambda = Math.min(3, Math.max(1, 1500 / A.length));
  const nTr = Math.max(1, Math.floor(A.length * (1 - valFrac)));
  const tr = A.slice(0, nTr), val = A.slice(nTr);
  const neg = buildNegSet(tr, negPool, Mglob, { delta });
  // candidate bits P⁺: "pPlus" = the replicator's frequency-selected set, or "union" = the naive ⋁A (all positive bits).
  let Pcur = candidates === "union" ? sortAsc(tr.flatMap((x) => [...x])) : [...neg.pPlus];
  const w = new Float64Array(tr.length).fill(1 / tr.length), parts = [];

  for (let round = 0; round < maxRounds && Pcur.length; round++) {
    const { r, f } = bitRates(Pcur, tr, neg.Neg, w);                  // reweighted intra-rates ⇒ score shifts each round
    let Qs;
    if (selectBy === "density") Qs = buildPartDensity(Pcur, samp(tr, 1200), samp(neg.Neg, 2000), r, f, { lambda });  // matched-metric selection (bigger samples ⇒ less bit-pick overfit)
    else if (selectBy === "densityProxy") { const rf = bitRatesD(Pcur, tr, neg.Neg, w); const part = buildPartGreedy(Pcur, r, f, { Rstar, lambda, rD: rf.r, fD: rf.f }); Qs = sortAsc(part.Q); }  // per-bit density-mass proxy
    else if (selectBy === "marginal") Qs = buildPartMarginal(Pcur, r, f, { lambda });  // closed-form marginal forward greedy on R−λ·FP
    else { const part = buildPartGreedy(Pcur, r, f, { Rstar, lambda, orderBy, noKnee }); Qs = sortAsc(part.Q); }
    if (!Qs.length) break;
    if (refine) Qs = refineOnSubset(Qs, tr, w, neg, { andObjective: refineAnd });  // §2.5: replicator refines the greedy bits (dependence axis)
    const t = setThreshold(Qs, val, neg.Neg, Rstar, lambda, thObj);  // t tuned on the FP-aware objective (negatives)
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
    for (const c of cands) { const vr = Sv.filter((s) => s > c).length / Sv.length, vf = Sn.length ? Sn.filter((s) => s > c).length / Sn.length : 0; const o = objScore(vr, vf, lambda, thObj); if (o > bObj) { bObj = o; theta = c; } }
  }
  return { parts, theta, neg, fires: (x) => S(x) > theta };
};

export default fitClassGreedy;
