"use strict";

/**
 * @file replicator.js
 * @brief §4–§5 — Pavan–Pelillo replicator on the DIRECT inhomogeneous payoff
 *   π(x) = u + Mx − ρx     (no homogenization, no dense Q)
 * over the simplex Δ. Three solvers on the same payoff (std / exp / hybrid),
 * compared by the §5 selection rule; support(x*) is the dominant part.
 *
 *   f(x) = uᵀx + ½ xᵀ(M − ρI) x          (the StQP objective, for selection)
 *   std:  x_i ← x_i·(π_i+s)/(xᵀ(π+s)),  s = max(0,−min π)+ε_s   (scalar positivity shift, const on Δ)
 *   exp:  x_i ← x_i·exp(η π_i)/Σ            (sign-robust, no shift)
 *   hyb:  std where min π>0 on the active set, else exp
 */

/** Sparse payoff π = u + Mx − ρx, computed only over the active set (x>0). */
const payoff = (u, edges, rho, x, out) => {
  const n = u.length; out.fill(0);
  const { i, j, m } = edges;
  for (let e = 0; e < m.length; e++) { const a = i[e], b = j[e], w = m[e]; const xa = x[a], xb = x[b]; if (xb !== 0) out[a] += w * xb; if (xa !== 0) out[b] += w * xa; }
  for (let p = 0; p < n; p++) out[p] += u[p] - rho * x[p];
  return out;
};

const objective = (u, edges, rho, x) => {
  const { i, j, m } = edges; let quad = 0, lin = 0, nrm = 0;
  for (let e = 0; e < m.length; e++) quad += m[e] * x[i[e]] * x[j[e]];   // ½·2·Σ_{a<b} = Σ over stored (i<j) edges
  for (let p = 0; p < x.length; p++) { lin += u[p] * x[p]; nrm += x[p] * x[p]; }
  return lin + quad - 0.5 * rho * nrm;                                   // uᵀx + ½xᵀMx − ½ρ‖x‖²
};

/**
 * @function replicate
 * @param {Float64Array} u @param {{i:Int32Array,j:Int32Array,m:Float64Array}} edges
 * @param {object} [opts]
 * @param {"std"|"exp"|"hybrid"} [opts.solver="hybrid"] @param {number} [opts.rho=0]
 * @param {number} [opts.eta=1] @param {number} [opts.eps=1e-6] @param {number} [opts.maxIter=500]
 * @param {number} [opts.tauSupp] @param {number} [opts.tauActive] @param {Float64Array} [opts.x0]
 * @returns {{x:Float64Array, Q:number[], wPart:number[], iters:number, matvecs:number, f:number}}
 */
export const replicate = (u, edges, opts = {}) => {
  const n = u.length;
  const { solver = "hybrid", rho = 0, eta = 1, eps = 1e-6, maxIter = 500 } = opts;
  const tauSupp = opts.tauSupp ?? 1e-6 / n, tauActive = opts.tauActive ?? tauSupp, epsS = 1e-9;
  // Pelillo/Bomze offsets (fixed, precomputed — NOT the per-iteration adaptive shift):
  //  · shiftConst γ (≡ +γ·eeᵀ): xᵀ(A+γeeᵀ)x = xᵀAx+γ on Δ ⇒ maximizers UNCHANGED, but makes π+γ≥0 so the
  //    STANDARD replicator is well-defined with fundamental-theorem monotone ascent (no oscillation). Applied as
  //    an O(n) constant per iter — never densifies M. "auto" ⇒ γ = ρ − min u − min(0,min M) + ε (guarantees ≥0).
  //  · shiftDiag κ (≡ +κ·I): π += κ·xᵢ. Bomze regularizer — CHANGES the solution (rewards concentration if κ>0,
  //    spread if κ<0); κ<0 is exactly what −ρ does. Provided to experiment with concavity/support size.
  let shiftConst = opts.shiftConst, shiftDiag = opts.shiftDiag ?? 0;
  if (shiftConst === "auto") { let mU = Infinity, mM = 0; for (let p = 0; p < n; p++) if (u[p] < mU) mU = u[p]; for (let e = 0; e < edges.m.length; e++) if (edges.m[e] < mM) mM = edges.m[e]; shiftConst = rho - mU - mM + epsS; }
  const fixedShift = typeof shiftConst === "number";
  // support-stability early-stop: the OUTPUT is support(x*) + its weight-ordering, not the fully-converged x.
  // Stop once the support SET is unchanged for `suppPatience` iters (after `minIter`) — the replicator otherwise
  // burns its full budget refining weights we discard (§ measured: ~2.5× faster, recall/FP within 0.5pp).
  const suppPatience = opts.suppPatience ?? 0, minIter = opts.minIter ?? 20;
  let prevSupp = null, stable = 0;

  // expU: treat the unary channel in the ODDS-RATIO domain, not log-odds. u = log((p⁺+α)/(p⁻+α)); exp(β·u) is
  // the (positive) odds ratio ^β — a positive multiplicative node prior instead of a signed additive one. Edges
  // stay signed (exp'ing them densifies / can't kill repulsion — see notes). This is the asymmetric u-vs-M split.
  const uu = opts.expU ? (() => { const v = new Float64Array(n); for (let p = 0; p < n; p++) v[p] = Math.exp(opts.expU * u[p]); return v; })() : u;

  let x = opts.x0 ? Float64Array.from(opts.x0) : new Float64Array(n).fill(1 / n);
  const prev = new Float64Array(n), pi = new Float64Array(n), ex = new Float64Array(n);
  let iters = 0, matvecs = 0;
  // DC (matrix-split) scratch + β. M⁺/M⁻ partition M's edges (|M⁺|+|M⁻|=nnz ⇒ same total matvec work as exp),
  // so the two half-matvecs cost ~one full Mx. β keeps num/den positive: β ≥ |min M|. Converges to the true
  // optimum, parameter-free (no dt), negatives-native. Parts are solver-invariant so output ≈ exp.
  const isDC = solver === "dc";
  const pos = isDC ? new Float64Array(n) : null, negp = isDC ? new Float64Array(n) : null;
  let beta = 0; if (isDC) { let mM = 0; for (let e = 0; e < edges.m.length; e++) if (edges.m[e] < mM) mM = edges.m[e]; beta = Math.abs(mM) + epsS; }
  // InImDyn (Rota Bulò–Pelillo–Bomze): coordinate-ascent on the StQP. Each step picks ONE pure strategy —
  // invade (argmax payoff) or immunize (min support payoff), whichever is more out-of-equilibrium — and takes
  // the EXACT line-search step along z = target−x. Negatives-native (no division/shift), one Mz matvec/step.
  const isInim = solver === "inim";
  const zsc = isInim ? new Float64Array(n) : null, z2sc = isInim ? new Float64Array(n) : null, mzsc = isInim ? new Float64Array(n) : null;

  for (; iters < maxIter; iters++) {
    prev.set(x);
    if (!isDC) { payoff(uu, edges, rho, x, pi); matvecs++; }               // DC builds pi from its own split matvec
    if (!isDC && shiftDiag) for (let p = 0; p < n; p++) pi[p] += shiftDiag * x[p];  // +κ·I diagonal offset
    let minPi = Infinity; if (!isDC) for (let p = 0; p < n; p++) if (x[p] > 0 && pi[p] < minPi) minPi = pi[p];
    const useStd = solver === "std" || (solver === "hybrid" && (fixedShift || minPi > 0));

    if (isDC) {                                                         // matrix-splitting (DC): num=(M⁺x+u⁺+β),
      pos.fill(0); negp.fill(0);                                        // den=(M⁻x+u⁻+ρx)+⟨x,π⟩+β — positivity-safe
      const ei = edges.i, ej = edges.j, em = edges.m;
      for (let e = 0; e < em.length; e++) { const a = ei[e], b = ej[e], wgt = em[e]; if (wgt > 0) { pos[a] += wgt * x[b]; pos[b] += wgt * x[a]; } else { negp[a] += -wgt * x[b]; negp[b] += -wgt * x[a]; } }
      matvecs++;
      let g = 0; for (let p = 0; p < n; p++) { const up = uu[p]; pos[p] += up > 0 ? up : 0; negp[p] += (up < 0 ? -up : 0) + rho * x[p]; pi[p] = pos[p] - negp[p]; g += x[p] * pi[p]; }
      let S = 0; for (let p = 0; p < n; p++) { const den = negp[p] + g + beta; x[p] = den > epsS ? x[p] * (pos[p] + beta) / den : 0; S += x[p]; }
      if (S <= 0) { iters++; break; }
      for (let p = 0; p < n; p++) x[p] /= S;
    } else if (isInim) {                                                // InImDyn: exact-line-search coordinate ascent
      let iInv = 0, mxp = -Infinity, iImm = -1, mnp = Infinity;         // invade=argmax payoff, immunize=min support
      for (let p = 0; p < n; p++) { if (pi[p] > mxp) { mxp = pi[p]; iInv = p; } if (x[p] > 0 && pi[p] < mnp) { mnp = pi[p]; iImm = p; } }
      const ei = edges.i, ej = edges.j, em = edges.m;
      const buildZ = (arr, i, invade) => {                              // direction toward target strategy
        if (invade) { for (let p = 0; p < n; p++) arr[p] = -x[p]; arr[i] += 1; return true; }   // z = e_i − x
        const xi = x[i], dn = 1 - xi; if (dn <= epsS) return false; const s = xi / dn;
        for (let p = 0; p < n; p++) arr[p] = s * x[p]; arr[i] = s * (x[i] - 1); return true;      // z = [xi/(1−xi)](x − e_i)
      };
      const lineSearch = (arr) => {                                     // exact δ* + Δf along z (one Mz matvec)
        mzsc.fill(0); for (let e = 0; e < em.length; e++) { const a = ei[e], b = ej[e], wgt = em[e]; mzsc[a] += wgt * arr[b]; mzsc[b] += wgt * arr[a]; }
        matvecs++; let c1 = 0, c2 = 0, zz = 0; for (let p = 0; p < n; p++) { c1 += pi[p] * arr[p]; c2 += arr[p] * mzsc[p]; zz += arr[p] * arr[p]; }
        c2 -= rho * zz; let del = c2 < 0 ? -c1 / c2 : (c1 > 0 ? 1 : 0); if (del < 0) del = 0; else if (del > 1) del = 1;
        return { del, df: del * c1 + 0.5 * del * del * c2 };
      };
      // FIX: compare the ACTUAL Δf of invade vs immunize (raw-excess comparison never immunized ⇒ support bloat)
      let bestZ = null, bestDel = 0, bestDf = 0;
      if (buildZ(zsc, iInv, true)) { const r2 = lineSearch(zsc); if (r2.df > bestDf) { bestDf = r2.df; bestDel = r2.del; bestZ = zsc; } }
      if (iImm >= 0 && buildZ(z2sc, iImm, false)) { const r2 = lineSearch(z2sc); if (r2.df > bestDf) { bestDf = r2.df; bestDel = r2.del; bestZ = z2sc; } }
      if (!bestZ || bestDf <= eps) { iters++; break; }                  // no improving move ⇒ equilibrium
      let S = 0; for (let p = 0; p < n; p++) { const v = x[p] + bestDel * bestZ[p]; x[p] = v > 0 ? v : 0; S += x[p]; }
      if (S <= 0) { iters++; break; } for (let p = 0; p < n; p++) x[p] /= S;
    } else if (solver === "cont") {                                     // continuous-time replicator (forward Euler,
      let g = 0; for (let p = 0; p < n; p++) if (x[p] > 0) g += x[p] * pi[p];   // subtraction-based ⇒ negatives native)
      let S = 0; const dt = opts.dt ?? 0.2;                            // ẋ_i = x_i(π_i − ⟨x,π⟩); dt=0.2 converges best
      for (let p = 0; p < n; p++) { const v = x[p] + dt * x[p] * (pi[p] - g); x[p] = v > 0 ? v : 0; S += x[p]; }
      if (S <= 0) { iters++; break; }
      for (let p = 0; p < n; p++) x[p] /= S;
    } else if (useStd) {                                                // multiplicative + scalar shift
      const s = fixedShift ? shiftConst : (minPi > 0 ? 0 : -minPi + epsS); let denom = 0;   // fixed γ vs adaptive
      for (let p = 0; p < n; p++) denom += x[p] * (pi[p] + s);
      if (denom <= 0) { iters++; break; }
      for (let p = 0; p < n; p++) x[p] = x[p] * (pi[p] + s) / denom;
    } else {                                                            // exponential (sign-robust)
      let mx = -Infinity; for (let p = 0; p < n; p++) if (x[p] > 0 && pi[p] > mx) mx = pi[p];
      let Z = 0; for (let p = 0; p < n; p++) { const e = x[p] > 0 ? x[p] * Math.exp(eta * (pi[p] - mx)) : 0; ex[p] = e; Z += e; }
      if (Z <= 0) { iters++; break; }
      for (let p = 0; p < n; p++) x[p] = ex[p] / Z;
    }
    // active-set prune: zero out sub-floor mass (keeps the simplex sparse), renormalize
    let keepSum = 0; for (let p = 0; p < n; p++) { if (x[p] < tauActive) x[p] = 0; else keepSum += x[p]; }
    if (keepSum > 0) for (let p = 0; p < n; p++) x[p] /= keepSum;

    let d1 = 0; for (let p = 0; p < n; p++) d1 += Math.abs(x[p] - prev[p]);
    if (!isInim && d1 < eps) { iters++; break; }                         // inim has its own Δf-based stop

    if (suppPatience && iters >= minIter) {                              // stop when the extracted bit-set settles
      let cnt = 0, same = prevSupp !== null;
      for (let p = 0; p < n; p++) if (x[p] > tauSupp) { cnt++; if (same && !prevSupp.has(p)) same = false; }
      if (same && prevSupp && cnt !== prevSupp.size) same = false;
      if (same) { if (++stable >= suppPatience) { iters++; break; } }
      else { stable = 0; prevSupp = new Set(); for (let p = 0; p < n; p++) if (x[p] > tauSupp) prevSupp.add(p); }
    }
  }

  const Q = [], wPart = []; for (let p = 0; p < n; p++) if (x[p] > tauSupp) { Q.push(p); wPart.push(x[p]); }
  // §5 cliff-vs-slope diagnostic: gap at the support edge (large ⇒ cliff/plateau, small ⇒ slope)
  const sortedX = [...x].sort((a, b) => b - a); const k = Q.length;
  const cliffGap = k > 0 && k < n ? (sortedX[k - 1] - sortedX[k]) / (sortedX[k - 1] || 1) : (k >= n ? 0 : 1);
  return { x, Q, wPart, iters, matvecs, f: objective(uu, edges, rho, x), cliffGap };
};

export default replicate;
