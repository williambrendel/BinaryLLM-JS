"use strict";

/**
 * @file replicator.js
 * @brief §4–§5 — replicator dynamics on the inhomogeneous payoff  π(x) = u + Mx − ρx  over the simplex Δ.
 * support(x*) is the dominant part. Two solvers, both parts-equivalent (the extracted support is the same
 * within CV noise); pick by preference:
 *
 *   f(x) = uᵀx + ½ xᵀ(M − ρI) x                                   the StQP objective
 *   exp:  x_i ← x_i·exp(π_i)/Σ                                     Pelillo exponential; sign-robust, DEFAULT
 *   dc :  x_i ← x_i·(M⁺x+u⁺+β)/((M⁻x+u⁻+ρx)+⟨x,π⟩+β)              matrix-split; parameter-free, converges
 *
 * The output is support(x*) + its weight-ordering, NOT the fully-converged x — so `suppPatience` stops the
 * iteration once the support SET settles (≈2.6× faster than running to weight-convergence, quality-neutral).
 */

/** Sparse payoff π = u + Mx − ρx, over the active set (x>0). */
const payoff = (u, edges, rho, x, out) => {
  const n = u.length; out.fill(0);
  const { i, j, m } = edges;
  for (let e = 0; e < m.length; e++) { const a = i[e], b = j[e], w = m[e]; const xa = x[a], xb = x[b]; if (xb !== 0) out[a] += w * xb; if (xa !== 0) out[b] += w * xa; }
  for (let p = 0; p < n; p++) out[p] += u[p] - rho * x[p];
  return out;
};

const objective = (u, edges, rho, x) => {
  const { i, j, m } = edges; let quad = 0, lin = 0, nrm = 0;
  for (let e = 0; e < m.length; e++) quad += m[e] * x[i[e]] * x[j[e]];
  for (let p = 0; p < x.length; p++) { lin += u[p] * x[p]; nrm += x[p] * x[p]; }
  return lin + quad - 0.5 * rho * nrm;
};

/**
 * @function replicate
 * @param {Float64Array} u @param {{i:Int32Array,j:Int32Array,m:Float64Array}} edges
 * @param {object} [opts] @param {"exp"|"dc"} [opts.solver="exp"] @param {number} [opts.rho=0]
 * @param {number} [opts.eps=1e-6] @param {number} [opts.maxIter=500] @param {number} [opts.suppPatience=0]
 * @returns {{x:Float64Array, Q:number[], wPart:number[], iters:number, f:number}}
 *   Q = support(x*) indices, wPart = their weights (descending selection uses these).
 */
export const replicate = (u, edges, opts = {}) => {
  const n = u.length;
  const { solver = "exp", rho = 0, eps = 1e-6, maxIter = 500 } = opts;
  const tauSupp = opts.tauSupp ?? 1e-6 / n, tauActive = opts.tauActive ?? tauSupp, epsS = 1e-9;
  const suppPatience = opts.suppPatience ?? 0, minIter = opts.minIter ?? 20;
  let prevSupp = null, stable = 0;

  let x = new Float64Array(n).fill(1 / n);
  const prev = new Float64Array(n), pi = new Float64Array(n), ex = new Float64Array(n);
  let iters = 0;

  // dc scratch + β. M⁺/M⁻ partition M's edges (|M⁺|+|M⁻|=nnz ⇒ same total matvec work as exp). β ≥ |min M| keeps
  // num/den positive.
  const isDC = solver === "dc";
  const pos = isDC ? new Float64Array(n) : null, negp = isDC ? new Float64Array(n) : null;
  let beta = 0; if (isDC) { let mM = 0; for (let e = 0; e < edges.m.length; e++) if (edges.m[e] < mM) mM = edges.m[e]; beta = Math.abs(mM) + epsS; }

  for (; iters < maxIter; iters++) {
    prev.set(x);
    if (isDC) {                                                         // matrix-splitting (DC)
      pos.fill(0); negp.fill(0);
      const ei = edges.i, ej = edges.j, em = edges.m;
      for (let e = 0; e < em.length; e++) { const a = ei[e], b = ej[e], wgt = em[e]; if (wgt > 0) { pos[a] += wgt * x[b]; pos[b] += wgt * x[a]; } else { negp[a] += -wgt * x[b]; negp[b] += -wgt * x[a]; } }
      let g = 0; for (let p = 0; p < n; p++) { const up = u[p]; pos[p] += up > 0 ? up : 0; negp[p] += (up < 0 ? -up : 0) + rho * x[p]; pi[p] = pos[p] - negp[p]; g += x[p] * pi[p]; }
      let S = 0; for (let p = 0; p < n; p++) { const den = negp[p] + g + beta; x[p] = den > epsS ? x[p] * (pos[p] + beta) / den : 0; S += x[p]; }
      if (S <= 0) { iters++; break; }
      for (let p = 0; p < n; p++) x[p] /= S;
    } else {                                                            // exponential (sign-robust)
      payoff(u, edges, rho, x, pi);
      let mx = -Infinity; for (let p = 0; p < n; p++) if (x[p] > 0 && pi[p] > mx) mx = pi[p];
      let Z = 0; for (let p = 0; p < n; p++) { const e = x[p] > 0 ? x[p] * Math.exp(pi[p] - mx) : 0; ex[p] = e; Z += e; }
      if (Z <= 0) { iters++; break; }
      for (let p = 0; p < n; p++) x[p] = ex[p] / Z;
    }
    // active-set prune: zero out sub-floor mass, renormalize
    let keepSum = 0; for (let p = 0; p < n; p++) { if (x[p] < tauActive) x[p] = 0; else keepSum += x[p]; }
    if (keepSum > 0) for (let p = 0; p < n; p++) x[p] /= keepSum;

    let d1 = 0; for (let p = 0; p < n; p++) d1 += Math.abs(x[p] - prev[p]);
    if (d1 < eps) { iters++; break; }

    if (suppPatience && iters >= minIter) {                             // stop once the extracted bit-set settles
      let cnt = 0, same = prevSupp !== null;
      for (let p = 0; p < n; p++) if (x[p] > tauSupp) { cnt++; if (same && !prevSupp.has(p)) same = false; }
      if (same && prevSupp && cnt !== prevSupp.size) same = false;
      if (same) { if (++stable >= suppPatience) { iters++; break; } }
      else { stable = 0; prevSupp = new Set(); for (let p = 0; p < n; p++) if (x[p] > tauSupp) prevSupp.add(p); }
    }
  }

  const Q = [], wPart = []; for (let p = 0; p < n; p++) if (x[p] > tauSupp) { Q.push(p); wPart.push(x[p]); }
  return { x, Q, wPart, iters, f: objective(u, edges, rho, x) };
};

export default replicate;
