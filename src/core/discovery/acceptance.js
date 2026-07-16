"use strict";

/**
 * @file acceptance.js
 * @brief §9 evaluation — runs a discovery model through the §7 max-pool and
 * reports the §9.1 primary metrics (train + held) plus the §9.3 coherence
 * diagnostics, and emits the §9.5 CSV rows. Scored on `fires(x)`, never on a
 * single part.
 */

import makeReadout from "./maxReadout.js";
import poolMetrics from "./metrics.js";
import peelNegative from "./peelNegative.js";
import andCount from "../math/sparse/andCount.js";

/** p⁺_glob ∧ M_τ = OR(A) ∖ D_τ (sorted). */
export const adaptedPositive = (A, Dtau) => {
  const Dset = new Set(Dtau), s = new Set();
  for (const y of A) for (const b of y) if (!Dset.has(b)) s.add(b);
  return Uint32Array.from(s).sort();
};

/**
 * §9.3 coherence diagnostics for a set of gates.
 * @returns {{K:number, sumPartSize:number, rho:number, overlap:number,
 *   containmentOk:boolean, pMinus:{min:number, mean:number, max:number},
 *   pMinusGlobal:number, negGrowthOk:boolean}}
 */
export const coherenceDiagnostics = (gates, A, Dtau, opts = {}) => {
  const { kappa = 2, m = 2, AbarH = null } = opts;
  const Pplus = adaptedPositive(A, Dtau), Pset = new Set(Pplus), Dset = new Set(Dtau);
  const union = new Set(); let sumSize = 0;
  let containmentOk = true;
  for (const g of gates) { sumSize += g.Q.length; for (const b of g.Q) { union.add(b); if (Dset.has(b) || !Pset.has(b)) containmentOk = false; } }
  const pm = gates.map((g) => g.pMinus.length);
  const pMinusGlobal = AbarH ? peelNegative([...Pplus], AbarH, A, { Dtau, kappa, m }).pMinus.length : NaN;
  return {
    K: gates.length,
    sumPartSize: sumSize,
    rho: Pplus.length ? union.size / Pplus.length : 0,                 // §9.3.2 positive conservation
    overlap: union.size ? sumSize / union.size : 0,                    // §9.3.3
    containmentOk,                                                     // §9.3.4
    pMinus: { min: Math.min(...pm, 0), mean: pm.length ? pm.reduce((a, b) => a + b, 0) / pm.length : 0, max: Math.max(...pm, 0) },
    pMinusGlobal,
    negGrowthOk: Number.isNaN(pMinusGlobal) ? null : pm.every((x) => x >= pMinusGlobal),  // §9.3.1
  };
};

/**
 * §9.1 primary metrics on the max-pool for train and held positive slices vs a
 * shared negative pool.
 * @returns {{train:object, held:object}}
 */
export const evaluateModel = (model, Atrain, Aheld, Abar) => {
  const rd = makeReadout(model.gates, model.Dtau);
  const fires = (x) => rd.fires(x);
  return { train: poolMetrics(fires, Atrain, Abar), held: poolMetrics(fires, Aheld, Abar) };
};

/** Deterministic 50/50 split of an array by index parity-free halving. */
export const halve = (arr) => { const h = arr.length >> 1; return [arr.slice(0, h), arr.slice(h)]; };

/**
 * Assemble one §9.5 CSV-ready row object for a produced model.
 * @param {object} model - {Dtau, gates, algo}
 * @param {Array} Atrain @param {Array} Aheld @param {Array} Abar
 * @param {object} cfg - {K, m, kappa, representation, lambda?}
 * @param {Array} AbarH - hard negatives, for re-based p⁻_global (§9.3.1)
 */
export const acceptanceRow = (model, Atrain, Aheld, Abar, cfg, AbarH) => {
  const ev = evaluateModel(model, Atrain, Aheld, Abar);
  const dg = coherenceDiagnostics(model.gates, [...Atrain, ...Aheld], model.Dtau, { kappa: cfg.kappa, m: cfg.m, AbarH });
  return {
    algo: model.algo, K: dg.K, m: cfg.m, kappa: cfg.kappa, representation: cfg.representation || "fuzzy", lambda: cfg.lambda ?? "",
    train_precision: ev.train.precision, train_recall: ev.train.recall, train_accuracy: ev.train.accuracy, train_FP: ev.train.fpRate,
    held_precision: ev.held.precision, held_recall: ev.held.recall, held_accuracy: ev.held.accuracy, held_FP: ev.held.fpRate,
    sum_part_size: dg.sumPartSize, mask_size: model.Dtau.length,
    pMinus_min: dg.pMinus.min, pMinus_mean: dg.pMinus.mean, pMinus_max: dg.pMinus.max, pMinus_global: dg.pMinusGlobal,
    rho: dg.rho, overlap: dg.overlap, containment_ok: dg.containmentOk, neg_growth_ok: dg.negGrowthOk,
  };
};

/** Serialize rows (array of flat objects) to CSV text. */
export const toCSV = (rows) => {
  if (rows.length === 0) return "";
  const cols = Object.keys(rows[0]);
  const fmt = (v) => (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(6)) : String(v));
  return [cols.join(","), ...rows.map((r) => cols.map((c) => fmt(r[c])).join(","))].join("\n");
};

export default acceptanceRow;
