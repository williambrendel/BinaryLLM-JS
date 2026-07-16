"use strict";

// benchmark/discrimMask.js — discriminativeness masking. d(b)=(freq_A+ε)/(freq_Ā+ε).
//   M_τ(κ) = ¬{ κ⁻¹ < d(b) < κ }  (keep BOTH tails: positive d≥κ AND negative d≤1/κ; mask the middle).
//   p⁺ = {d ≥ κ},  p⁻ = {d ≤ 1/κ}  (from the SAME ranking).  score=(|p⁺∧M∧x|−|p⁻∧M∧x|)/|M∧x| > t.
// κ-sweep = Pareto (FP vs recall). Overlaid on: NONE, GLOBAL freq mask, CONTRAST top-k (one-sided).
// p⁻ ablation: d-ranking p⁻ vs FP-coverage PEEL, under the discriminativeness mask.
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/discrimMask.js english.txt <corpus>

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { encodeWord } from "../src/core/signatures/wordEncoder.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGET = process.env.TARGET || "bank";
const R = Number(process.env.R || 5), D = R + 1, DD = Number(process.env.DELTA || 0.001), K = Number(process.env.K || 30);

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F;
  const pc = new Map(); const partsOf = (w) => { let a = pc.get(w); if (!a) { a = [...new Set(encodeWord(dict, w))]; pc.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const ws = tokenizeStream(line.slice(seg.start, seg.end)).filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws);
  }
  const band = (d) => (d === 1 ? 2 : d <= 3 ? 1 : 0);
  const stmp = new Int32Array(DIM).fill(-1); let sp = 0;
  const feat = (s, t) => { sp++; const y = []; for (let d = 1; d <= D; d++) { const q = t - d; if (q < 0) break; const off = band(d) * F; for (const a of partsOf(s[q])) { const b = a + off; if (stmp[b] !== sp) { stmp[b] = sp; y.push(b); } } } return y; };

  const M = new Float64Array(DIM); let Ntot = 0; const POS = []; const neg = []; let ni = 0, negTot = 0;
  for (const s of sents) for (let t = 1; t < s.length; t++) if (s[t] !== TARGET) negTot++;
  const stride = Math.max(1, Math.floor(negTot / 150000));
  for (const s of sents) for (let t = 1; t < s.length; t++) { const f = feat(s, t); Ntot++; for (const b of f) M[b]++; if (s[t] === TARGET) POS.push(f); else if (ni++ % stride === 0) neg.push(f); }
  const rare = (b) => M[b] / Ntot <= DD;
  const cntA = new Float64Array(DIM); for (const f of POS) for (const b of f) cntA[b]++;
  const A = POS.length, Abar = Ntot - A, eps = 1 / Abar;
  const d = new Float64Array(DIM); for (let b = 0; b < DIM; b++) { const cAb = cntA[b], cBb = M[b] - cAb; if (cAb === 0 && M[b] === 0) { d[b] = 1; continue; } d[b] = (cAb / A + eps) / (cBb / Abar + eps); }

  const rc = (arr, fn) => { let k = 0; for (const f of arr) if (fn(f)) k++; return k; };
  const eval4 = (fn) => ({ rec: rc(POS, fn) / POS.length, fp: rc(neg, fn) / neg.length });
  const prec = (rec, fp) => { const TP = rec * POS.length, FP = fp * negTot; return TP + FP > 0 ? TP / (TP + FP) : 0; };

  const w = process.stdout;
  w.write(`# discrimMask — ${path.basename(corpusArg)} | TARGET=${TARGET} | |A|=${A} negSample=${neg.length} negTot=${negTot}\n`);

  // ── discriminativeness κ-sweep (t=0 gate: net positive evidence) ──
  w.write(`\n── DISCRIMINATIVENESS band  M_τ=¬{κ⁻¹<d<κ},  p⁺={d≥κ}, p⁻={d≤1/κ},  fire iff (|p⁺∧x|−|p⁻∧x|)>0 ──\n`);
  w.write(`   κ      |p⁺|   |p⁻|   |M_τ|    recall   FP%      precision\n`);
  for (const k of [1.2, 1.5, 2, 3, 5, 10]) {
    let np = 0, nm = 0, nM = 0; for (let b = 0; b < DIM; b++) { if (M[b] === 0) continue; if (d[b] >= k) { np++; nM++; } else if (d[b] <= 1 / k) { nm++; nM++; } }
    const fn = (f) => { let pp = 0, pm = 0; for (const b of f) { if (d[b] >= k) pp++; else if (d[b] <= 1 / k) pm++; } return (pp - pm) > 0; };
    const e = eval4(fn);
    w.write(`   ${String(k).padEnd(4)}  ${String(np).padStart(6)} ${String(nm).padStart(6)} ${String(nM).padStart(7)}   ${(100 * e.rec).toFixed(1).padStart(5)}%   ${(100 * e.fp).toFixed(3).padStart(6)}%   ${(100 * prec(e.rec, e.fp)).toFixed(2).padStart(6)}%\n`);
  }

  // ── baselines: NONE, GLOBAL freq, CONTRAST top-k (one-sided) — corner gate coverage=1 ──
  const OR = new Uint8Array(DIM); for (let b = 0; b < DIM; b++) if (cntA[b] > 0) OR[b] = 1;
  const orBits = []; for (let b = 0; b < DIM; b++) if (OR[b]) orBits.push(b); orBits.sort((a, b) => d[b] - d[a]);
  const topk = (k) => { const p = new Uint8Array(DIM); for (let i = 0; i < Math.min(k, orBits.length); i++) p[orBits[i]] = 1; return p; };
  const cornerV = (pset) => (f) => { let h = false; for (const b of f) { if (!OR[b]) return false; if (pset[b]) h = true; } return h; };
  const cornerNone = (f) => { for (const b of f) if (!OR[b]) return false; return f.length > 0; };
  const cornerGlobal = (f) => { let h = false; for (const b of f) { if (!OR[b]) return false; if (rare(b)) h = true; } return h; };
  w.write(`\n── baselines (corner gate coverage=1) ──\n`);
  const line = (name, fn) => { const e = eval4(fn); w.write(`   ${name.padEnd(22)} recall ${(100 * e.rec).toFixed(1).padStart(5)}%   FP ${(100 * e.fp).toFixed(3).padStart(6)}%   prec ${(100 * prec(e.rec, e.fp)).toFixed(2)}%\n`); };
  line("NONE (x⊆OR)", cornerNone);
  line(`GLOBAL freq (δ=${DD})`, cornerGlobal);
  for (const k of [300, 800]) line(`CONTRAST top-${k}`, cornerV(topk(k)));

  // ── p⁻ ablation: discriminativeness mask + peeled p⁻ vs + d-ranking p⁻ (κ=2) ──
  const kap = 2; const pPlus = new Uint8Array(DIM), pMinusD = new Uint8Array(DIM), keep = new Uint8Array(DIM);
  for (let b = 0; b < DIM; b++) { if (M[b] === 0) continue; if (d[b] >= kap) { pPlus[b] = 1; keep[b] = 1; } else if (d[b] <= 1 / kap) { pMinusD[b] = 1; keep[b] = 1; } }
  // peel p⁻: common∉p⁺ bits (here: kept negatives) appearing in ≥K gate-1 FPs
  const cnp = new Map(); for (const f of neg) { let ok = true; for (const b of f) if (rare(b) && !(d[b] >= kap)) { ok = false; break; } if (!ok) continue; for (const b of f) if (!(d[b] >= kap) && keep[b]) cnp.set(b, (cnp.get(b) || 0) + 1); }
  const pMinusPeel = new Uint8Array(DIM); let peN = 0; for (const [b, c] of cnp) if (c >= K) { pMinusPeel[b] = 1; peN++; }
  const scored = (pm) => (f) => { let pp = 0, mm = 0, kk = 0; for (const b of f) { if (!keep[b]) continue; kk++; if (pPlus[b]) pp++; else if (pm[b]) mm++; } return kk > 0 && (pp - mm) / kk > 0; };
  w.write(`\n── p⁻ ABLATION (discriminativeness mask, κ=${kap};  |p⁺|=${pPlus.reduce((a, x) => a + x, 0)}) ──\n`);
  { const e = eval4(scored(pMinusD)); w.write(`   p⁻ = d-ranking {d≤1/κ}  (|p⁻|=${pMinusD.reduce((a, x) => a + x, 0)})   recall ${(100 * e.rec).toFixed(1)}%  FP ${(100 * e.fp).toFixed(3)}%  prec ${(100 * prec(e.rec, e.fp)).toFixed(2)}%\n`); }
  { const e = eval4(scored(pMinusPeel)); w.write(`   p⁻ = FP-coverage PEEL   (|p⁻|=${peN})   recall ${(100 * e.rec).toFixed(1)}%  FP ${(100 * e.fp).toFixed(3)}%  prec ${(100 * prec(e.rec, e.fp)).toFixed(2)}%\n`); }
};

main();
