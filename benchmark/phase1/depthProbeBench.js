"use strict";

// benchmark/phase1/depthProbeBench.js — [PROTOTYPE] DIAGNOSTIC: is "gate-model ≈ linear" a property of the DATA
// (task ~linearly separable) or the METHOD (optimizer can't find nonlinear structure)?  Build synthetic labels
// from two cue bits present in the real context x:  y_XOR = (b1∈x) ⊕ (b2∈x)  (NOT linearly separable — linear
// ceiling ~75%),  y_AND = (b1∈x) ∧ (b2∈x)  (linearly separable — a single LTF does it).  Compare a LINEAR
// classifier vs a 1-hidden-layer THRESHOLD-GATE model on both.  If gate ≫ linear on XOR ⇒ the gate nonlinearity
// IS exploited when present ⇒ any real-task tie means the real task is ~linearly separable (depth buys nothing
// here).  If gate ties linear on XOR too ⇒ the optimizer is the problem.
//   node --max-old-space-size=8192 benchmark/phase1/depthProbeBench.js <dict> <corpus>

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../../src/core/parts/tokenize.js";
import { encodeWord } from "../../src/core/signatures/wordEncoder.js";
import { segmentText } from "../../src/utilities/textSegmentation/segmentText.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const R = Number(process.env.R || 5), D = R + 1;
const NCTX = Number(process.env.NCTX || 60000), MINBITFREQ = Number(process.env.MINBITFREQ || 20);
const DGATES = Number(process.env.DGATES || 64), EPOCHS = Number(process.env.EPOCHS || 20), LR = Number(process.env.LR || 0.2);
const BATCH = Number(process.env.BATCH || 64), MOM = Number(process.env.MOM || 0.9), WINIT = Number(process.env.WINIT || 0.3), CLIP = 1.0;

const [dictArg, corpusArg] = process.argv.slice(2);
const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F;
const pc = new Map(); const partsOf = (wd) => { let a = pc.get(wd); if (!a) { a = [...new Set(encodeWord(dict, wd))]; pc.set(wd, a); } return a; };
const sents = [];
for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
  const ws = tokenizeStream(line.slice(seg.start, seg.end)).filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
  if (ws.length >= 2) sents.push(ws);
}
const band = (dd) => (dd === 1 ? 2 : dd <= 3 ? 1 : 0);
const stmp = new Int32Array(DIM).fill(-1); let sp = 0;
const feat = (s, t) => { sp++; const y = []; for (let dd = 1; dd <= D; dd++) { const q = t - dd; if (q < 0) break; const off = band(dd) * F; for (const a of partsOf(s[q])) { const b = a + off; if (stmp[b] !== sp) { stmp[b] = sp; y.push(b); } } } return y; };

// collect a pool of real context vectors (any position)
const ctx = []; let step = Math.max(1, Math.floor((sents.length * 6) / NCTX));
let cc = 0; for (const s of sents) { for (let t = 1; t < s.length; t++) { if (cc++ % step === 0) ctx.push(feat(s, t)); } if (ctx.length >= NCTX) break; }
const bitCount = new Map(); for (const x of ctx) for (const b of x) bitCount.set(b, (bitCount.get(b) || 0) + 1);
// compact vocabulary of frequent bits
const idx = new Map(); for (const [b, c] of bitCount) if (c >= MINBITFREQ) idx.set(b, idx.size);
const V = idx.size;
const enc = (x) => { const o = []; for (const b of x) { const j = idx.get(b); if (j !== undefined) o.push(j); } return Int32Array.from(o); };
const X = ctx.map(enc);
// pick two cue bits with freq near 0.3 and low correlation (so XOR is balanced and both matter)
const cf = new Float64Array(V); for (const x of X) for (const j of x) cf[j]++; for (let j = 0; j < V; j++) cf[j] /= X.length;
// GROUP cues: two disjoint bit-sets, each with union-coverage ~0.5, so the nonlinear (1,1) quadrant is large.
// Single context bits max out at low freq, so single-bit XOR can't be made hard — groups fix that.
let pool = []; for (let j = 0; j < V; j++) if (cf[j] > 0.03 && cf[j] < 0.2) pool.push(j);
pool.sort((a, b) => cf[b] - cf[a]);
const GA = new Set(), GB = new Set(); let miss = new Float64Array(X.length).fill(1), covA = 0, covB = 0;
// greedily add bits (alternating) until each group covers ~0.5 of contexts
const coverageIfAdd = (set, j) => { let c = 0; for (let i = 0; i < X.length; i++) { let hit = false; for (const k of X[i]) if (set.has(k) || k === j) { hit = true; break; } if (hit) c++; } return c / X.length; };
for (const j of pool) { if (covA <= covB && covA < 0.5) { GA.add(j); covA = coverageIfAdd(GA, -1); } else if (covB < 0.5) { GB.add(j); covB = coverageIfAdd(GB, -1); } if (covA >= 0.5 && covB >= 0.5) break; }
const hitsA = (x) => { for (const k of x) if (GA.has(k)) return true; return false; };
const hitsB = (x) => { for (const k of x) if (GB.has(k)) return true; return false; };
process.stderr.write(`|GA|=${GA.size} cov=${covA.toFixed(2)} · |GB|=${GB.size} cov=${covB.toFixed(2)}\n`);
process.stderr.write(`ctx=${X.length} V=${V}\n`);

const mkLabels = (kind) => X.map((x) => { const a = hitsA(x), b = hitsB(x); return kind === "XOR" ? (a !== b ? 1 : 0) : (a && b ? 1 : 0); });
const split = (Y) => { const trX = [], trY = [], hX = [], hY = []; for (let i = 0; i < X.length; i++) (i % 5 === 0 ? (hX.push(X[i]), hY.push(Y[i])) : (trX.push(X[i]), trY.push(Y[i]))); return { trX, trY, hX, hY }; };
const baserate = (Y) => { const p = Y.reduce((a, b) => a + b, 0) / Y.length; return Math.max(p, 1 - p); };

// LINEAR 2-class log-odds classifier (counted) — the linear ceiling
const linAcc = (trX, trY, hX, hY) => {
  const c1 = new Float64Array(V), c0 = new Float64Array(V); let n1 = 0, n0 = 0;
  for (let i = 0; i < trX.length; i++) { const t = trY[i] ? c1 : c0; if (trY[i]) n1++; else n0++; for (const j of trX[i]) t[j]++; }
  const AL = 1 / trX.length, u = new Float64Array(V); for (let j = 0; j < V; j++) u[j] = Math.log((c1[j] / (n1 || 1) + AL) / (c0[j] / (n0 || 1) + AL));
  const bias = Math.log((n1 + 1) / (n0 + 1));
  let ok = 0; for (let i = 0; i < hX.length; i++) { let s = bias; for (const j of hX[i]) s += u[j]; if ((s > 0 ? 1 : 0) === hY[i]) ok++; } return ok / hX.length;
};

// 1-hidden-layer threshold-GATE model (soft train / hard eval), 2-class
const sig = (z) => 1 / (1 + Math.exp(-z));
const gateAcc = (trX, trY, hX, hY, d) => {
  let rng = 7 >>> 0; const rnd = () => ((rng = (rng * 1664525 + 1013904223) >>> 0) / 4294967296 - 0.5);
  const W = new Float32Array(d * V); for (let k = 0; k < W.length; k++) W[k] = rnd() * WINIT;
  const tG = new Float32Array(d), bG = new Float32Array(d);
  const E = new Float32Array(2 * d); for (let k = 0; k < E.length; k++) E[k] = rnd() * 0.2; const cb = new Float32Array(2);
  const h = new Float32Array(d); const dW = new Float32Array(d * V), vW = new Float32Array(d * V), mE = new Float32Array(2 * d);
  const order = Int32Array.from({ length: trX.length }, (_, i) => i);
  for (let ep = 0; ep < EPOCHS; ep++) {
    for (let i = order.length - 1; i > 0; i--) { const r = Math.floor((((rng = (rng * 1664525 + 1013904223) >>> 0)) / 4294967296) * (i + 1)); const t = order[i]; order[i] = order[r]; order[r] = t; }
    const lr = LR * (1 - 0.9 * ep / EPOCHS);
    for (let bi = 0; bi < order.length; bi += BATCH) {
      const bend = Math.min(order.length, bi + BATCH), nb = bend - bi;
      const dt = new Float32Array(d), db = new Float32Array(d), dE = new Float32Array(2 * d), dcb = new Float32Array(2); const touched = [];
      for (let oi = bi; oi < bend; oi++) {
        const x = trX[order[oi]], y = trY[order[oi]], L = x.length;
        for (let j = 0; j < d; j++) { let acc = 0; const base = j * V; for (let p = 0; p < L; p++) acc += W[base + x[p]]; h[j] = sig(acc + bG[j] - tG[j] * L); }
        let l0 = cb[0], l1 = cb[1]; for (let j = 0; j < d; j++) { l0 += E[j] * h[j]; l1 += E[d + j] * h[j]; }
        const m = Math.max(l0, l1); const e0 = Math.exp(l0 - m), e1 = Math.exp(l1 - m), Z = e0 + e1; const p0 = e0 / Z, p1 = e1 / Z;
        const g0 = p0 - (y === 0 ? 1 : 0), g1 = p1 - (y === 1 ? 1 : 0); dcb[0] += g0; dcb[1] += g1;
        for (let j = 0; j < d; j++) { dE[j] += g0 * h[j]; dE[d + j] += g1 * h[j];
          const dh = g0 * E[j] + g1 * E[d + j]; const dz = dh * h[j] * (1 - h[j]); const base = j * V;
          for (let p = 0; p < L; p++) { const k = base + x[p]; if (dW[k] === 0) touched.push(k); dW[k] += dz; } db[j] += dz; dt[j] -= dz * L; }
      }
      for (const k of touched) { vW[k] = MOM * vW[k] + dW[k] / nb; let v = W[k] - lr * vW[k]; if (v > CLIP) v = CLIP; else if (v < -CLIP) v = -CLIP; W[k] = v; dW[k] = 0; }
      for (let j = 0; j < d; j++) { tG[j] -= lr * dt[j] / nb; bG[j] -= lr * db[j] / nb; }
      for (let k = 0; k < E.length; k++) { mE[k] = MOM * mE[k] + dE[k] / nb; E[k] -= lr * mE[k]; }
      cb[0] -= lr * dcb[0] / nb; cb[1] -= lr * dcb[1] / nb;
    }
  }
  const ev = (hard) => { let ok = 0; for (let i = 0; i < hX.length; i++) { const x = hX[i], L = x.length;
    for (let j = 0; j < d; j++) { let acc = 0; const base = j * V; for (let p = 0; p < L; p++) acc += W[base + x[p]]; const zj = acc + bG[j] - tG[j] * L; h[j] = hard ? (zj > 0 ? 1 : 0) : sig(zj); }
    let l0 = cb[0], l1 = cb[1]; for (let j = 0; j < d; j++) { l0 += E[j] * h[j]; l1 += E[d + j] * h[j]; }
    if ((l1 > l0 ? 1 : 0) === hY[i]) ok++; } return ok / hX.length; };
  return { soft: ev(false), hard: ev(true) };
};

const w = process.stdout;
w.write(`# [PROTOTYPE] nonlinearity diagnostic — linear vs 1-hidden-layer gate model on synthetic cue labels (d=${DGATES})\n`);
for (const kind of ["AND", "XOR"]) {
  const Y = mkLabels(kind); const { trX, trY, hX, hY } = split(Y);
  const lin = linAcc(trX, trY, hX, hY); const g = gateAcc(trX, trY, hX, hY, DGATES);
  w.write(`  ${kind}  base=${(100 * baserate(Y)).toFixed(0)}%  |  linear ${(100 * lin).toFixed(1)}%  |  gate soft ${(100 * g.soft).toFixed(1)}%  hard ${(100 * g.hard).toFixed(1)}%\n`);
}
w.write(`# read: XOR → gate ≫ linear ⇒ nonlinearity IS exploited (real-task tie = data is linearly separable); gate ≈ linear on XOR ⇒ optimizer is the limit.\n`);
