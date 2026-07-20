"use strict";

// benchmark/phase1/learnAtDBench.js — [PROTOTYPE] CAN WE LEARN d GATES END-TO-END, keeping the two-binary-matrix
// inference form?  Each gate fires iff  |W⁺_j∧x| + b_j − |W⁻_j∧x| > t_j·|x|  — TWO integer popcounts at inference
// (W⁺,W⁻ binary; t,b scalars).  TRAIN SOFT / DEPLOY HARD: train float gate weights W with a sigmoid activation so
// gradients flow (a pure-hard STE forward dead-zones — nothing crosses the ternary threshold at cold start); EVAL
// with the hard binary-activation gate  h=step(W·x + b − t|x| > 0)  — the deployable form.  The d-bit code feeds a
// linear readout over N target words (next-token classification).
// Question: does the learned d-gate layer DISCRIMINATE (scale with d, pass the linear per-class SLOG baseline)?
//   node --max-old-space-size=8192 benchmark/phase1/learnAtDBench.js <dict> <corpus>

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

const TARGETS = (process.env.TARGETS ||
  "time,world,state,music,century,river,government,bank,physics,philosophy,hydrogen,water,energy,system,number,language," +
  "history,science,city,country,war,book,film,company,church,school,family,group,party,market,law,art,king,president," +
  "university,population,building,region,species,culture").split(",");
const R = Number(process.env.R || 5), D = R + 1;
const PERCLASS = Number(process.env.PERCLASS || 1200);
const MINBITFREQ = Number(process.env.MINBITFREQ || 3);
const DLIST = (process.env.DLIST || "16,64,256").split(",").map(Number);
const EPOCHS = Number(process.env.EPOCHS || 12), LR = Number(process.env.LR || 0.1), BATCH = Number(process.env.BATCH || 64);
const CLIP = Number(process.env.CLIP || 1.0);
const WINIT = Number(process.env.WINIT || 0.05), MOM = Number(process.env.MOM || 0.9);
const SLOGSCALE = Number(process.env.SLOGSCALE || 0.3);

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

const tset = new Set(TARGETS), classA = new Map(TARGETS.map((t) => [t, []]));
for (const s of sents) for (let t = 1; t < s.length; t++) if (tset.has(s[t])) { const A = classA.get(s[t]); if (A.length < 4 * PERCLASS) A.push(feat(s, t)); }
const CLASSES = TARGETS.filter((w) => (classA.get(w) || []).length >= 80);
const N = CLASSES.length;

const samp = (arr, n) => { if (arr.length <= n) return arr.slice(); const step = arr.length / n, out = []; for (let i = 0; i < arr.length; i += step) out.push(arr[Math.floor(i)]); return out; };
const trainX = [], trainY = [], heldX = [], heldY = [];
CLASSES.forEach((w, c) => { const A = samp(classA.get(w), PERCLASS); const cut = Math.floor(A.length * 0.8);
  for (let i = 0; i < A.length; i++) (i < cut ? (trainX.push(A[i]), trainY.push(c)) : (heldX.push(A[i]), heldY.push(c))); });

const bf = new Map(); for (const x of trainX) for (const b of x) bf.set(b, (bf.get(b) || 0) + 1);
const idx = new Map(); for (const [b, c] of bf) if (c >= MINBITFREQ) idx.set(b, idx.size);
const V = idx.size;
const enc = (x) => { const o = []; for (const b of x) { const j = idx.get(b); if (j !== undefined) o.push(j); } return Int32Array.from(o); };
const TX = trainX.map(enc), HX = heldX.map(enc);
process.stderr.write(`N=${N} classes | train=${TX.length} held=${HX.length} | learnable bits V=${V}\n`);

// ---- baseline: discovered per-class SLOG gates (N one-vs-all log-odds), argmax = "discover, don't learn" ----
const globalRate = new Float64Array(V); for (const x of TX) for (const j of x) globalRate[j]++; for (let j = 0; j < V; j++) globalRate[j] /= TX.length;
const AL = 1 / TX.length;
const U = [];
for (let c = 0; c < N; c++) { const cnt = new Float64Array(V); let n = 0; for (let i = 0; i < TX.length; i++) if (trainY[i] === c) { n++; for (const j of TX[i]) cnt[j]++; }
  const u = new Float64Array(V); for (let j = 0; j < V; j++) u[j] = Math.log((cnt[j] / (n || 1) + AL) / (globalRate[j] + AL)); U.push(u); }
const slogAcc = (() => { let ok = 0; for (let i = 0; i < HX.length; i++) { let best = -Infinity, arg = 0; for (let c = 0; c < N; c++) { let s = 0; for (const j of HX[i]) s += U[c][j]; if (s > best) { best = s; arg = c; } } if (arg === heldY[i]) ok++; } return ok / HX.length; })();

// ---- learn d gates: soft train (float W + sigmoid), hard eval (step gate = deployable two-matrix form) ----
const sig = (z) => 1 / (1 + Math.exp(-z));
function trainD(d, seed, opts = {}) {
  const { initSlog = false, freeze = false, identityReadout = false } = opts;
  let rng = seed >>> 0; const rnd = () => ((rng = (rng * 1664525 + 1013904223) >>> 0) / 4294967296 - 0.5);
  const W = new Float32Array(d * V);
  if (initSlog) { for (let j = 0; j < d; j++) { const u = U[j % N], base = j * V; for (let k = 0; k < V; k++) { let v = u[k] * SLOGSCALE; if (v > CLIP) v = CLIP; else if (v < -CLIP) v = -CLIP; W[base + k] = v; } } }
  else { for (let k = 0; k < W.length; k++) W[k] = rnd() * WINIT; }
  const tG = new Float32Array(d), bG = new Float32Array(d);
  const E = new Float32Array(N * d);
  if (identityReadout) { for (let k = 0; k < E.length; k++) E[k] = rnd() * 0.05; for (let j = 0; j < d; j++) E[(j % N) * d + j] += 1; }
  else { for (let k = 0; k < E.length; k++) E[k] = rnd() * 0.2; }
  const cb = new Float32Array(N);
  if (freeze) { const sx = TX.slice(0, 2000); for (let j = 0; j < d; j++) { const base = j * V; const rs = sx.map((x) => { let a = 0; for (const p of x) a += W[base + p]; return a / x.length; }).sort((a, b) => a - b); tG[j] = rs[rs.length >> 1]; } }
  const h = new Float32Array(d), z = new Float32Array(d), logit = new Float32Array(N);
  const dW = new Float32Array(d * V), vW = new Float32Array(d * V), mE = new Float32Array(N * d);
  const order = Int32Array.from({ length: TX.length }, (_, i) => i);
  for (let ep = 0; ep < EPOCHS; ep++) {
    for (let i = order.length - 1; i > 0; i--) { const r = Math.floor((((rng = (rng * 1664525 + 1013904223) >>> 0)) / 4294967296) * (i + 1)); const t = order[i]; order[i] = order[r]; order[r] = t; }
    const lr = LR * (1 - 0.9 * ep / EPOCHS);
    for (let bi = 0; bi < order.length; bi += BATCH) {
      const bend = Math.min(order.length, bi + BATCH), nb = bend - bi;
      const dt = new Float32Array(d), db = new Float32Array(d), dE = new Float32Array(N * d), dcb = new Float32Array(N); const touched = [];
      for (let oi = bi; oi < bend; oi++) {
        const ex = order[oi], x = TX[ex], y = trainY[ex], L = x.length;
        for (let j = 0; j < d; j++) { let acc = 0; const base = j * V; for (let p = 0; p < L; p++) acc += W[base + x[p]]; const zj = acc + bG[j] - tG[j] * L; z[j] = zj; h[j] = sig(zj); }
        let mx = -Infinity; for (let c = 0; c < N; c++) { let s = cb[c]; const eb = c * d; for (let j = 0; j < d; j++) s += E[eb + j] * h[j]; logit[c] = s; if (s > mx) mx = s; }
        let Z = 0; for (let c = 0; c < N; c++) { logit[c] = Math.exp(logit[c] - mx); Z += logit[c]; }
        for (let c = 0; c < N; c++) { const g = logit[c] / Z - (c === y ? 1 : 0); dcb[c] += g; const eb = c * d; for (let j = 0; j < d; j++) dE[eb + j] += g * h[j]; }
        if (!freeze) for (let j = 0; j < d; j++) { let dh = 0; for (let c = 0; c < N; c++) dh += (logit[c] / Z - (c === y ? 1 : 0)) * E[c * d + j];
          const dz = dh * h[j] * (1 - h[j]); const base = j * V;
          for (let p = 0; p < L; p++) { const k = base + x[p]; if (dW[k] === 0) touched.push(k); dW[k] += dz; } db[j] += dz; dt[j] -= dz * L; }
      }
      if (!freeze) { for (const k of touched) { vW[k] = MOM * vW[k] + dW[k] / nb; let v = W[k] - lr * vW[k]; if (v > CLIP) v = CLIP; else if (v < -CLIP) v = -CLIP; W[k] = v; dW[k] = 0; }
        for (let j = 0; j < d; j++) { tG[j] -= lr * dt[j] / nb; bG[j] -= lr * db[j] / nb; } }
      for (let k = 0; k < E.length; k++) { mE[k] = MOM * mE[k] + dE[k] / nb; E[k] -= lr * mE[k]; }
      for (let c = 0; c < N; c++) cb[c] -= lr * dcb[c] / nb;
    }
  }
  const evalMode = (hard) => { let ok = 0; for (let i = 0; i < HX.length; i++) { const x = HX[i], L = x.length;
    for (let j = 0; j < d; j++) { let acc = 0; const base = j * V; for (let p = 0; p < L; p++) acc += W[base + x[p]]; const zj = acc + bG[j] - tG[j] * L; h[j] = hard ? (zj > 0 ? 1 : 0) : sig(zj); }
    let best = -Infinity, arg = 0; for (let c = 0; c < N; c++) { let s = cb[c]; const eb = c * d; for (let j = 0; j < d; j++) s += E[eb + j] * h[j]; if (s > best) { best = s; arg = c; } }
    if (arg === heldY[i]) ok++; } return ok / HX.length; };
  let nz = 0; for (let k = 0; k < W.length; k++) if (Math.abs(W[k]) > 0.3) nz++;
  return { soft: evalMode(false), hard: evalMode(true), wpGate: nz / d };
}

const w = process.stdout;
w.write(`# [PROTOTYPE] gate-training methods vs the counted baseline | ${N} target words | N-way next-token held acc\n`);
w.write(`# random-guess=${(100 / N).toFixed(1)}%  ·  SLOG-linear argmax (counted, no training) = ${(100 * slogAcc).toFixed(1)}%\n`);
w.write(`  method                         d     soft-acc  hard-acc  vs-SLOG  |W|>.3/gate\n`);
const row = (name, d, r) => w.write(`  ${name.padEnd(28)}  ${String(d).padStart(3)}    ${(100 * r.soft).toFixed(1).padStart(5)}%   ${(100 * r.hard).toFixed(1).padStart(5)}%   ${((100 * r.hard / slogAcc) - 100 >= 0 ? "+" : "") + ((100 * r.hard / slogAcc) - 100).toFixed(0).padStart(3)}%     ${r.wpGate.toFixed(0)}\n`);
// Q1: SLOG-init GD — starts AT the counted solution (d=N, identity readout); does gradient climb past it?
row("SLOG-init + GD (Q1)", N, trainD(N, 999, { initSlog: true, identityReadout: true }));
// Q3: frozen random gates + convex readout (ELM) — threshold nonlinearity for free, no gate training
row("random-frozen ELM (Q3)", 256, trainD(256, 999, { freeze: true }));
// reference: plain random-init GD
row("random-init + GD", 256, trainD(256, 12345 + 256, {}));
w.write(`# read: SLOG-init>SLOG ⇒ GD earns its cost; ELM≈SLOG ⇒ threshold nonlinearity is ~free; both≈SLOG ⇒ counting suffices at this scale.\n`);
