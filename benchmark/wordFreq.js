"use strict";

// ============================================================================
// benchmark/wordFreq.js — WORD-level atoms (each distinct word = one bit), frequency
// computed per WORD, not per word-part. Same [L|C] concat structure & masked-coverage
// detector as bandConcat, so the numbers are directly comparable. Tests whether word-level
// frequency fixes the "rare word peels into common fragments" (masonry) problem and the
// 1-rare-bit false-positive corner.
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/wordFreq.js <corpus>
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";

const TARGET = process.env.TARGET || "bank";
const R = Number(process.env.R || 5);

const main = () => {
  const [corpusArg] = process.argv.slice(2);
  // vocab: word -> id
  const vocab = new Map();
  const idOf = (w) => { let i = vocab.get(w); if (i === undefined) { i = vocab.size; vocab.set(w, i); } return i; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const toks = tokenizeStream(line.slice(seg.start, seg.end));
    const ws = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws.map((x) => ({ w: x, id: idOf(x) })));
  }
  const V = vocab.size, DIM = 2 * V;

  // feature for predicting word[t]: L-union band [0,V) = OR of words[t-1-R..t-1]; C band [V,2V) = word[t-1]
  const stamp = new Int32Array(DIM).fill(-1); let pid = 0;
  const feat = (s, t) => { pid++; const y = []; for (let q = Math.max(0, t - 1 - R); q <= t - 1; q++) { const id = s[q].id; if (stamp[id] !== pid) { stamp[id] = pid; y.push(id); } } const c = s[t - 1].id + V; if (stamp[c] !== pid) { stamp[c] = pid; y.push(c); } return y; };

  let negTot = 0; for (const s of sents) for (let t = 1; t < s.length; t++) if (s[t].w !== TARGET) negTot++;
  const step = Math.max(1, Math.floor(negTot / 3000));
  const POS = [], NEG = []; let ni = 0;
  for (const s of sents) for (let t = 1; t < s.length; t++) { const f = feat(s, t); if (s[t].w === TARGET) POS.push(f); else { if (ni % step === 0) NEG.push(f); ni++; } }

  const pAll = new Uint8Array(DIM); let pn = 0;
  for (const f of POS) for (const a of f) if (!pAll[a]) { pAll[a] = 1; pn++; }
  const M = new Float64Array(DIM); let Ntot = 0;
  for (const s of sents) for (let t = 1; t < s.length; t++) { const f = feat(s, t); Ntot++; for (const a of f) M[a]++; }

  const covMasked = (SET, d, P) => { let s = 0, s2 = 0, n = 0, empty = 0; for (const y of SET) { let yr = 0, k = 0; for (const a of y) if (M[a] / Ntot <= d) { yr++; if (P[a]) k++; } if (yr === 0) { empty++; continue; } const c = k / yr; s += c; s2 += c * c; n++; } const m = n ? s / n : 0; return { m, sd: n ? Math.sqrt(Math.max(0, s2 / n - m * m)) : 0, empty, tot: n + empty }; };

  const w = process.stdout;
  w.write(`# wordFreq [L|C] word-level — ${path.basename(corpusArg)} | TARGET=${TARGET} | V=${V} DIM=${DIM}\n`);
  w.write(`# POS n=${POS.length}  NEG n=${NEG.length}   |p|=${pn}   mean|y| pos=${(POS.reduce((a, f) => a + f.length, 0) / POS.length).toFixed(1)}\n`);

  const DELS = [Infinity, 0.01, 0.005, 0.002, 0.001];
  w.write(`\n── masked coverage (in-sample) ──\n   δ       |p'|   cov pos   cov neg (rare)     ratio\n`);
  for (const d of DELS) { let pp = 0; for (let a = 0; a < DIM; a++) if (pAll[a] && M[a] / Ntot <= d) pp++; const cp = covMasked(POS, d, pAll), cn = covMasked(NEG, d, pAll); w.write(`   ${(d === Infinity ? "∞" : d).toString().padEnd(6)} ${String(pp).padStart(6)}   ${cp.m.toFixed(4)}   ${cn.m.toFixed(4)} ± ${cn.sd.toFixed(4)}   ${(cp.m / (cn.m || 1e-9)).toFixed(2)}\n`); }

  const pHO = new Uint8Array(DIM); const Pb = POS.filter((_, i) => i % 2 === 0), Pt = POS.filter((_, i) => i % 2 === 1);
  for (const f of Pb) for (const a of f) pHO[a] = 1;
  w.write(`\n── HELD-OUT (p from ${Pb.length} pos, tested on ${Pt.length} pos + neg) ──\n   δ       cov pos-HO   %∅   cov neg      %∅   ratio\n`);
  for (const d of DELS) { const cp = covMasked(Pt, d, pHO), cn = covMasked(NEG, d, pHO); w.write(`   ${(d === Infinity ? "∞" : d).toString().padEnd(6)}  ${cp.m.toFixed(4)}   ${(100 * cp.empty / cp.tot).toFixed(0).padStart(3)}%  ${cn.m.toFixed(4)}   ${(100 * cn.empty / cn.tot).toFixed(0).padStart(3)}%   ${(cp.m / (cn.m || 1e-9)).toFixed(2)}\n`); }

  // masked perfect-coverage detector + sample FPs
  const DD = 0.001;
  const inv = new Array(V); for (const [word, i] of vocab) inv[i] = word;
  let mFired = 0, mTP = 0, mUndef = 0, mN = 0; const fpEx = [];
  for (const s of sents) for (let t = 1; t < s.length; t++) {
    const f = feat(s, t); mN++; let yr = 0, allIn = true, ry = [];
    for (const a of f) if (M[a] / Ntot <= DD) { yr++; ry.push(a); if (!pAll[a]) { allIn = false; break; } }
    if (yr === 0) { mUndef++; continue; }
    if (allIn) { mFired++; if (s[t].w === TARGET) mTP++; else if (fpEx.length < 12 && t % 7 === 0) { const ctx = []; for (let q = Math.max(0, t - 1 - R); q <= t - 1; q++) ctx.push(s[q].w); fpEx.push({ ctx: ctx.join(" "), tgt: s[t].w, yr, matched: ry.map((a) => (a < V ? inv[a] + "@L" : inv[a - V] + "@C")) }); } }
  }
  w.write(`\n── MASKED perfect-coverage detector (δ=${DD}, word-level) N=${mN} ──\n`);
  w.write(`  fired=${mFired} (${(100 * mFired / mN).toFixed(3)}%)  TP=${mTP}/${POS.length} (recall ${(100 * mTP / POS.length).toFixed(1)}%)  FALSE POSITIVES=${mFired - mTP}  precision=${(100 * mTP / mFired).toFixed(3)}%  undefined=${(100 * mUndef / mN).toFixed(1)}%\n`);
  for (const e of fpEx) w.write(`  {${e.ctx}} → [${e.tgt}]  (${e.yr} rare word${e.yr > 1 ? "s" : ""}: ${e.matched.join(" ")})\n`);
};

main();
