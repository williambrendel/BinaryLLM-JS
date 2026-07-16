"use strict";

// ============================================================================
// benchmark/boostGates.js — MINIMAL end-to-end: gates as weak learners, boosted by residual-Gain.
// Validates the unification's two load-bearing claims:
//   (1) committing gates by Gain(S) drives train NLL below unigram
//   (2) redundant gates self-deduplicate — the twin scores Gain≈0 after the first commits
// Setup: 3F [L|L1∪L2|C] sparse; classes = top-C next-words + UNK; one positive-only gate per target class
//   (p⁻=∅ here — precision refinement, orthogonal to the boosting/dedup mechanism).
//   Gain(S)=Σ_c (Σ_S r_i[c])²/(Σ_S h_i[c]+γ);  w_g[c]=η·Σr/(Σh+γ);  commit argmax Gain>λ.
// Usage: C=30 NTR=20000 node --max-old-space-size=8192 benchmark/boostGates.js english.txt <corpus>
// ============================================================================

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

const R = Number(process.env.R || 5), D = R + 1, DD = Number(process.env.DELTA || 0.001);
const C = Number(process.env.C || 30), NTR = Number(process.env.NTR || 20000);
const GAMMA = 1.0, ETA = Number(process.env.ETA || 0.3), LAMBDA = Number(process.env.LAM || 1.0), TG = Number(process.env.TG || 0.5), WCLIP = Number(process.env.WCLIP || 2.0);
const clip = (x) => (x > WCLIP ? WCLIP : x < -WCLIP ? -WCLIP : x);

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F;
  const pc = new Map(); const partsOf = (w) => { let a = pc.get(w); if (!a) { a = [...new Set(encodeWord(dict, w))]; pc.set(w, a); } return a; };
  const words = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const ws = tokenizeStream(line.slice(seg.start, seg.end)).filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) words.push(ws);
  }
  const band = (d) => (d === 1 ? 2 : d <= 3 ? 1 : 0);
  const stmp = new Int32Array(DIM).fill(-1); let sp = 0;
  const feat = (s, t) => { sp++; const y = []; for (let d = 1; d <= D; d++) { const q = t - d; if (q < 0) break; const off = band(d) * F; for (const a of partsOf(s[q])) { const b = a + off; if (stmp[b] !== sp) { stmp[b] = sp; y.push(b); } } } return y; };

  // vocabulary: top-C next-words → classes 0..C-1, else UNK=C
  const wc = new Map(); for (const s of words) for (let t = 1; t < s.length; t++) wc.set(s[t], (wc.get(s[t]) || 0) + 1);
  const top = [...wc.entries()].sort((a, b) => b[1] - a[1]).slice(0, C).map((e) => e[0]);
  const cls = new Map(top.map((w, i) => [w, i])); const NC = C + 1; const classOf = (w) => (cls.has(w) ? cls.get(w) : C);

  // build positions (feat, class), global M, split train
  const M = new Float64Array(DIM); const pos = [];
  for (const s of words) for (let t = 1; t < s.length; t++) { const f = feat(s, t); for (const b of f) M[b]++; pos.push({ f, c: classOf(s[t]) }); }
  const Ntot = pos.length; const rare = (b) => M[b] / Ntot <= DD;
  const train = pos.slice(0, Math.min(NTR, pos.length)); const Ntr = train.length;

  // one positive-only gate per target class: p⁺=rare∩support(A_c), M_c=¬(common∧support); fire iff posM/kept>TG
  const gates = [];
  for (let c = 0; c < C; c++) {
    const sup = new Uint8Array(DIM); for (const p of train) if (p.c === c) for (const b of p.f) sup[b] = 1;
    const pPlus = (b) => rare(b) && sup[b];      // rare positive bits
    const keep = (b) => rare(b) || !sup[b];      // M_c = ¬(common ∧ support)
    const S = []; for (let i = 0; i < Ntr; i++) { let posM = 0, kept = 0; for (const b of train[i].f) { if (!keep(b)) continue; kept++; if (pPlus(b)) posM++; } if (kept > 0 && posM / kept > TG) S.push(i); }
    if (S.length >= 20) gates.push({ c, S, w: new Float64Array(NC), used: false });
  }
  // REDUNDANCY INJECTION: a PROBE duplicate (not a candidate) of a MID-size gate that fits within the clip
  const byS = gates.map((g, i) => [i, g.S.length]).sort((a, b) => a[1] - b[1]);
  const twinIdx = byS[Math.floor(byS.length * 0.5)][0];
  const dup = { c: gates[twinIdx].c, S: gates[twinIdx].S.slice(), w: new Float64Array(NC), used: false, isDup: true };

  // boosting: b = log unigram; logits[i][.]; commit gates by Gain
  const cnt = new Float64Array(NC); for (const p of train) cnt[p.c]++;
  const bvec = new Float64Array(NC); for (let c = 0; c < NC; c++) bvec[c] = Math.log((cnt[c] + 1e-9) / Ntr);
  const logit = Array.from({ length: Ntr }, () => Float64Array.from(bvec));
  const P = Array.from({ length: Ntr }, () => new Float64Array(NC));
  const recomputeAt = (i) => { const lg = logit[i]; let mx = -1e9; for (let c = 0; c < NC; c++) if (lg[c] > mx) mx = lg[c]; let Z = 0; for (let c = 0; c < NC; c++) { P[i][c] = Math.exp(lg[c] - mx); Z += P[i][c]; } for (let c = 0; c < NC; c++) P[i][c] /= Z; };
  const recompute = () => { let ll = 0; for (let i = 0; i < Ntr; i++) { recomputeAt(i); ll += -Math.log(P[i][train[i].c] + 1e-12); } return ll / Ntr; };
  // JOINT REFIT: coordinate descent — re-fit each committed gate's w to the closed-form optimum given the others
  const jointRefit = (sweeps) => { for (let s = 0; s < sweeps; s++) for (const g of gates) if (g.used) { for (const i of g.S) { for (let c = 0; c < NC; c++) logit[i][c] -= g.w[c]; recomputeAt(i); } for (let c = 0; c < NC; c++) { let sr = 0, sh = 0; for (const i of g.S) { const p = P[i][c]; sr += (c === train[i].c ? 1 : 0) - p; sh += p * (1 - p); } g.w[c] = clip(sr / (sh + GAMMA)); } for (const i of g.S) { for (let c = 0; c < NC; c++) logit[i][c] += g.w[c]; recomputeAt(i); } } };
  const ll0 = recompute();

  const w = process.stdout;
  w.write(`# boostGates — ${path.basename(corpusArg)} | C=${C} classes NC=${NC} | Ntr=${Ntr} | gates proposed=${gates.length}\n`);
  w.write(`# unigram train NLL = ${ll0.toFixed(4)}\n\n`);

  const gainOf = (g) => { let G = 0; for (let c = 0; c < NC; c++) { let sr = 0, sh = 0; for (const i of g.S) { const p = P[i][c]; sr += (c === train[i].c ? 1 : 0) - p; sh += p * (1 - p); } G += (sr * sr) / (sh + GAMMA); } return G; };
  const dupGain0 = gainOf(dup);
  let committed = 0;
  for (let iter = 1; iter <= gates.length; iter++) {
    recompute();
    let best = null, bg = LAMBDA; for (const g of gates) if (!g.used && !g.isDup) { const G = gainOf(g); if (G > bg) { bg = G; best = g; } }
    if (!best) { w.write(`# stop: no candidate Gain > λ=${LAMBDA}\n`); break; }
    for (let c = 0; c < NC; c++) { let sr = 0, sh = 0; for (const i of best.S) { const p = P[i][c]; sr += (c === train[i].c ? 1 : 0) - p; sh += p * (1 - p); } best.w[c] = clip(ETA * sr / (sh + GAMMA)); }
    for (const i of best.S) for (let c = 0; c < NC; c++) logit[i][c] += best.w[c];
    best.used = true; committed++;
    if (committed % 10 === 0) jointRefit(3);   // joint refit every 10 commits
    const ll = recompute();
    if (committed <= 6 || committed % 5 === 0 || best.isDup) w.write(`  commit ${committed}: gate→'${top[best.c]}'${best.isDup ? " [DUP]" : ""}  |S|=${best.S.length}  Gain=${bg.toFixed(1)}  trainNLL=${ll.toFixed(4)}\n`);
  }
  jointRefit(6); const llF = recompute();
  w.write(`\n# committed ${committed}/${gates.length} gates. unigram ${ll0.toFixed(4)} → boosted ${llF.toFixed(4)}  (Δ=${(ll0 - llF).toFixed(4)} nats)\n`);
  // dedup check: committed gates should have Gain≈0 under the jointly-refit model
  const wl2 = (g) => Math.sqrt(g.w.reduce((a, x) => a + x * x, 0));
  w.write(`\n# DEDUP CHECK — probe = exact copy of committed gate '${top[dup.c]}' (|S|=${dup.S.length}):\n`);
  w.write(`  probe Gain BEFORE any commit : ${dupGain0.toFixed(1)}\n`);
  w.write(`  probe Gain AFTER twin committed + refit : ${gainOf(dup).toFixed(2)}   (reduction ${(100 * (1 - gainOf(dup) / dupGain0)).toFixed(1)}%)\n`);
  w.write(`  twin committed=${gates[twinIdx].used}  ‖w_twin‖=${wl2(gates[twinIdx]).toFixed(3)} (clip ceiling ${(WCLIP * Math.sqrt(NC)).toFixed(1)})  → if probe Gain ≈0, the residual deduplicated as designed\n`);
};

main();
