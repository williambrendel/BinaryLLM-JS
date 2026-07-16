"use strict";

// benchmark/discover.js — real-corpus run of the multi-part discovery trainer (§6/§9).
// Reuses canon.js's 3F [L | L1∪L2 | C] featurization to build target contexts, then runs
// 6b (K-sweep), 6a, and the 6c baselines through the §9 acceptance harness. Emits the
// §9.5 CSV and a readable summary. Deterministic (fixed-stride sampling, seeded baselines).
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/discover.js english.txt <corpus>

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { encodeWord } from "../src/core/signatures/wordEncoder.js";
import { fuzzyEncode } from "../src/core/parts/encoding/fuzzyEncode.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";
import buildHardNegatives from "../src/core/discovery/hardNegatives.js";
import discoverSeq from "../src/core/discovery/discoverSeq.js";
import fitParallel from "../src/core/discovery/fitParallel.js";
import { baselinePerExample, baselineRandomChunk, baselineRandomChunkSA } from "../src/core/discovery/baselines.js";
import fitSA from "../src/core/discovery/fitSA.js";
import { acceptanceRow, toCSV } from "../src/core/discovery/acceptance.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGET = process.env.TARGET || "bank";
const R = Number(process.env.R || 5), D = R + 1, DD = Number(process.env.DELTA || 0.001);
const NEGSAMPLE = Number(process.env.NEGSAMPLE || 40000);
const KMAX = Number(process.env.KMAX || 6);

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F;
  const REP = process.env.REP || "fuzzy";                        // fuzzy (default, gain mass) | sparse (ablation)
  const enc = REP === "sparse" ? encodeWord : fuzzyEncode;
  const pc = new Map(); const partsOf = (w) => { let a = pc.get(w); if (!a) { a = [...new Set(enc(dict, w))]; pc.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const ws = tokenizeStream(line.slice(seg.start, seg.end)).filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws);
  }
  const band = (dd) => (dd === 1 ? 2 : dd <= 3 ? 1 : 0);
  const stmp = new Int32Array(DIM).fill(-1); let sp = 0;
  const feat = (s, t) => { sp++; const y = []; for (let dd = 1; dd <= D; dd++) { const q = t - dd; if (q < 0) break; const off = band(dd) * F; for (const a of partsOf(s[q])) { const b = a + off; if (stmp[b] !== sp) { stmp[b] = sp; y.push(b); } } } return y.sort((a, b) => a - b); };

  const half = sents.length >> 1;
  const M = new Float64Array(DIM); let Ntot = 0, negTot = 0;
  const Atrain = [], Aheld = [], negS = []; let ni = 0;
  for (let si = 0; si < sents.length; si++) {
    const s = sents[si], isTrain = si < half;
    for (let t = 1; t < s.length; t++) {
      const f = feat(s, t); Ntot++; for (const b of f) M[b]++;
      if (s[t] === TARGET) (isTrain ? Atrain : Aheld).push(f);
      else { negTot++; if (ni++ % Math.max(1, Math.floor((sents.length * 8) / NEGSAMPLE)) === 0) negS.push(f); }
    }
  }
  // M_glob = common bits (freq > δ): the union of bits composing common words.
  const Mglob = new Set(); for (let b = 0; b < DIM; b++) if (M[b] / Ntot > DD) Mglob.add(b);

  const A = [...Atrain, ...Aheld];
  const MINHARD = Number(process.env.MINHARD || 2000);
  const { Dtau, AbarH, t0, scoreG0 } = buildHardNegatives(Atrain, negS, Mglob, { minSize: MINHARD });
  const w = process.stdout;
  w.write(`# discover — ${path.basename(corpusArg)} | TARGET=${TARGET} | trainPos=${Atrain.length} heldPos=${Aheld.length} negSample=${negS.length} negTot=${negTot}\n`);
  w.write(`# |M_glob|=${Mglob.size} |D_τ|=${Dtau.length} |Ā_h|=${AbarH.length} (of ${negS.length}) | DIM=${DIM}\n\n`);

  const KAPPA = Number(process.env.KAPPA || 2);
  const cfg = (K, lambda) => ({ K, m: 2, kappa: KAPPA, representation: `${REP}3F`, lambda });
  const O = { m: 2, kappa: KAPPA, sMin: 5 };

  // eligibility diagnostic: how many bits are enriched (d>κ) in Atrain vs Ā_h?
  {
    const hp = new Map(), hn = new Map();
    for (const y of Atrain) for (const b of y) hp.set(b, (hp.get(b) || 0) + 1);
    for (const y of AbarH) for (const b of y) hn.set(b, (hn.get(b) || 0) + 1);
    const nP = Atrain.length, nN = AbarH.length, eps = 1 / (nN + 1);
    let elig = 0, maxd = 0; const Dset = new Set(Dtau);
    for (const [b, cp] of hp) { if (Dset.has(b) || cp < 2) continue; const d = (cp / nP + eps) / ((hn.get(b) || 0) / nN + eps); if (d > KAPPA) elig++; if (d > maxd) maxd = d; }
    w.write(`# eligibility (κ=${KAPPA}): ${elig} bits with d>κ (of ${hp.size} seen in Atrain, support≥2); max d=${maxd.toFixed(1)}\n\n`);
  }
  const rows = [];
  const push = (model, K, lambda) => { const r = acceptanceRow(model, Atrain, Aheld, negS, cfg(K, lambda), AbarH); rows.push(r); return r; };

  // g0 baseline row (single coarse gate)
  const g0model = { Dtau, gates: [{ Q: [...new Set(Atrain.flat())].filter((b) => !Dtau.includes(b)).sort((a, b) => a - b), pMinus: [], t: t0 }], algo: "g0" };
  push(g0model, 1, "");

  // broad FP-control pool (sampled) + coverage floor — the real-run recipe
  const thrNeg = negS.filter((_, j) => j % Math.max(1, Math.floor(negS.length / 6000)) === 0);
  const MINCOVER = Number(process.env.MINCOVER || 5);
  // 6b parallel — K sweep (Heaps: K scales with |A|; re-seed on by default)
  const KLIST = (process.env.KLIST || "5,10,20").split(",").map(Number).filter((k) => k <= KMAX);
  for (const K of KLIST) { const m = { ...fitParallel(Atrain, AbarH, Dtau, K, O), algo: "parallel" }; push(m, K, ""); }
  // 6a sequential — λ sweep (thrNeg controls FP; minCover stops the runaway)
  for (const lambda of [0.005, 0.002, 0.001]) { const m = { ...discoverSeq(Atrain, AbarH, Dtau, { ...O, lambda, maxK: 400, thrNeg, minCover: MINCOVER }), algo: "seq" }; push(m, m.gates.length, lambda); }
  // 6d SA-over-gain (a method, not a baseline)
  push({ ...fitSA(Atrain, AbarH, Dtau, 10, { ...O, iters: 2000 }), algo: "sa_gain" }, 10, "");
  // 6c baselines — the floor discovery must beat
  push({ ...baselinePerExample(Atrain, AbarH, Dtau, O), algo: "per_example" }, Atrain.length, "");
  for (const c of [1, 2]) push({ ...baselineRandomChunk(Atrain, AbarH, Dtau, 10, { ...O, c, seed: 7, shareBudget: 60 }), algo: `random_chunk_c${c}` }, 10, "");
  push({ ...baselineRandomChunkSA(Atrain, AbarH, Dtau, 10, { ...O, c: 2, seed: 7, shareBudget: 60, iters: 3000 }), algo: "random_chunk_sa" }, 10, "");

  // summary table — full metric board (8 metrics + coherence diagnostics)
  w.write(`\n  algo             K  | train-rec train-prec tr-FP% | held-rec held-prec held-FP% | ρ    ovl   |p⁻|  contain\n`);
  for (const r of rows) w.write(`  ${r.algo.padEnd(16)} ${String(r.K).padStart(3)} | ${(100 * r.train_recall).toFixed(1).padStart(5)}%   ${(100 * r.train_precision).toFixed(2).padStart(6)}%  ${(100 * r.train_FP).toFixed(2).padStart(5)}% | ${(100 * r.held_recall).toFixed(1).padStart(5)}%   ${(100 * r.held_precision).toFixed(2).padStart(6)}%  ${(100 * r.held_FP).toFixed(2).padStart(5)}% | ${r.rho.toFixed(2)} ${r.overlap.toFixed(2).padStart(4)} ${r.pMinus_mean.toFixed(0).padStart(4)}   ${r.containment_ok}\n`);

  const out = path.join(REPO, "benchmark", `discover_${TARGET}.csv`);
  fs.writeFileSync(out, toCSV(rows));
  w.write(`\n# CSV → ${path.relative(REPO, out)}  (${rows.length} rows)\n`);
};

main();
