"use strict";

// ============================================================================
// benchmark/multiGate.js — multi-p⁺ via RESIDUAL PEELING (sequential covering of gates).
// Inner object unchanged (3F [L|L1∪L2|C], M_k=¬(M_glob∧p⁺_k), peeled p⁻_k) but built on A_rem.
//   loop: p⁺_k={count_{A_rem}≥ν}; M_k; peel p⁻_k from ¬p⁺_k FPs; sweep→pick t_k;
//         A_cov={i∈A_rem: score_k(y_i)>t_k}; if |A_cov|<s_min stop; A_rem \= A_cov.
// The gate DEFINES its cluster (A_cov = what it fires on) — no partition search, no distance.
// Acceptance: does gate1 fire on finance (swiss/ubs/corporation), gate2 on geo (danube/river/west)?
// Report per gate: |A_cov|,|p⁺|,|p⁻|, FP-rate, and top-20 p⁺ atoms by count in the cluster.
// Usage: TARGET=bank R=5 NU=2 K=30 SMIN=30 node --max-old-space-size=8192 benchmark/multiGate.js english.txt <corpus>
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

const TARGET = process.env.TARGET || "bank";
const R = Number(process.env.R || 5), D = R + 1, DD = Number(process.env.DELTA || 0.001);
const NU = Number(process.env.NU || 2), K = Number(process.env.K || 30), SMIN = Number(process.env.SMIN || 30), TK = Number(process.env.TK || 0.9);

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F, parts = dict.allParts();
  const pc = new Map(); const partsOf = (w) => { let a = pc.get(w); if (!a) { a = [...new Set(encodeWord(dict, w))]; pc.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const ws = tokenizeStream(line.slice(seg.start, seg.end)).filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws);
  }
  const band = (d) => (d === 1 ? 2 : d <= 3 ? 1 : 0);
  const st = new Int32Array(DIM).fill(-1); let sp = 0;
  const feat = (words, t) => { sp++; const y = []; for (let d = 1; d <= D; d++) { const q = t - d; if (q < 0) break; const off = band(d) * F; for (const a of partsOf(words[q])) { const b = a + off; if (st[b] !== sp) { st[b] = sp; y.push(b); } } } return y; };
  const lab = (b) => parts[b % F].value + "@" + ["L", "M", "C"][Math.floor(b / F)];

  const M = new Float64Array(DIM); let N = 0; const POS = []; const neg = []; let ni = 0;
  let negTot = 0; for (const words of sents) for (let t = 1; t < words.length; t++) if (words[t] !== TARGET) negTot++;
  const stride = Math.max(1, Math.floor(negTot / 120000));
  for (const words of sents) for (let t = 1; t < words.length; t++) { const f = feat(words, t); N++; for (const b of f) M[b]++; if (words[t] === TARGET) POS.push(f); else if (ni++ % stride === 0) neg.push(f); }
  const rareBit = (b) => M[b] / N <= DD;

  const w = process.stdout;
  w.write(`# multiGate 3F [L|L1∪L2|C] — ${path.basename(corpusArg)} | TARGET=${TARGET} ν=${NU} K=${K} s_min=${SMIN} t_k=${TK} | |A|=${POS.length} negSample=${neg.length}\n`);

  let Arem = POS.slice(); let k = 0; let covered = 0;
  while (Arem.length >= SMIN && k < 6) {
    k++;
    const cnt = new Float64Array(DIM); for (const f of Arem) for (const b of f) cnt[b]++;
    const pPlus = (b) => cnt[b] >= NU;
    const keep = (b) => rareBit(b) || !pPlus(b);
    // peel p⁻_k: gate-1 FPs (all rare bits ∈ p⁺_k), tally common∉p⁺_k, ≥K
    const cnp = new Map();
    for (const f of neg) { let ok = true; for (const b of f) if (rareBit(b) && !pPlus(b)) { ok = false; break; } if (!ok) continue; for (const b of f) if (!rareBit(b) && !pPlus(b)) cnp.set(b, (cnp.get(b) || 0) + 1); }
    const peel = new Set(); for (const [b, c] of cnp) if (c >= K) peel.add(b);
    const score = (f) => { let posM = 0, ng = 0, kept = 0; for (const b of f) { if (!keep(b)) continue; kept++; if (pPlus(b)) { if (rareBit(b)) posM++; } else if (peel.has(b)) ng++; } return kept > 0 ? (posM - ng) / kept : 0; };
    // cluster = what it fires on
    const cov = [], rem2 = []; for (const f of Arem) (score(f) > TK ? cov : rem2).push(f);
    if (cov.length < SMIN) { w.write(`\n[stop] gate ${k} would cover only ${cov.length} < s_min=${SMIN}\n`); break; }
    let fp = 0; for (const f of neg) if (score(f) > TK) fp++;
    // top-20 p⁺ atoms by count within the cluster
    const cc = new Float64Array(DIM); for (const f of cov) for (const b of f) cc[b]++;
    const atoms = []; for (let b = 0; b < DIM; b++) if (rareBit(b) && pPlus(b) && cc[b] > 0) atoms.push([b, cc[b]]);
    atoms.sort((a, b) => b[1] - a[1]);
    let ppn = 0; for (let b = 0; b < DIM; b++) if (pPlus(b)) ppn++;
    covered += cov.length;
    w.write(`\n──── GATE ${k}  |A_cov|=${cov.length} (cum ${covered}/${POS.length})  |p⁺|=${ppn}  |p⁻|=${peel.size}  FP-rate=${(100 * fp / neg.length).toFixed(3)}%  ────\n`);
    w.write(`   top p⁺ atoms (value@band × count-in-cluster): ${atoms.slice(0, 20).map(([b, c]) => `${lab(b)}(${c | 0})`).join(" ")}\n`);
    Arem = rem2;
  }
  w.write(`\n# ${k} gates; covered ${covered}/${POS.length}; ${Arem.length} residual\n`);
};

main();
