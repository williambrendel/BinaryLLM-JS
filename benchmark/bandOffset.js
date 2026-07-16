"use strict";

// ============================================================================
// benchmark/bandOffset.js — POSITIONALLY RESOLVED left context: C | L1 | L2 | ... | LR,
// each preceding word in its OWN namespace band (offset d → [(d-1)F, dF)); NO OR across
// positions ("concatenate, never union", applied WITHIN the left context). D=R+1 bands.
// Compares to the 2-band concat: does resolving L by position lift the masked pos/neg ratio
// and the per-band purity? Metrics: masked coverage (in-sample + held-out), per-offset purity.
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/bandOffset.js english.txt <corpus>
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
const R = Number(process.env.R || 5);
const D = R + 1;                    // offsets 1..D  (d=1 is C, the immediate predecessor)
const NEG_N = Number(process.env.NEG || 3000);

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size(), DIM = D * F;
  const cache = new Map();
  const atomsOf = (w) => { let a = cache.get(w); if (!a) { a = Int32Array.from(new Set(encodeWord(dict, w))); cache.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const toks = tokenizeStream(line.slice(seg.start, seg.end));
    const ws = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws.map((x) => ({ w: x, atoms: atomsOf(x) })));
  }

  // feature for predicting word[t]: offset d=1..D → word[t-d], ids +(d-1)*F  (no cross-position OR)
  const feat = (s, t) => { const y = []; for (let d = 1; d <= D; d++) { const q = t - d; if (q < 0) break; const off = (d - 1) * F; for (const a of s[q].atoms) y.push(a + off); } return y; };

  let posTot = 0, negTot = 0;
  for (const s of sents) for (let t = 1; t < s.length; t++) { if (s[t].w === TARGET) posTot++; else negTot++; }
  const step = Math.max(1, Math.floor(negTot / NEG_N));
  const POS = [], NEG = []; let ni = 0;
  for (const s of sents) for (let t = 1; t < s.length; t++) { const f = feat(s, t); if (s[t].w === TARGET) POS.push(f); else { if (ni % step === 0) NEG.push(f); ni++; } }

  const pAll = new Uint8Array(DIM); let pn = 0;
  for (const f of POS) for (const a of f) if (!pAll[a]) { pAll[a] = 1; pn++; }

  // global doc-freq M over ALL positions
  const M = new Float64Array(DIM); let Ntot = 0;
  for (const s of sents) for (let t = 1; t < s.length; t++) { const f = feat(s, t); Ntot++; for (const a of f) M[a]++; }

  // masked coverage = |p'∩y'|/|y'|, ' = drop global-freq > δ ; empties excluded
  const covMasked = (SET, d, P) => { let s = 0, s2 = 0, n = 0, empty = 0; for (const y of SET) { let yr = 0, k = 0; for (const a of y) if (M[a] / Ntot <= d) { yr++; if (P[a]) k++; } if (yr === 0) { empty++; continue; } const c = k / yr; s += c; s2 += c * c; n++; } const m = n ? s / n : 0; return { m, sd: n ? Math.sqrt(Math.max(0, s2 / n - m * m)) : 0, empty, tot: n + empty }; };

  const w = process.stdout;
  w.write(`# bandOffset — ${path.basename(corpusArg)} | TARGET=${TARGET} R=${R} D=${D} bands sparse | F=${F} DIM=${DIM}\n`);
  w.write(`# POS n=${POS.length}  NEG n=${NEG.length}   |p|=${pn} (${(100 * pn / DIM).toFixed(3)}% of dim)\n`);

  const DELS = [Infinity, 0.01, 0.005, 0.002, 0.001];
  w.write(`\n── masked coverage (in-sample p) ──\n   δ       |p'|    cov pos     cov neg (rare)      ratio\n`);
  for (const d of DELS) { let pp = 0; for (let a = 0; a < DIM; a++) if (pAll[a] && M[a] / Ntot <= d) pp++; const cp = covMasked(POS, d, pAll), cn = covMasked(NEG, d, pAll); w.write(`   ${(d === Infinity ? "∞" : d).toString().padEnd(6)} ${String(pp).padStart(6)}   ${cp.m.toFixed(4)}   ${cn.m.toFixed(4)} ± ${cn.sd.toFixed(4)}   ${(cp.m / (cn.m || 1e-9)).toFixed(2)}\n`); }

  // held-out
  const pHO = new Uint8Array(DIM); const Pb = POS.filter((_, i) => i % 2 === 0), Pt = POS.filter((_, i) => i % 2 === 1);
  for (const f of Pb) for (const a of f) pHO[a] = 1;
  w.write(`\n── HELD-OUT (p from ${Pb.length} pos, tested on ${Pt.length} pos + neg) ──\n   δ       cov pos-HO   %∅   cov neg      %∅   ratio\n`);
  for (const d of DELS) { const cp = covMasked(Pt, d, pHO), cn = covMasked(NEG, d, pHO); w.write(`   ${(d === Infinity ? "∞" : d).toString().padEnd(6)}  ${cp.m.toFixed(4)}   ${(100 * cp.empty / cp.tot).toFixed(0).padStart(3)}%  ${cn.m.toFixed(4)}   ${(100 * cn.empty / cn.tot).toFixed(0).padStart(3)}%   ${(cp.m / (cn.m || 1e-9)).toFixed(2)}\n`); }

  // per-offset band purity at δ=0.001
  const DD = 0.001;
  w.write(`\n── per-offset band purity at δ=${DD} (d=1 is C, immediate predecessor) ──\n   offset   total   common   rare   %rare\n`);
  for (let d = 1; d <= D; d++) { let tot = 0, com = 0, rare = 0; for (let a = (d - 1) * F; a < d * F; a++) if (pAll[a]) { tot++; if (M[a] / Ntot > DD) com++; else rare++; } w.write(`   ${d === 1 ? "C  (t-1)" : "L" + (d - 1) + " (t-" + d + ")"}   ${String(tot).padStart(5)}   ${String(com).padStart(5)}   ${String(rare).padStart(4)}   ${tot ? (100 * rare / tot).toFixed(0) : "-"}%\n`); }
};

main();
