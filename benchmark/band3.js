"use strict";

// ============================================================================
// benchmark/band3.js — 3-band hybrid: L | L1 | C  (concatenate near, OR the far tail).
//   C  band [2F,3F): word t-1 (immediate predecessor)   — resolved
//   L1 band [F,2F) : word t-2                            — resolved
//   L  band [0,F)  : OR of words t-3 .. t-D (far tail)   — pooled
// ENCODER=fuzzy|sparse. Tests: does fuzzy blow up the COMMON mask but leave the pos/neg
// separation (ratio) similar to sparse? Metrics: masked coverage (in-sample+held-out),
// per-band purity, and mask size at δ=0.001.
// Usage: ENCODER=fuzzy TARGET=bank R=5 node --max-old-space-size=8192 benchmark/band3.js english.txt <corpus>
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { fuzzyEncodeWord, encodeWord } from "../src/core/signatures/wordEncoder.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGET = process.env.TARGET || "bank";
const R = Number(process.env.R || 5);
const D = R + 1;                     // far tail spans t-3 .. t-D
const NEG_N = Number(process.env.NEG || 3000);
const ENC = process.env.ENCODER === "sparse" ? encodeWord : fuzzyEncodeWord;

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F;
  const cache = new Map();
  const atomsOf = (w) => { let a = cache.get(w); if (!a) { a = Int32Array.from(new Set(ENC(dict, w))); cache.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const toks = tokenizeStream(line.slice(seg.start, seg.end));
    const ws = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws.map((x) => ({ w: x, atoms: atomsOf(x) })));
  }

  const stampF = new Int32Array(DIM).fill(-1); let fpid = 0;
  // 3-band feature for predicting word[t]
  const feat = (s, t) => {
    fpid++; const y = [];
    for (let q = t - 3; q >= Math.max(0, t - D); q--) for (const a of s[q].atoms) if (stampF[a] !== fpid) { stampF[a] = fpid; y.push(a); }   // L band [0,F): far OR
    if (t - 2 >= 0) for (const a of s[t - 2].atoms) y.push(a + F);       // L1 band [F,2F): t-2
    if (t - 1 >= 0) for (const a of s[t - 1].atoms) y.push(a + 2 * F);   // C  band [2F,3F): t-1
    return y;
  };

  let negTot = 0;
  for (const s of sents) for (let t = 1; t < s.length; t++) if (s[t].w !== TARGET) negTot++;
  const step = Math.max(1, Math.floor(negTot / NEG_N));
  const POS = [], NEG = []; let ni = 0;
  for (const s of sents) for (let t = 1; t < s.length; t++) { const f = feat(s, t); if (s[t].w === TARGET) POS.push(f); else { if (ni % step === 0) NEG.push(f); ni++; } }

  const pAll = new Uint8Array(DIM); let pn = 0;
  for (const f of POS) for (const a of f) if (!pAll[a]) { pAll[a] = 1; pn++; }
  const M = new Float64Array(DIM); let Ntot = 0;
  for (const s of sents) for (let t = 1; t < s.length; t++) { const f = feat(s, t); Ntot++; for (const a of f) M[a]++; }

  const covMasked = (SET, d, P) => { let s = 0, s2 = 0, n = 0, empty = 0; for (const y of SET) { let yr = 0, k = 0; for (const a of y) if (M[a] / Ntot <= d) { yr++; if (P[a]) k++; } if (yr === 0) { empty++; continue; } const c = k / yr; s += c; s2 += c * c; n++; } const m = n ? s / n : 0; return { m, sd: n ? Math.sqrt(Math.max(0, s2 / n - m * m)) : 0, empty, tot: n + empty }; };

  const w = process.stdout;
  w.write(`# band3 [L|L1|C] — ${path.basename(corpusArg)} | ENCODER=${process.env.ENCODER === "sparse" ? "sparse" : "fuzzy"} TARGET=${TARGET} | F=${F} DIM=${DIM}\n`);
  w.write(`# POS n=${POS.length}  NEG n=${NEG.length}   |p|=${pn} (${(100 * pn / DIM).toFixed(3)}% of dim)   mean|y| pos=${(POS.reduce((a, f) => a + f.length, 0) / POS.length).toFixed(1)}\n`);

  const DELS = [0.002, 0.0015, 0.001, 0.0008, 0.0006, 0.0004, 0.0003, 0.0002, 0.0001];
  w.write(`\n── masked coverage (in-sample p) ──\n   δ       |p'|    cov pos   cov neg (rare)      ratio\n`);
  for (const d of DELS) { let pp = 0; for (let a = 0; a < DIM; a++) if (pAll[a] && M[a] / Ntot <= d) pp++; const cp = covMasked(POS, d, pAll), cn = covMasked(NEG, d, pAll); w.write(`   ${(d === Infinity ? "∞" : d).toString().padEnd(6)} ${String(pp).padStart(6)}   ${cp.m.toFixed(4)}   ${cn.m.toFixed(4)} ± ${cn.sd.toFixed(4)}   ${(cp.m / (cn.m || 1e-9)).toFixed(2)}\n`); }

  const pHO = new Uint8Array(DIM); const Pb = POS.filter((_, i) => i % 2 === 0), Pt = POS.filter((_, i) => i % 2 === 1);
  for (const f of Pb) for (const a of f) pHO[a] = 1;
  w.write(`\n── HELD-OUT (p from ${Pb.length} pos, tested on ${Pt.length} pos + neg) ──\n   δ       cov pos-HO   %∅   cov neg      %∅   ratio\n`);
  for (const d of DELS) { const cp = covMasked(Pt, d, pHO), cn = covMasked(NEG, d, pHO); w.write(`   ${(d === Infinity ? "∞" : d).toString().padEnd(6)}  ${cp.m.toFixed(4)}   ${(100 * cp.empty / cp.tot).toFixed(0).padStart(3)}%  ${cn.m.toFixed(4)}   ${(100 * cn.empty / cn.tot).toFixed(0).padStart(3)}%   ${(cp.m / (cn.m || 1e-9)).toFixed(2)}\n`); }

  const DD = 0.001;
  const names = ["L  (t-3..t-6, OR)", "L1 (t-2)", "C  (t-1)"];
  w.write(`\n── per-band purity + mask at δ=${DD} ──\n   band                total   common(masked)   rare(kept)   %rare\n`);
  let totCom = 0;
  for (let b = 0; b < 3; b++) { let tot = 0, com = 0, rare = 0; for (let a = b * F; a < (b + 1) * F; a++) if (pAll[a]) { tot++; if (M[a] / Ntot > DD) com++; else rare++; } totCom += com; w.write(`   ${names[b].padEnd(20)} ${String(tot).padStart(5)}   ${String(com).padStart(10)}   ${String(rare).padStart(9)}   ${tot ? (100 * rare / tot).toFixed(0) : "-"}%\n`); }
  w.write(`   → total common bits masked out = ${totCom} / ${pn} (${(100 * totCom / pn).toFixed(0)}% of p)\n`);
};

main();
