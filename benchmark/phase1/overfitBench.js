"use strict";

// benchmark/phase1/overfitBench.js — per-signature "overfit fingerprints". For each positive signature s, greedily
// find the MINIMAL rare bit-conjunction Q_s ⊆ s whose AND fires on <1% of negatives (the minimal viable
// discriminative set + its threshold t_s=|Q_s|). Then mine A_overfit={(Q_s,t_s)} for structure: fingerprint sizes,
// bit recurrence, distinct fingerprints, shared-core coverage, and overlap with the best greedy part.
//   node --max-old-space-size=8192 benchmark/phase1/overfitBench.js <dict> <corpus>  [TARGETS=state,century FPMAX=0.01 CAP=20000]

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../../src/core/parts/tokenize.js";
import { encodeWord } from "../../src/core/signatures/wordEncoder.js";
import { segmentText } from "../../src/utilities/textSegmentation/segmentText.js";
import { fitClassGreedy } from "../../src/core/phase1/greedy.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGETS = (process.env.TARGETS || "state,century,time").split(",");
const R = Number(process.env.R || 5), D = R + 1;
const NEGPOOL = Number(process.env.NEGPOOL || 60000), DELTA = Number(process.env.DELTA || 0.05), DBAND = Number(process.env.DBAND || 0.05);
const FPMAX = Number(process.env.FPMAX || 0.01), CAP = Number(process.env.CAP || 20000);

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
const feat = (s, t) => { sp++; const y = []; for (let dd = 1; dd <= D; dd++) { const q = t - dd; if (q < 0) break; const off = band(dd) * F; for (const a of partsOf(s[q])) { const b = a + off; if (stmp[b] !== sp) { stmp[b] = sp; y.push(b); } } } return y.sort((a, b) => a - b); };

const tset = new Set(TARGETS), classA = new Map(TARGETS.map((t) => [t, []])), negPool = [];
let ni = 0; for (const s of sents) for (let t = 1; t < s.length; t++) { if (tset.has(s[t])) classA.get(s[t]).push(feat(s, t)); else if (ni++ % Math.max(1, Math.floor((sents.length * 8) / NEGPOOL)) === 0) negPool.push(feat(s, t)); }
const Mglob = new Set(); { const bc = new Map(); let N = 0; for (const y of negPool) { N++; for (const b of y) bc.set(b, (bc.get(b) || 0) + 1); } }  // (Mglob unused for fingerprints; kept for greedy)
{ const bc = new Map(); let N = 0; for (const [word, arr] of classA) for (const y of arr) { N++; for (const b of y) bc.set(b, (bc.get(b) || 0) + 1); } for (const [b, c] of bc) if (c / Math.max(1, N) > DELTA) Mglob.add(b); }

const NEG = negPool, nNeg = NEG.length;
// neg posting lists over every bit that occurs in negatives (sorted neg-index arrays) + per-bit neg count.
const post = new Map(); for (let i = 0; i < NEG.length; i++) for (const b of NEG[i]) { let l = post.get(b); if (!l) { l = []; post.set(b, l); } l.push(i); }
const fCount = (b) => (post.get(b) ? post.get(b).length : 0);
const intersect = (a, b) => { const out = []; let i = 0, j = 0; while (i < a.length && j < b.length) { if (a[i] === b[j]) { out.push(a[i]); i++; j++; } else if (a[i] < b[j]) i++; else j++; } return out; };

const w = process.stdout;
w.write(`# per-signature overfit fingerprints | FPMAX=${(FPMAX * 100).toFixed(1)}% | negPool=${nNeg}\n`);
for (const word of TARGETS) {
  const Aw = classA.get(word); if (!Aw || Aw.length < 40) continue;
  const step = Math.max(1, Math.floor(Aw.length / CAP));
  const sample = []; for (let i = 0; i < Aw.length; i += step) sample.push(Aw[i]);
  const g = fitClassGreedy(Aw.slice(0, Math.floor(Aw.length * 0.75)), negPool, Mglob, { delta: DBAND, peel: true, candidates: "union", Rstar: 0.6, thObj: "diff" });
  const greedyQ = new Set(g.parts.flatMap((p) => [...p.Qs]));

  const sizes = [], fps = [], bitUse = new Map(), fpKeys = new Map(); let viable = 0, single = 0;
  for (const s of sample) {
    const bits = [...s].sort((a, b) => fCount(a) - fCount(b));      // rarest-in-neg first
    let cur = null; const Q = []; let ok = false;
    for (const b of bits) {
      Q.push(b); const pb = post.get(b) || [];
      cur = cur === null ? pb.slice() : intersect(cur, pb);
      if (cur.length / nNeg < FPMAX) { ok = true; break; }
    }
    if (!ok) continue;                                              // no viable <FPMAX fingerprint from this signature
    viable++; sizes.push(Q.length); fps.push(cur.length / nNeg); if (Q.length === 1) single++;
    for (const b of Q) bitUse.set(b, (bitUse.get(b) || 0) + 1);
    const key = Q.slice().sort((a, b) => a - b).join(","); fpKeys.set(key, (fpKeys.get(key) || 0) + 1);
  }
  sizes.sort((a, b) => a - b);
  const mean = (arr) => (arr.length ? arr.reduce((s, x) => s + x, 0) / arr.length : 0);
  const hist = [1, 2, 3, 4, 5].map((k) => sizes.filter((s) => (k < 5 ? s === k : s >= 5)).length);
  const distinctBits = bitUse.size;
  const byUse = [...bitUse.entries()].sort((a, b) => b[1] - a[1]);
  // shared-core coverage: fewest top-recurring bits whose union touches X% of fingerprints
  const cover = (frac) => { const need = Math.ceil(frac * viable); const covered = new Set(); let bitsUsed = 0; const perBitSigs = new Map(); // approx via greedy set cover on fingerprint keys
    // rebuild fingerprint bit-sets
    const fpSets = [...fpKeys.entries()].flatMap(([k, c]) => Array(c).fill(k.split(",").map(Number)));
    const remaining = new Set(fpSets.map((_, i) => i)); const contains = new Map();
    fpSets.forEach((fp, i) => fp.forEach((b) => { let l = contains.get(b); if (!l) { l = []; contains.set(b, l); } l.push(i); }));
    while (covered.size < need && remaining.size) { let best = null, bestGain = -1; for (const [b, idxs] of contains) { let g2 = 0; for (const i of idxs) if (remaining.has(i)) g2++; if (g2 > bestGain) { bestGain = g2; best = b; } } if (!best || bestGain <= 0) break; bitsUsed++; for (const i of (contains.get(best) || [])) { remaining.delete(i); covered.add(i); } } return bitsUsed; };
  const topGreedyFrac = byUse.slice(0, 50).filter(([b]) => greedyQ.has(b)).length / Math.min(50, byUse.length || 1);
  const distinctFp = fpKeys.size;
  const topFp = [...fpKeys.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, c]) => `${c}×{${k.split(",").length}b}`).join(" ");

  w.write(`\n=== ${word} | |A|=${Aw.length} sampled=${sample.length} viable(<${(FPMAX * 100).toFixed(0)}%FP)=${viable} (${(100 * viable / sample.length).toFixed(0)}%) ===\n`);
  w.write(`  fingerprint size |Q_s|: mean ${mean(sizes).toFixed(2)} median ${sizes[Math.floor(sizes.length / 2)]} min ${sizes[0]} max ${sizes[sizes.length - 1]} | hist[1,2,3,4,5+]=[${hist}] | single-bit ${single} (${(100 * single / viable).toFixed(0)}%)\n`);
  w.write(`  achieved FP: mean ${(100 * mean(fps)).toFixed(2)}% | distinct bits used ${distinctBits} | distinct fingerprints ${distinctFp} (${(100 * distinctFp / viable).toFixed(0)}% unique) | top ${topFp}\n`);
  w.write(`  bit recurrence: top bit in ${byUse[0][1]} fps (${(100 * byUse[0][1] / viable).toFixed(0)}%) | top-10 bits Σuse ${byUse.slice(0, 10).reduce((s, e) => s + e[1], 0)} | in greedyQ: top-50 ${(100 * topGreedyFrac).toFixed(0)}%\n`);
  w.write(`  shared-core set-cover: ${cover(0.5)} bits cover 50% of fps · ${cover(0.9)} bits cover 90% · ${cover(1.0)} bits cover 100% (of ${distinctFp} distinct)\n`);
  // does greedy's Q capture the fingerprint bits? counts + positive-coverage
  const fpBitsInG = [...bitUse.keys()].filter((b) => greedyQ.has(b)).length;
  let sigWithBitInG = 0; for (const [k, c] of fpKeys) if (k.split(",").some((b) => greedyQ.has(+b))) sigWithBitInG += c;   // signatures whose fingerprint bit ∈ greedy
  const useInG = byUse.reduce((s, [b, c]) => s + (greedyQ.has(b) ? c : 0), 0), totUse = byUse.reduce((s, [, c]) => s + c, 0);
  w.write(`  greedy capture: |greedyQ|=${greedyQ.size} vs distinct fp-bits ${distinctBits} (${(100 * distinctBits / Math.max(1, greedyQ.size)).toFixed(0)}% more) | fp-bits in greedyQ ${fpBitsInG} (${(100 * fpBitsInG / distinctBits).toFixed(0)}%) | positives whose fp-bit∈greedyQ ${(100 * sigWithBitInG / viable).toFixed(0)}% | use-weighted ${(100 * useInG / totUse).toFixed(0)}%\n`);
}
