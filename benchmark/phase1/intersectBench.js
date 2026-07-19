"use strict";

// benchmark/phase1/intersectBench.js — split greedy's union Q by the per-signature fingerprint set:
//   I    = Q ∩ fingerprints   (rare + recurring + greedy-kept — the candidate "best" bits)
//   Sc   = Q \ fingerprints   (scaffold: greedy-kept but not any signature's atomic discriminator)
//   tail = fingerprints \ Q   (one-off rare bits greedy dropped)
// Put the SAME FP-aware density gate on each bit set; report |bits| / TRAIN / HELD / FP. Are the intersection bits the
// low-FP core?  node --max-old-space-size=8192 benchmark/phase1/intersectBench.js <dict> <corpus>  [TARGETS=... FPMAX=0.01]

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../../src/core/parts/tokenize.js";
import { encodeWord } from "../../src/core/signatures/wordEncoder.js";
import { segmentText } from "../../src/utilities/textSegmentation/segmentText.js";
import { fitClassGreedy } from "../../src/core/phase1/greedy.js";
import buildNegSet from "../../src/core/phase1/negSet.js";
import andCount from "../../src/core/math/sparse/andCount.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGETS = (process.env.TARGETS || "state,century,time,music,bank,physics").split(",");
const R = Number(process.env.R || 5), D = R + 1;
const NEGPOOL = Number(process.env.NEGPOOL || 60000), DELTA = Number(process.env.DELTA || 0.05), DBAND = Number(process.env.DBAND || 0.05);
const FPMAX = Number(process.env.FPMAX || 0.01), CAP = Number(process.env.CAP || 12000), SAMP = Number(process.env.SAMP || 5000);

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

const tset = new Set(TARGETS), classA = new Map(TARGETS.map((t) => [t, []])), negPool = [], bitCount = new Map();
let Ntot = 0, ni = 0; const addC = (y) => { Ntot++; for (const b of y) bitCount.set(b, (bitCount.get(b) || 0) + 1); return y; };
for (const s of sents) for (let t = 1; t < s.length; t++) { if (tset.has(s[t])) classA.get(s[t]).push(addC(feat(s, t))); else if (ni++ % Math.max(1, Math.floor((sents.length * 8) / NEGPOOL)) === 0) negPool.push(addC(feat(s, t))); }
const Mglob = new Set(); for (const [b, c] of bitCount) if (c / Ntot > DELTA) Mglob.add(b);

const sortAsc = (a) => [...new Set(a)].sort((x, y) => x - y);
const samp = (arr, n) => { if (arr.length <= n) return arr; const step = arr.length / n, out = []; for (let i = 0; i < arr.length; i += step) out.push(arr[Math.floor(i)]); return out; };
const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);

// per-signature minimal rare-conjunction fingerprint bits over training positives (bits with <FPMAX conjunction FP).
const fingerprintBits = (Atr, NEG) => {
  const post = new Map(); for (let i = 0; i < NEG.length; i++) for (const b of NEG[i]) { let l = post.get(b); if (!l) { l = []; post.set(b, l); } l.push(i); }
  const fc = (b) => (post.get(b) ? post.get(b).length : 0);
  const inter = (a, b) => { const out = []; let i = 0, j = 0; while (i < a.length && j < b.length) { if (a[i] === b[j]) { out.push(a[i]); i++; j++; } else if (a[i] < b[j]) i++; else j++; } return out; };
  const bits = new Set();
  for (const s of samp(Atr, CAP)) {
    const ord = [...s].sort((a, b) => fc(a) - fc(b)); let cur = null; const Q = [];
    for (const b of ord) { Q.push(b); const pb = post.get(b) || []; cur = cur === null ? pb.slice() : inter(cur, pb); if (cur.length / NEG.length < FPMAX) break; }
    if (cur && cur.length / NEG.length < FPMAX) for (const b of Q) bits.add(b);
  }
  return bits;
};
// FP-aware density gate on a bit set S: t = argmax recall−λ·FP on train pos + neg; eval train/held/FP.
const gateEval = (S, trPos, valPos, te, negNeg, lambda) => {
  const Qs = sortAsc(S); if (!Qs.length) return [0, 0, 0];
  const dens = (x) => (x.length ? andCount(x, Qs) / x.length : 0);
  const pos = valPos.map(dens), neg = negNeg.map(dens);
  const cands = [...new Set([0, ...pos, ...neg])].sort((a, b) => a - b);
  let bt = 0, bo = -Infinity;
  for (const t of cands) { const r = pos.filter((v) => v >= t).length / (pos.length || 1); const f = neg.length ? neg.filter((v) => v >= t).length / neg.length : 0; if (r - lambda * f > bo) { bo = r - lambda * f; bt = t; } }
  const fires = (x) => dens(x) >= bt;
  return [rate(fires, trPos), rate(fires, te), rate(fires, negNeg)];
};

// held-out (recall, FP) frontier sweeping the density threshold, + FP at matched recall / recall at matched FP.
const frontier = (S, te, negS) => { const Qs = sortAsc(S), out = []; for (let t = 0.01; t <= 0.6; t += 0.01) { const g = (x) => x.length && andCount(x, Qs) >= t * x.length; out.push([rate(g, te), rate(g, negS)]); } return out; };
const fpAtRec = (fr, R) => { let best = Infinity; for (const [r, f] of fr) if (r >= R - 1e-9) best = Math.min(best, f); return best === Infinity ? null : best; };

const w = process.stdout;
w.write(`# I=Q∩fp vs greedy-Q — FP at MATCHED recall (held) | FPMAX=${(FPMAX * 100).toFixed(0)}% | density gate\n`);
w.write(`  word          |A|    |Q|/|I|      R_greedy  FP@Rg: Q→I (Δ)   |   FP@rec60 Q/I   FP@rec70 Q/I   FP@rec80 Q/I\n`);
for (const word of TARGETS) {
  const Aw = classA.get(word); if (Aw.length < 40) continue;
  const nfa = Math.floor(Aw.length * 0.75); const fa = Aw.slice(0, nfa), te = Aw.slice(nfa);
  const nTr = Math.floor(fa.length * 0.75); const tr = fa.slice(0, nTr), val = fa.slice(nTr);
  const lambda = Math.min(3, Math.max(1, 1500 / fa.length));
  const g = fitClassGreedy(fa, negPool, Mglob, { delta: DBAND, peel: true, candidates: "union", Rstar: 0.6, thObj: "diff" });
  const Q = new Set(g.parts.flatMap((p) => [...p.Qs]));
  const neg = buildNegSet(fa, negPool, Mglob, { delta: DBAND });
  const fp = fingerprintBits(tr, neg.Neg);
  const I = [...Q].filter((b) => fp.has(b));
  const negS = samp(neg.Neg, SAMP);
  const [, Rg] = gateEval([...Q], samp(tr, SAMP), val, te, negS, lambda);     // greedy-Q FP-aware held recall
  const frQ = frontier([...Q], te, negS), frI = frontier(I, te, negS);
  const pc2 = (v) => (v == null ? " ·" : (100 * v).toFixed(0));
  const fpQg = fpAtRec(frQ, Rg), fpIg = fpAtRec(frI, Rg);
  const dg = fpQg != null && fpIg != null ? (100 * (fpQg - fpIg)).toFixed(0) : "·";
  const at = (R) => `${pc2(fpAtRec(frQ, R))}/${pc2(fpAtRec(frI, R))}`;
  w.write(`  ${word.padEnd(12)}${String(Aw.length).padStart(6)}   ${(Q.size + "/" + I.length).padEnd(12)} ${(100 * Rg).toFixed(0)}%      ${pc2(fpQg)}→${pc2(fpIg)} (${dg})       |   ${at(0.6).padEnd(13)} ${at(0.7).padEnd(13)} ${at(0.8)}\n`);
}
