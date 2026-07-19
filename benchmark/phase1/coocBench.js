"use strict";

// benchmark/phase1/coocBench.js — does the co-occurrence STRUCTURE inside greedy's Q buy a lower threshold at low FP?
// Compare two gates on greedy's union-of-bits Q over their recall/FP frontier: (a) flat density |Q∩x| ≥ t·|x|;
// (b) co-occurrence score(x)=Σ_{a,b∈x∩Q} M_Q[a,b] (class-vs-neg pairwise weights). Report recall @ FP≈5/10/15%.
//   node --max-old-space-size=8192 benchmark/phase1/coocBench.js <dict> <corpus>  [TARGETS=state,century,time]

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../../src/core/parts/tokenize.js";
import { encodeWord } from "../../src/core/signatures/wordEncoder.js";
import { segmentText } from "../../src/utilities/textSegmentation/segmentText.js";
import { fitClassGreedy, refineOnSubset } from "../../src/core/phase1/greedy.js";
import buildNegSet, { pairKey } from "../../src/core/phase1/negSet.js";
import buildAffinity from "../../src/core/phase1/affinity.js";
import andCount from "../../src/core/math/sparse/andCount.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGETS = (process.env.TARGETS || "state,century,time,music,bank").split(",");
const R = Number(process.env.R || 5), D = R + 1, RHO = Number(process.env.RHO || 80);
const NEGPOOL = Number(process.env.NEGPOOL || 60000), DELTA = Number(process.env.DELTA || 0.05), DBAND = Number(process.env.DBAND || 0.05);

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
// recall @ target FP on a frontier of (θ, recall, fp) points: highest recall with fp ≤ target.
const recAtFp = (front, target) => { let best = 0; for (const [, r, f] of front) if (f <= target && r > best) best = r; return best; };

const w = process.stdout;
w.write(`# co-occurrence gate vs flat density on greedy's Q | recall @ FP≤5/10/15%\n`);
w.write(`  word          |A|   |Q|  subparts        flat: rec@5/10/15       cooc: rec@5/10/15\n`);
for (const word of TARGETS) {
  const Aw = classA.get(word); if (Aw.length < 40) continue;
  const nfa = Math.floor(Aw.length * 0.75); const fa = Aw.slice(0, nfa), te = Aw.slice(nfa);
  const g = fitClassGreedy(fa, negPool, Mglob, { delta: DBAND, peel: true, candidates: "union", Rstar: 0.6, thObj: "diff" });
  const Q = sortAsc(g.parts.flatMap((p) => [...p.Qs])); if (Q.length < 4) { w.write(`  ${word}: greedy Q too small (${Q.length})\n`); continue; }
  const wU = new Float64Array(fa.length).fill(1 / fa.length);
  const neg = buildNegSet(fa, negPool, Mglob, { delta: DBAND });
  const subNeg = { ...neg, pPlus: Q, pset: new Set(Q) };
  const { u, edges } = buildAffinity(fa, wU, subNeg, {});                 // M_Q over greedy's bits
  const pm = new Map(); for (let e = 0; e < edges.m.length; e++) { const a = Q[edges.i[e]], b = Q[edges.j[e]]; pm.set(pairKey(a, b), edges.m[e]); }
  const Qset = new Set(Q);
  const coocScore = (x) => { const xq = []; for (const b of x) if (Qset.has(b)) xq.push(b); let s = 0; for (let i = 0; i < xq.length; i++) for (let j = i + 1; j < xq.length; j++) { const v = pm.get(pairKey(xq[i], xq[j])); if (v) s += v; } return s; };
  // subpart decomposition: peel the replicator on Q a few times
  let Qrem = [...Q]; const subs = [];
  for (let k = 0; k < 5 && Qrem.length >= 3; k++) { const cl = refineOnSubset(Qrem, fa, wU, neg, {}); if (!cl.length || cl.length === Qrem.length) { subs.push(cl.length || Qrem.length); break; } subs.push(cl.length); const cs = new Set(cl); Qrem = Qrem.filter((b) => !cs.has(b)); }
  // frontiers on te (recall) + neg.Neg (FP)
  const flat = []; for (let t = 0.02; t <= 0.5; t += 0.02) { const Qs = Q; const rec = te.filter((x) => x.length && andCount(x, Qs) >= t * x.length).length / te.length; const fp = neg.Neg.filter((x) => x.length && andCount(x, Qs) >= t * x.length).length / neg.Neg.length; flat.push([t, rec, fp]); }
  const teS = te.map(coocScore), neS = neg.Neg.map(coocScore);
  const cand = [...new Set([...teS, ...neS])].sort((a, b) => a - b);
  const cooc = []; for (const th of cand) { const rec = teS.filter((s) => s >= th).length / te.length; const fp = neS.filter((s) => s >= th).length / neS.length; cooc.push([th, rec, fp]); }
  const fmt = (fr) => `${(recAtFp(fr, 0.05) * 100).toFixed(0)}/${(recAtFp(fr, 0.10) * 100).toFixed(0)}/${(recAtFp(fr, 0.15) * 100).toFixed(0)}`;
  w.write(`  ${word.padEnd(12)}${String(Aw.length).padStart(5)} ${String(Q.length).padStart(4)}  ${("[" + subs.join(",") + "]").padEnd(15)} ${fmt(flat).padEnd(23)} ${fmt(cooc)}\n`);
}
