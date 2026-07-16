"use strict";

// ============================================================================
// benchmark/hardNegMine.js — hard-negative mining for mask discovery (one round).
// p = ⋁_{i∈A} y_i, t_p = min|y_i|. Fire over corpus → false positives. Select HARD negatives two ways:
//   raw : top-N FP by popcount(y∧p)          (the TRAP — length-biased; longest sentences win)
//   cov : top-N FP by popcount(y∧p)/|y|, |y|≥LMIN   (coverage fraction — "looks like bank regardless of length")
// Then compute per-atom contrast three ways and compare the top atoms:
//   global : freq_A / freq_corpus            (negatives = everything — coarse: swiss/ubs proper nouns)
//   raw-HN : freq_A / freq_rawHN
//   cov-HN : freq_A / freq_covHN             (the bet: fine discriminators, common atoms cancel)
// Question 1 (precondition): are the cov hard-negatives semantically adjacent (fin/geo) or random/short?
// Question 2 (payload): does HN contrast surface DIFFERENT/finer atoms than global contrast?
// Usage: TARGET=bank ENC='L|C' R=5 N=5000 node --max-old-space-size=8192 benchmark/hardNegMine.js <dict> <corpus>
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { fuzzyEncodeWord } from "../src/core/signatures/wordEncoder.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";
import { kindToString } from "../src/core/parts/kind.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGET = process.env.TARGET || "bank";
const ENC = process.env.ENC || "L|C";
const R = process.env.R === "inf" ? Infinity : Number(process.env.R || 5);
const NHN = Number(process.env.N || 5000);       // hard-negative pool size
const LMIN = Number(process.env.LMIN || 8);      // min |y| for cov selection (avoid trivially-short full-coverage)
const NAMIN = Number(process.env.NAMIN || 3);    // only rank atoms recurrent in A (n_A ≥ this)

const FIELD = { fin: "banks banking credit loan loans lend lender lending borrow financial finance deposit deposits depositor mortgage money investment investor account accounts monetary currency capital fund funds cash savings interest debt bond bonds securities teller atm branch".split(" "), geo: "river rivers riverbank shore shores water waters stream streams lake creek valley bay coast estuary flood embankment".split(" ") };
const SENSE_RE = /mortgag|teller|deposit|lend|swi|ubs|orporat|danub|river|shore|creek|estuar|financ|invest|credit|monet|currenc|loan|money|branch/;

const win = (enc, r, p, L) => enc === "L|C" ? [r === Infinity ? 0 : Math.max(0, p - 1 - r), p - 1, -1] : [r === Infinity ? 0 : Math.max(0, p - r), r === Infinity ? L - 1 : Math.min(L - 1, p + r), p];

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const parts = dict.allParts(), F = dict.size();
  const cache = new Map();
  const atomsOf = (w) => { let a = cache.get(w); if (!a) { a = Int32Array.from(new Set(fuzzyEncodeWord(dict, w))); cache.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const toks = tokenizeStream(line.slice(seg.start, seg.end));
    const ws = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws.map((x) => ({ w: x, atoms: atomsOf(x) })));
  }
  const finSet = new Set(FIELD.fin), geoSet = new Set(FIELD.geo);
  const inP = new Uint8Array(F), stamp = new Int32Array(F).fill(-1); let pid = 0;

  // ---- PASS 1: p, nA, absA, t_p ----
  const nA = new Float64Array(F); let absA = 0, psz = 0;
  for (const s of sents) { const L = s.length; for (let p = 0; p < L; p++) { if (s[p].w !== TARGET) continue; const [lo, hi, sk] = win(ENC, R, p, L); if (hi < lo) continue; absA++; pid++; for (let q = lo; q <= hi; q++) { if (q === sk) continue; for (const a of s[q].atoms) { if (!inP[a]) { inP[a] = 1; psz++; } if (stamp[a] !== pid) { stamp[a] = pid; nA[a]++; } } } } }
  let tP = Infinity;
  for (const s of sents) { const L = s.length; for (let p = 0; p < L; p++) { if (s[p].w !== TARGET) continue; const [lo, hi, sk] = win(ENC, R, p, L); if (hi < lo) continue; pid++; let k = 0; for (let q = lo; q <= hi; q++) { if (q === sk) continue; for (const a of s[q].atoms) if (inP[a] && stamp[a] !== pid) { stamp[a] = pid; k++; } } if (k > 0 && k < tP) tP = k; } }
  if (tP === Infinity) tP = 1;

  // ---- PASS 2a: global M[a], and FP score arrays for thresholding ----
  const M = new Float64Array(F); let N = 0;
  const covScores = [], rawScores = [];
  for (const s of sents) { const L = s.length; for (let p = 0; p < L; p++) { const [lo, hi, sk] = win(ENC, R, p, L); if (hi < lo) continue; N++; pid++; let k = 0, yl = 0; for (let q = lo; q <= hi; q++) { if (q === sk) continue; for (const a of s[q].atoms) if (stamp[a] !== pid) { stamp[a] = pid; M[a]++; yl++; if (inP[a]) k++; } } if (s[p].w === TARGET) continue; rawScores.push(k); if (yl >= LMIN) covScores.push(k / yl); } }
  const kth = (arr, n) => { const c = Float64Array.from(arr); c.sort(); return c[Math.max(0, c.length - n)]; };   // n-th largest
  const rawThr = kth(rawScores, NHN), covThr = kth(covScores, NHN);

  // ---- PASS 2b: accumulate hard-negative atom counts + composition, for each method ----
  const mk = () => ({ nHN: new Float64Array(F), c: 0, ylSum: 0, fin: 0, geo: 0, cw: new Map() });
  const RAW = mk(), COV = mk();
  const collect = (G, atomsSeenPid, cw, yl) => { G.c++; G.ylSum += yl; if (finSet.has(cw)) G.fin++; else if (geoSet.has(cw)) G.geo++; G.cw.set(cw, (G.cw.get(cw) || 0) + 1); };
  for (const s of sents) {
    const L = s.length;
    for (let p = 0; p < L; p++) {
      if (s[p].w === TARGET) continue;
      const [lo, hi, sk] = win(ENC, R, p, L); if (hi < lo) continue;
      pid++; let k = 0, yl = 0; const ys = [];
      for (let q = lo; q <= hi; q++) { if (q === sk) continue; for (const a of s[q].atoms) if (stamp[a] !== pid) { stamp[a] = pid; ys.push(a); yl++; if (inP[a]) k++; } }
      const cw = s[p].w;
      if (k >= rawThr) { collect(RAW, pid, cw, yl); for (const a of ys) RAW.nHN[a]++; }
      if (yl >= LMIN && yl > 0 && (k / yl) >= covThr) { collect(COV, pid, cw, yl); for (const a of ys) COV.nHN[a]++; }
    }
  }

  const w = process.stdout;
  w.write(`# hardNegMine — ${path.basename(corpusArg)} | TARGET=${TARGET} ENC=[${ENC}] r=${R === Infinity ? "∞" : R} | N_HN=${NHN} LMIN=${LMIN} NAMIN=${NAMIN}\n`);
  w.write(`# |A|=${absA}  |p|=${psz}  t_p=${tP}  N=${N}  corpus mean|y|=${(rawScores.length ? 0 : 0)}\n`);

  // ---- composition ----
  const comp = (G, name, thr) => {
    const meanY = G.c ? G.ylSum / G.c : 0;
    const baseFin = FIELD.fin.reduce((s, wd) => s, 0);
    w.write(`\n── HARD NEGATIVES [${name}] (thr=${thr.toFixed ? thr.toFixed(3) : thr}) : ${G.c} contexts, mean|y|=${meanY.toFixed(1)}\n`);
    w.write(`   field share among HN center words:  fin=${(100 * G.fin / G.c).toFixed(2)}%   geo=${(100 * G.geo / G.c).toFixed(2)}%\n`);
    const top = [...G.cw.entries()].sort((a, b) => b[1] - a[1]).slice(0, 18);
    w.write(`   top HN center words: ${top.map(([wd, c]) => `${wd}(${c})`).join(" ")}\n`);
  };
  comp(RAW, "raw popcount", rawThr);
  comp(COV, "cov fraction", covThr);

  // ---- contrast atom rankings ----
  const rank = (denomFreq, label) => {
    const rows = [];
    for (let a = 0; a < F; a++) { if (nA[a] < NAMIN) continue; const fa = (nA[a] + 0.5) / (absA + 1); const fb = denomFreq(a); const lift = fa / fb; rows.push({ a, lift, nA: nA[a] }); }
    rows.sort((x, y) => y.lift - x.lift);
    const top = rows.slice(0, 25);
    w.write(`\n── TOP 25 CONTRAST ATOMS [${label}] (n_A≥${NAMIN}) ──\n`);
    w.write(`   ${top.map((r) => `${parts[r.a].value}:${kindToString(parts[r.a].kind)[0]}${SENSE_RE.test(parts[r.a].value) ? "*" : ""}(${r.nA | 0},L${r.lift < 100 ? r.lift.toFixed(0) : "∞"})`).join(" ")}\n`);
    const sense = top.filter((r) => SENSE_RE.test(parts[r.a].value)).length;
    w.write(`   sense-adjacent in top-25: ${sense}/25   (marked *)\n`);
    return new Set(top.map((r) => r.a));
  };
  const globF = (a) => (M[a] - nA[a] + 0.5) / (N - absA + 1);
  const rawF = (a) => (RAW.nHN[a] + 0.5) / (RAW.c + 1);
  const covF = (a) => (COV.nHN[a] + 0.5) / (COV.c + 1);
  const gSet = rank(globF, "global  freq_A/freq_corpus");
  const rSet = rank(rawF, "raw-HN  freq_A/freq_rawHN");
  const cSet = rank(covF, "cov-HN  freq_A/freq_covHN");

  const inter = (X, Y) => [...X].filter((a) => Y.has(a)).length;
  w.write(`\n── OVERLAP of top-25 atom sets (does HN surface NEW discriminators?) ──\n`);
  w.write(`   global∩cov-HN = ${inter(gSet, cSet)}/25    global∩raw-HN = ${inter(gSet, rSet)}/25    cov-HN∩raw-HN = ${inter(cSet, rSet)}/25\n`);
};

main();
