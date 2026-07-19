"use strict";

// benchmark/phase1/andBench.js — §4.1 CV: additive objective (u + Mx) vs AND-objective (M̃ = GMG, unary gates edges,
// no linear term) — replicator both. Ship AND iff confFP drops (esp. bank) at equal-or-better held recall.
//   node --max-old-space-size=8192 benchmark/phase1/andBench.js <dict> <corpus>   [TARGETS=... RHO=80]

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../../src/core/parts/tokenize.js";
import { encodeWord } from "../../src/core/signatures/wordEncoder.js";
import { segmentText } from "../../src/utilities/textSegmentation/segmentText.js";
import fitClass from "../../src/core/phase1/fit.js";
import { jaccard } from "../../src/core/math/sparse/jaccard.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGETS = (process.env.TARGETS || "time,world,state,music,century,river,government,bank,physics,philosophy,hydrogen").split(",");
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

const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
const intraJac = (G) => { let s = 0, n = 0; for (let i = 0; i < G.length; i++) for (let j = i + 1; j < G.length; j++) { s += jaccard(G[i].Qbits, G[j].Qbits); n++; } return n ? s / n : 0; };
const meanQ = (G) => (G.length ? G.reduce((s, g) => s + g.Qbits.length, 0) / G.length : 0);

const CFG = { add: { rho: RHO, delta: DBAND }, and: { rho: RHO, delta: DBAND, andObjective: true } };
const KEYS = ["add", "and"];
const w = process.stdout;
w.write(`# §4.1 additive (u+Mx) vs AND (M̃=GMG) | replicator both | canon=${TARGETS.length} | 4-fold | held-rec/confFP/K/|Q|/jac\n`);
w.write(`  word          |A|    additive (u+Mx)             AND (M̃=GMG)\n`);
const agg = Object.fromEntries(KEYS.map((k) => [k, { r: 0, f: 0, k: 0, q: 0, j: 0 }])); let N = 0;
const rows = [];
for (const word of TARGETS) {
  const Aw = classA.get(word); if (Aw.length < 8) continue;
  const acc = Object.fromEntries(KEYS.map((k) => [k, [0, 0, 0, 0, 0]]));
  for (let fold = 0; fold < 4; fold++) {
    const fa = [], te = []; for (let k = 0; k < Aw.length; k++) (k % 4 === fold ? te : fa).push(Aw[k]);
    for (const key of KEYS) {
      const fit = fitClass(fa, negPool, Mglob, CFG[key]);
      acc[key][0] += rate((y) => fit.head.fires(y), te); acc[key][1] += rate((y) => fit.head.fires(y), fit.neg.Neg);
      acc[key][2] += fit.G.length; acc[key][3] += meanQ(fit.G); acc[key][4] += intraJac(fit.G);
    }
  }
  const row = { word, A: Aw.length }; const cells = KEYS.map((key) => {
    const a = acc[key].map((v) => v / 4);
    agg[key].r += a[0] * 100; agg[key].f += a[1] * 100; agg[key].k += a[2]; agg[key].q += a[3]; agg[key].j += a[4];
    row[key] = { R: +(a[0] * 100).toFixed(1), F: +(a[1] * 100).toFixed(1), K: +a[2].toFixed(1), Q: +a[3].toFixed(0), J: +a[4].toFixed(2) };
    return `${(a[0] * 100).toFixed(0)}/${(a[1] * 100).toFixed(0)}/${a[2].toFixed(0)}/${a[3].toFixed(0)}/${a[4].toFixed(2)}`.padEnd(28);
  });
  N++; rows.push(row);
  w.write(`  ${word.padEnd(12)}${String(Aw.length).padStart(6)}   ${cells.join("")}\n`);
}
const M = (o) => `${(o.r / N).toFixed(0)}/${(o.f / N).toFixed(0)}/${(o.k / N).toFixed(1)}/${(o.q / N).toFixed(0)}/${(o.j / N).toFixed(2)}`;
w.write(`  ${"MEAN".padEnd(12)}${"".padStart(6)}   ${KEYS.map((k) => M(agg[k]).padEnd(28)).join("")}\n`);
w.write(`# JSON ${JSON.stringify(rows)}\n`);
