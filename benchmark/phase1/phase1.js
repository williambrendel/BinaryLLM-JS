"use strict";

// benchmark/phase1.js — Phase-1 deployed-config evaluation. Featurizes each target word's positive contexts
// (3F canon signature) + a shared confusable negative pool, then runs the deployed `fitClass` (exp replicator +
// support-early-stop + PMI-difference affinity + AdaBoost + θ-tuned α-sum head) under 4-fold cross-validation,
// reporting held-out recall / confusable-FP per word.
//
// Usage:  node --max-old-space-size=8192 benchmark/phase1.js <dict> <corpus>
//   TARGETS="time,state,bank"  comma-separated words (default: the 11 canonical targets)
//   NEGPOOL=60000  negative-pool cap · DELTA=0.05  common-bit threshold · RHO=80  replicator regularizer

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../../src/core/parts/tokenize.js";
import { encodeWord } from "../../src/core/signatures/wordEncoder.js";
import { segmentText } from "../../src/utilities/textSegmentation/segmentText.js";
import fitClass from "../../src/core/phase1/fit.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGETS = (process.env.TARGETS || "time,world,state,music,century,river,government,bank,physics,philosophy,hydrogen").split(",");
const R = Number(process.env.R || 5), D = R + 1;
const NEGPOOL = Number(process.env.NEGPOOL || 60000), RHO = Number(process.env.RHO || 80);
const DELTA = Number(process.env.DELTA || 0.05), DBAND = Number(process.env.DBAND || 0.05);

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F;
  const pc = new Map(); const partsOf = (w) => { let a = pc.get(w); if (!a) { a = [...new Set(encodeWord(dict, w))]; pc.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const ws = tokenizeStream(line.slice(seg.start, seg.end)).filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws);
  }
  // 3F canonical context signature: for each target occurrence, encode the D preceding words into 3 distance
  // bands (dd=1 adjacent → band 2, dd∈2–3 near → band 1, dd≥4 far → band 0).
  const band = (dd) => (dd === 1 ? 2 : dd <= 3 ? 1 : 0);
  const stmp = new Int32Array(DIM).fill(-1); let sp = 0;
  const feat = (s, t) => { sp++; const y = []; for (let dd = 1; dd <= D; dd++) { const q = t - dd; if (q < 0) break; const off = band(dd) * F; for (const a of partsOf(s[q])) { const b = a + off; if (stmp[b] !== sp) { stmp[b] = sp; y.push(b); } } } return y.sort((a, b) => a - b); };

  // featurize: per-target positive contexts + a shared confusable negative pool (strided sample of non-targets).
  const tset = new Set(TARGETS), classA = new Map(TARGETS.map((t) => [t, []])), negPool = [], bitCount = new Map();
  let Ntot = 0, ni = 0; const addC = (y) => { Ntot++; for (const b of y) bitCount.set(b, (bitCount.get(b) || 0) + 1); return y; };
  for (const s of sents) for (let t = 1; t < s.length; t++) {
    if (tset.has(s[t])) classA.get(s[t]).push(addC(feat(s, t)));
    else if (ni++ % Math.max(1, Math.floor((sents.length * 8) / NEGPOOL)) === 0) negPool.push(addC(feat(s, t)));
  }
  const Mglob = new Set(); for (const [b, c] of bitCount) if (c / Ntot > DELTA) Mglob.add(b);   // common bits

  // 4-fold cross-validation of the deployed config, held-out recall / confusable-FP per word.
  const rate = (fn, pool) => (pool.length ? pool.filter(fn).length / pool.length : 0);
  process.stdout.write(`# phase1 deployed-config CV (4-fold held-out) | words=${TARGETS.length} negPool=${negPool.length}\n`);
  process.stdout.write(`  word          |A|    K   trainRec  recHeld   confFP    ms/fit\n`);
  let sR = 0, sF = 0, sTr = 0, sK = 0, sMs = 0, nw = 0;
  for (const word of TARGETS) {
    const Aw = classA.get(word); if (Aw.length < 8) continue;
    let rS = 0, fS = 0, trS = 0, kS = 0, msS = 0;
    for (let f = 0; f < 4; f++) {
      const fa = [], te = []; for (let k = 0; k < Aw.length; k++) (k % 4 === f ? te : fa).push(Aw[k]);
      const t0 = process.hrtime.bigint(); const fit = fitClass(fa, negPool, Mglob, { rho: RHO, delta: DBAND }); msS += Number(process.hrtime.bigint() - t0) / 1e6;
      rS += rate((y) => fit.head.fires(y), te); fS += rate((y) => fit.head.fires(y), fit.neg.Neg); trS += rate((y) => fit.head.fires(y), fa); kS += fit.G.length;
    }
    const rec = 100 * rS / 4, fp = 100 * fS / 4, tr = 100 * trS / 4, k = kS / 4, ms = msS / 4;
    sR += rec; sF += fp; sTr += tr; sK += k; sMs += ms; nw++;
    process.stdout.write(`  ${word.padEnd(12)}${String(Aw.length).padStart(5)}   ${k.toFixed(0).padStart(2)}   ${tr.toFixed(1).padStart(5)}%   ${rec.toFixed(1).padStart(5)}%   ${fp.toFixed(1).padStart(5)}%   ${ms.toFixed(0).padStart(5)}\n`);
  }
  process.stdout.write(`  ${"MEAN".padEnd(12)}${"".padStart(5)}   ${(sK / nw).toFixed(0).padStart(2)}   ${(sTr / nw).toFixed(1).padStart(5)}%   ${(sR / nw).toFixed(1).padStart(5)}%   ${(sF / nw).toFixed(1).padStart(5)}%   ${(sMs / nw).toFixed(0).padStart(5)}\n`);
};

main();
