"use strict";

// benchmark/phase1/parallelBench.js — verify `parallelFit` is IDENTICAL to sequential `fitClass` (fitClass is
// deterministic) and measure the multi-core speedup. Featurizes once, extracts each word's model both ways.
//
// Usage:  node --max-old-space-size=8192 benchmark/phase1/parallelBench.js <dict> <corpus>
//   TARGETS="..."  WORKERS=7  NEGPOOL=60000

import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../../src/core/parts/tokenize.js";
import { encodeWord } from "../../src/core/signatures/wordEncoder.js";
import { segmentText } from "../../src/utilities/textSegmentation/segmentText.js";
import fitClass from "../../src/core/phase1/fit.js";
import { parallelFit } from "../../src/core/phase1/parallelFit.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGETS = (process.env.TARGETS || "time,world,state,music,century,river,government,bank,physics,philosophy,hydrogen").split(",");
const R = Number(process.env.R || 5), D = R + 1;
const NEGPOOL = Number(process.env.NEGPOOL || 60000), RHO = Number(process.env.RHO || 80);
const DELTA = Number(process.env.DELTA || 0.05), DBAND = Number(process.env.DBAND || 0.05);
const WORKERS = Number(process.env.WORKERS || Math.max(1, os.cpus().length - 1));

const [dictArg, corpusArg] = process.argv.slice(2);
const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F;
const pc = new Map(); const partsOf = (w) => { let a = pc.get(w); if (!a) { a = [...new Set(encodeWord(dict, w))]; pc.set(w, a); } return a; };
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
for (const s of sents) for (let t = 1; t < s.length; t++) {
  if (tset.has(s[t])) classA.get(s[t]).push(addC(feat(s, t)));
  else if (ni++ % Math.max(1, Math.floor((sents.length * 8) / NEGPOOL)) === 0) negPool.push(addC(feat(s, t)));
}
const Mglob = new Set(); for (const [b, c] of bitCount) if (c / Ntot > DELTA) Mglob.add(b);
const words = TARGETS.filter((w) => (classA.get(w) || []).length >= 8);
const opts = { rho: RHO, delta: DBAND };
const w = process.stdout;
w.write(`# parallelFit vs sequential | words=${words.length} negPool=${negPool.length} workers=${WORKERS} (cores=${os.cpus().length})\n`);

// sequential baseline
const tS = process.hrtime.bigint(); const seq = new Map();
for (const word of words) seq.set(word, fitClass(classA.get(word), negPool, Mglob, opts));
const seqMs = Number(process.hrtime.bigint() - tS) / 1e6;

// parallel
const tP = process.hrtime.bigint();
const par = await parallelFit(new Map(words.map((word) => [word, classA.get(word)])), negPool, Mglob, opts, { workers: WORKERS });
const parMs = Number(process.hrtime.bigint() - tP) / 1e6;

// verify identical models
const gKey = (G) => G.map((g) => `${g.m}:${g.alpha.toFixed(6)}:${[...g.Qbits].join(",")}`).join("|");
let mismatch = 0;
for (const word of words) {
  const a = seq.get(word), b = par.get(word);
  if (!b || b.error || gKey(a.G) !== gKey(b.G) || Math.abs(a.theta - b.theta) > 1e-9) { mismatch++; w.write(`  ⚠ MISMATCH ${word}${b && b.error ? " (" + b.error + ")" : ""}\n`); }
}
w.write(`  identical models: ${words.length - mismatch}/${words.length}\n`);
w.write(`  sequential ${seqMs.toFixed(0)}ms | parallel ${parMs.toFixed(0)}ms | speedup ${(seqMs / parMs).toFixed(2)}× on ${WORKERS} workers\n`);
