"use strict";

// ============================================================================
// benchmark/poolReadout.js
//
// The decisive test: does the discovered POOL pay off? Held-out log-loss of
//   (1) unigram              (2) bigram P(next|cur)
//   (3) single big tree      (4) forest pool → second-pass tree
// on a positional train/test split, scored uniformly (leaf + path decoder) with
// the calibration gate vs unigram. Log-loss is the decision metric — purity never
// was. Records stream from the corpus into memory (bounded, no cached dataset).
//
// Usage: node benchmark/poolReadout.js <dict> <corpus> [maxRecords]
//   CHUNK=8000 DMAX=25 ALPHA=1 POOLCAP=512 HEAD=gpt
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { streamSignatureChunks } from "../src/core/signatures/streamSignatures.js";
import { buildCartTree, buildPoolTree, buildUnigram, buildContextGram, evaluate, evaluatePath, calibrationGate } from "../src/core/predict/index.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, dir) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(dir, p) : p);
const loadDict = (arg) => { const p = resolveIn(arg, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const unionSorted = (a, b) => { const out = []; let i = 0, j = 0; while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } } while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out; };

const collectParts = (node, pool) => {
  if (node.part === undefined) return;
  const key = node.part.slice().sort((a, b) => a - b).join(",");
  const e = pool.get(key); if (e) e.count++; else pool.set(key, { bits: node.part, k: node.k, count: 1 });
  collectParts(node.present, pool); collectParts(node.absent, pool);
};

const main = async () => {
  const [dictArg, corpusArg, maxArg] = process.argv.slice(2);
  if (!dictArg || !corpusArg) { process.stderr.write("usage: node benchmark/poolReadout.js <dict> <corpus> [maxRecords]\n"); process.exit(2); }
  const maxRecords = Number(maxArg || 150000);
  const chunkSize = Number(process.env.CHUNK || 8000);
  const Dmax = Number(process.env.DMAX || 25);
  const alpha = Number(process.env.ALPHA ?? 1);
  const poolCap = Number(process.env.POOLCAP || 512);
  const HEAD = process.env.HEAD === "bert" ? "bert" : "gpt";
  const dict = loadDict(dictArg), F = dict.size();
  const head = HEAD === "bert"
    ? { featuresOf: (r) => [...r.L, ...Array.from(r.R, (b) => b + F)], labelOf: (r) => r.curWord }
    : { featuresOf: (r) => unionSorted(r.L, r.C), labelOf: (r) => r.nextWord };

  // Stream records into memory up to the cap (bounded, no cached dataset).
  const radius = process.env.RADIUS ? Number(process.env.RADIUS) : null;
  const fuzzy = process.env.FUZZY === "1";
  const positional = process.env.POSITIONAL === "1";
  const all = []; let V = 0;
  for await (const { records, vocab } of streamSignatureChunks(dict, corpusArg, { chunkSize, radius, fuzzy, positional })) {
    for (const r of records) { if (all.length < maxRecords) all.push(r); }
    V = vocab.length;
    if (all.length >= maxRecords) break;
  }
  // prevWord (for the trigram baseline) — the word before cur, in stream order.
  // radius-1 GPT features = {prev ∪ cur}, so a word trigram P(next|prev,cur) is the
  // matching baseline; the subword version must beat IT, not just unigram/bigram.
  for (let i = 0; i < all.length; i++) all[i].prevWord = i > 0 ? all[i - 1].curWord : -1;

  const cut = Math.floor(all.length * 0.8);
  const train = all.slice(0, cut), test = all.slice(cut);
  const smooth = Number(process.env.SMOOTH || 0.1); // α: too large over big V swamps small leaves
  const cfg = { V, labelOf: head.labelOf, alpha: smooth };

  // Baselines.
  const uni = buildUnigram(train, head.labelOf);
  const big = buildContextGram(train, head.labelOf, (r) => r.curWord);
  const tri = buildContextGram(train, head.labelOf, (r) => r.prevWord * (V + 1) + r.curWord);

  const gainType = process.env.GAIN || "hb";
  // Single big tree (free growth).
  const single = buildCartTree(train, { featuresOf: head.featuresOf, labelOf: head.labelOf, sMin: 20, alpha, D: 3, Dmax, gainType });

  // Forest → pool → second-pass tree.
  const pm = new Map();
  for (let s = 0; s < train.length; s += chunkSize) {
    const t = buildCartTree(train.slice(s, s + chunkSize), { featuresOf: head.featuresOf, labelOf: head.labelOf, sMin: 20, alpha, D: 3, Dmax: 8, gainType });
    collectParts(t.root, pm);
  }
  const pool = [...pm.values()].sort((a, b) => b.count - a.count).slice(0, poolCap);
  const poolTree = buildPoolTree(train, { featuresOf: head.featuresOf, labelOf: head.labelOf, pool, sMin: 20, Dmax, alpha });

  // Smoothing is a per-model hyperparameter: α too large over big V swamps small
  // leaves. Build models ONCE, then sweep α at evaluation only (cheap vs rebuild).
  // Trees are scored by their intended readout — the path decoder, not the leaf.
  const grid = (process.env.GRID || "0.1,0.03,0.01,0.003,0.001").split(",").map(Number);
  const testEval = test.slice(0, Number(process.env.TESTCAP || 6000));
  const at = (m, a, isTree) => (isTree ? evaluatePath(m, testEval, { V, labelOf: head.labelOf, alpha: a }).ll : evaluate(m, testEval, { V, labelOf: head.labelOf, alpha: a }).ll);
  const accOf = (m) => (100 * evaluate(m, testEval, { V, labelOf: head.labelOf, alpha: 0.1 }).acc).toFixed(1);

  const models = [["unigram", uni, false], ...(HEAD === "gpt" ? [["bigram P(next|cur)", big, false], ["trigram P(next|prev,cur)", tri, false]] : []), ["single big tree", single, true], ["forest pool → 2nd-pass", poolTree, true]];
  const lls = models.map(([, m, t]) => grid.map((a) => at(m, a, t)));
  const uniBest = Math.min(...lls[0]);

  process.stdout.write(`# poolReadout — ${path.basename(corpusArg)} | ${HEAD} | radius=${radius ?? "∞"}${fuzzy ? " fuzzy" : ""} | train=${train.length} test=${testEval.length} V=${V} | pool=${pool.length}/${pm.size} | trees scored by PATH decoder\n\n`);
  process.stdout.write(`model                     ${grid.map((a) => ("α=" + a).padStart(8)).join("")}     best   acc    gate\n`);
  models.forEach(([name], i) => {
    const best = Math.min(...lls[i]);
    const cells = lls[i].map((x) => (x === best ? "*" : " ") + x.toFixed(3)).map((s) => s.padStart(8)).join("");
    const gate = i === 0 ? " ref" : best < uniBest - 1e-9 ? "pass" : "FAIL";
    process.stdout.write(`${name.padEnd(24)}${cells}  ${best.toFixed(3)}  ${accOf(models[i][1])}%  ${gate}\n`);
  });
  process.stdout.write(`\n(best = min ll over α grid; gate = model's best beats unigram's best ${uniBest.toFixed(3)})\n`);
};

main();
