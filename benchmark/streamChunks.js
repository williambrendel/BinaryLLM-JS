"use strict";

// ============================================================================
// benchmark/streamChunks.js
//
// Stream a corpus in bounded chunks (signatures encoded on the fly, never cached),
// grow ONE tree per chunk, and report per-chunk tree stats + live heap — proving
// the footprint stays flat no matter how big the corpus is. Also aggregates the
// terminal-reason breakdown across chunks: the "pure leaves on big wiki?" answer
// at scale, without ever materializing a full-corpus dataset.
//
// Usage:
//   node benchmark/streamChunks.js <dict.txt|.bin> <corpus> [maxChunks]
//     CHUNK=20000 DMAX=25 ALPHA=1 HEAD=gpt
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { streamSignatureChunks } from "../src/core/signatures/streamSignatures.js";
import { buildCartTree } from "../src/core/predict/index.js";
import { NO_LABEL } from "../src/core/signatures/signatureDataset.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");

const unionSorted = (a, b) => {
  const out = []; let i = 0, j = 0;
  while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } }
  while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out;
};

const resolveIn = (p, dir) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(dir, p) : p);
const loadDict = (arg) => {
  const p = resolveIn(arg, DICT_DIR);
  return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1"));
};

const main = async () => {
  const [dictArg, corpusArg, maxArg] = process.argv.slice(2);
  if (!dictArg || !corpusArg) { process.stderr.write("usage: node benchmark/streamChunks.js <dict> <corpus> [maxChunks]\n"); process.exit(2); }
  const maxChunks = Number(maxArg || 20);
  const chunkSize = Number(process.env.CHUNK || 20000);
  const Dmax = Number(process.env.DMAX || 25);
  const alpha = Number(process.env.ALPHA ?? 1);
  const HEAD = process.env.HEAD === "bert" ? "bert" : "gpt";
  const dict = loadDict(dictArg);
  const F = dict.size();
  // GPT = L∪C (union, F). BERT = [L|R] concatenated (2F): R offset by F, direction-aware.
  const head = HEAD === "bert"
    ? { featuresOf: (r) => [...r.L, ...Array.from(r.R, (b) => b + F)], labelOf: (r) => r.curWord }
    : { featuresOf: (r) => unionSorted(r.L, r.C), labelOf: (r) => r.nextWord };
  const mb = () => (process.memoryUsage().heapUsed / 1048576).toFixed(0);
  process.stdout.write(`# streamChunks — ${path.basename(corpusArg)} | ${HEAD} | chunk=${chunkSize} Dmax=${Dmax} α=${alpha} | dict F=${dict.size()}\n`);
  process.stdout.write(`chunk  records  leaves  depth   pure  smin  depth-cap nosplit   heapMB  vocab\n`);

  const agg = { pure: 0, smin: 0, depth: 0, nosplit: 0, leaves: 0 };
  let n = 0;
  for await (const { records, vocab } of streamSignatureChunks(dict, corpusArg, { chunkSize })) {
    n++;
    const tree = buildCartTree(records, { featuresOf: head.featuresOf, labelOf: head.labelOf, sMin: 20, alpha, D: 3, Dmax });
    const depth = (function d(node) { return node.part === undefined ? 0 : 1 + Math.max(d(node.present), d(node.absent)); })(tree.root);
    const tr = tree.terminalReasons;
    for (const k of ["pure", "smin", "depth", "nosplit"]) agg[k] += tr[k];
    agg.leaves += tree.leaves;
    process.stdout.write(
      `${String(n).padStart(4)}  ${String(records.length).padStart(7)}  ${String(tree.leaves).padStart(6)}  ${String(depth).padStart(4)}   ${String(tr.pure).padStart(5)} ${String(tr.smin).padStart(5)} ${String(tr.depth).padStart(8)} ${String(tr.nosplit).padStart(7)}   ${mb().padStart(6)}  ${String(vocab.length).padStart(6)}\n`,
    );
    if (n >= maxChunks) break;
  }
  const pct = (x) => (100 * x / agg.leaves).toFixed(1);
  process.stdout.write(`\nAGG over ${n} chunks, ${agg.leaves} leaves:  pure=${agg.pure} (${pct(agg.pure)}%)  smin=${agg.smin} (${pct(agg.smin)}%)  depth=${agg.depth} (${pct(agg.depth)}%)  nosplit=${agg.nosplit} (${pct(agg.nosplit)}%)\n`);
};

main();
