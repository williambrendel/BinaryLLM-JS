"use strict";

// ============================================================================
// benchmark/poolCompare.js
//
// Forest-of-N-trees vs one-big-tree, over the SAME data: which discovers the
// better POOL of parts? Stream N chunks; per chunk grow one tree and union its
// parts (the forest pool); also accumulate the chunks and grow ONE big tree over
// all of them (the single pool). Compare pool size, distinct bits covered, part
// width, redundancy, and overlap. (The single tree must hold all data in memory —
// exactly the cost the forest avoids; that asymmetry is part of the answer.)
//
// Usage: node benchmark/poolCompare.js <dict> <corpus> [N]
//   CHUNK=8000 CHUNK_DMAX=8 BIG_DMAX=25 ALPHA=1 HEAD=gpt
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { streamSignatureChunks } from "../src/core/signatures/streamSignatures.js";
import { buildCartTree } from "../src/core/predict/index.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, dir) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(dir, p) : p);
const loadDict = (arg) => { const p = resolveIn(arg, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const unionSorted = (a, b) => {
  const out = []; let i = 0, j = 0;
  while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } }
  while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out;
};

// Collect distinct parts (keyed by sorted bit-set) from a tree into `pool`.
const collectParts = (node, pool, bits) => {
  if (node.part === undefined) return;
  const key = node.part.slice().sort((a, b) => a - b).join(",");
  const e = pool.get(key); if (e) e.count++; else pool.set(key, { bits: node.part, k: node.k, count: 1 });
  for (const b of node.part) bits.add(b);
  collectParts(node.present, pool, bits); collectParts(node.absent, pool, bits);
};

const summarize = (pool, bits, name) => {
  let instances = 0, widthSum = 0;
  for (const e of pool.values()) { instances += e.count; widthSum += e.bits.length; }
  return { name, distinct: pool.size, instances, bits: bits.size, meanWidth: (widthSum / pool.size).toFixed(2), redundancy: (instances / pool.size).toFixed(2) };
};

const main = async () => {
  const [dictArg, corpusArg, nArg] = process.argv.slice(2);
  if (!dictArg || !corpusArg) { process.stderr.write("usage: node benchmark/poolCompare.js <dict> <corpus> [N]\n"); process.exit(2); }
  const N = Number(nArg || 25);
  const chunkSize = Number(process.env.CHUNK || 8000);
  const chunkDmax = Number(process.env.CHUNK_DMAX || 8);
  const bigDmax = Number(process.env.BIG_DMAX || 25);
  const alpha = Number(process.env.ALPHA ?? 1);
  const HEAD = process.env.HEAD === "bert" ? "bert" : "gpt";
  const dict = loadDict(dictArg);
  const F = dict.size();
  // GPT = L∪C (union, F). BERT = [L|R] concatenated (2F): R offset by F, direction-aware.
  const head = HEAD === "bert"
    ? { featuresOf: (r) => [...r.L, ...Array.from(r.R, (b) => b + F)], labelOf: (r) => r.curWord }
    : { featuresOf: (r) => unionSorted(r.L, r.C), labelOf: (r) => r.nextWord };

  const forestPool = new Map(), forestBits = new Set();
  const all = [];
  let n = 0;
  for await (const { records } of streamSignatureChunks(dict, corpusArg, { chunkSize })) {
    const tree = buildCartTree(records, { featuresOf: head.featuresOf, labelOf: head.labelOf, sMin: 20, alpha, D: 3, Dmax: chunkDmax });
    collectParts(tree.root, forestPool, forestBits);
    for (const r of records) all.push(r);
    if (++n >= N) break;
  }

  const bigTree = buildCartTree(all, { featuresOf: head.featuresOf, labelOf: head.labelOf, sMin: 20, alpha, D: 3, Dmax: bigDmax });
  const singlePool = new Map(), singleBits = new Set();
  collectParts(bigTree.root, singlePool, singleBits);

  const f = summarize(forestPool, forestBits, `forest (${n}×${chunkSize}, Dmax ${chunkDmax})`);
  const s = summarize(singlePool, singleBits, `single (1×${all.length}, Dmax ${bigDmax})`);

  // overlap
  let sharedParts = 0; for (const k of forestPool.keys()) if (singlePool.has(k)) sharedParts++;
  let sharedBits = 0; for (const b of forestBits) if (singleBits.has(b)) sharedBits++;
  const jac = sharedBits / (forestBits.size + singleBits.size - sharedBits);

  process.stdout.write(`# poolCompare — ${path.basename(corpusArg)} | ${HEAD} | α=${alpha} | total records=${all.length}\n\n`);
  process.stdout.write(`pool                              distinct  instances  bits   meanWidth  redundancy\n`);
  for (const r of [f, s]) process.stdout.write(`${r.name.padEnd(32)}  ${String(r.distinct).padStart(8)}  ${String(r.instances).padStart(9)}  ${String(r.bits).padStart(5)}  ${String(r.meanWidth).padStart(9)}  ${String(r.redundancy).padStart(10)}\n`);
  process.stdout.write(`\noverlap: ${sharedParts} shared distinct parts (${(100 * sharedParts / Math.min(forestPool.size, singlePool.size)).toFixed(1)}% of smaller)  |  bit Jaccard=${jac.toFixed(3)} (${sharedBits} shared bits)\n`);
};

main();
