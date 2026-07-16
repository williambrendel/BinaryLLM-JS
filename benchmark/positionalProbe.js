"use strict";

// ============================================================================
// benchmark/positionalProbe.js
//
// Does keeping POSITION (per-offset subspaces) restore peelability at a wide
// window? Streams the same corpus slice two ways at the same radius — OR-pooled
// vs position-preserving — builds a tree on each (default IG/α=0 grower), and
// compares purity + total IG. If positional r5 climbs toward r1's numbers, the
// greedy landscape was reshaped: peels become the local optimum again.
//
// Usage: node benchmark/positionalProbe.js <dict> <corpus> [maxRecords]
//   RADIUS=5 CHUNK=20000 SMIN=20 DMAX=25
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
const resolveIn = (p, dir) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(dir, p) : p);
const loadDict = (arg) => { const p = resolveIn(arg, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };
const unionSorted = (a, b) => { const out = []; let i = 0, j = 0; while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } } while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out; };
const featOf = (r) => unionSorted(r.L, r.C);
const labOf = (r) => r.nextWord;

const collectN = async (dict, corpus, opts, n) => { const all = []; for await (const { records } of streamSignatureChunks(dict, corpus, opts)) { for (const r of records) if (all.length < n) all.push(r); if (all.length >= n) break; } return all; };

const leafStats = (node, acc) => {
  if (node.part === undefined) { const size = node.hist.total; let top = 0, H = 0, dist = node.hist.counts.size; for (const c of node.hist.counts.values()) { if (c > top) top = c; const p = c / size; H -= p * Math.log(p); } acc.push({ size, dist, top, H }); return; }
  leafStats(node.present, acc); leafStats(node.absent, acc);
};
const depthOf = (n) => (n.part === undefined ? 0 : 1 + Math.max(depthOf(n.present), depthOf(n.absent)));

const summarize = (records, name, out) => {
  const tree = buildCartTree(records, { featuresOf: featOf, labelOf: labOf, sMin: Number(process.env.SMIN || 20), D: 3, Dmax: Number(process.env.DMAX || 25) });
  const L = []; leafStats(tree.root, L);
  const tot = L.reduce((s, l) => s + l.size, 0);
  const modal = 100 * L.reduce((s, l) => s + l.top / l.size, 0) / L.length;
  const sizeOverDist = L.reduce((s, l) => s + l.size, 0) / L.reduce((s, l) => s + l.dist, 0);
  const wH = L.reduce((s, l) => s + l.size * l.H, 0) / tot;
  let Hr = 0; const rc = new Map(); let rn = 0; for (const r of records) { const l = labOf(r); if (l === NO_LABEL) continue; rc.set(l, (rc.get(l) || 0) + 1); rn++; } for (const c of rc.values()) { const p = c / rn; Hr -= p * Math.log(p); }
  const meanSig = records.reduce((s, r) => s + featOf(r).length, 0) / records.length;
  out.push(`${name.padEnd(20)}  leaves=${String(tree.leaves).padStart(4)}  depth=${String(depthOf(tree.root)).padStart(2)}  meanSig=${meanSig.toFixed(1).padStart(5)}  size/dist=${sizeOverDist.toFixed(2)}  modal=${modal.toFixed(1)}%  leafH=${wH.toFixed(3)}  totalIG=${(Hr - wH).toFixed(3)}`);
};

const main = async () => {
  const [dictArg, corpusArg, nArg] = process.argv.slice(2);
  const N = Number(nArg || 20000);
  const radius = Number(process.env.RADIUS || 5);
  const chunkSize = Number(process.env.CHUNK || 20000);
  const dict = loadDict(dictArg);
  const pooled = await collectN(dict, corpusArg, { chunkSize, radius }, N);
  const positional = await collectN(dict, corpusArg, { chunkSize, radius, positional: true }, N);
  const out = [];
  out.push(`# positionalProbe — ${path.basename(corpusArg)} | radius=${radius} | records=${pooled.length} | grower=IG/α0`);
  summarize(pooled, `pooled r${radius}`, out);
  summarize(positional, `positional r${radius}`, out);
  process.stdout.write(out.join("\n") + "\n");
};

main();
