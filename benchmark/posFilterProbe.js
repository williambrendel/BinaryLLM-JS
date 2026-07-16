"use strict";

// ============================================================================
// benchmark/posFilterProbe.js
//
// Hypothesis: the "different labels in the same leaf" residue is samples with few
// words on the left (early sentence positions → truncated context). Two checks on
// r5 (radius 5, GPT L∪C → nextWord):
//   (a) unmatched-vs-leaf-mode rate by target position — do early positions dominate?
//   (b) re-run purity excluding position < 5.
//
// Usage: node benchmark/posFilterProbe.js <dict> <corpus> [maxRecords]  (RADIUS via env)
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
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };
const unionSorted = (a, b) => { const out = []; let i = 0, j = 0; while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } } while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out; };
const featOf = (r) => unionSorted(r.L, r.C), labOf = (r) => r.nextWord;

const stats = (recs, name) => {
  const tree = buildCartTree(recs, { featuresOf: featOf, labelOf: labOf, sMin: 20, D: 3, Dmax: 25 });
  // leaf stats
  const leaves = [];
  (function walk(n) { if (n.part === undefined) { leaves.push(n.hist); return; } walk(n.present); walk(n.absent); })(tree.root);
  const tot = leaves.reduce((s, l) => s + l.total, 0);
  let modal = 0, wH = 0; for (const l of leaves) { let top = 0, H = 0; for (const c of l.counts.values()) { if (c > top) top = c; const p = c / l.total; H -= p * Math.log(p); } modal += top / l.total; wH += (l.total / tot) * H; }
  return { tree, leaves: tree.leaves, modal: 100 * modal / leaves.length, leafH: wH };
};

const main = async () => {
  const [dictArg, corpusArg, nArg] = process.argv.slice(2);
  const N = Number(nArg || 30000), radius = Number(process.env.RADIUS || 5);
  const dict = loadDict(dictArg);
  const all = [];
  for await (const { records } of streamSignatureChunks(dict, corpusArg, { chunkSize: 20000, radius })) { for (const r of records) if (all.length < N) all.push(r); if (all.length >= N) break; }
  const valid = all.filter((r) => labOf(r) !== NO_LABEL);

  // (a) position analysis on the full tree
  const full = stats(valid, "full");
  const argmax = (h) => { let bw = -1, bc = -1; for (const [w, c] of h.counts) if (c > bc) { bc = c; bw = w; } return bw; };
  const buckets = new Map(); // pos -> {n, unmatched}
  let mMatched = 0, nMatched = 0, mUnmatched = 0, nUnmatched = 0;
  for (const r of valid) { const leaf = full.tree.route(r); const matched = labOf(r) === argmax(leaf); const b = r.pos >= 10 ? 10 : r.pos; let e = buckets.get(b); if (!e) buckets.set(b, (e = { n: 0, u: 0 })); e.n++; if (!matched) e.u++; if (matched) { mMatched += r.pos; nMatched++; } else { mUnmatched += r.pos; nUnmatched++; } }

  const w = process.stdout;
  w.write(`# posFilterProbe — ${path.basename(corpusArg)} | r${radius} | records=${valid.length}\n\n`);
  w.write(`(a) does the impure residue skew to early positions?\n`);
  w.write(`   mean target-position:  matches-leaf-mode=${(mMatched / nMatched).toFixed(2)}   DIFFERS-from-mode=${(mUnmatched / nUnmatched).toFixed(2)}\n`);
  w.write(`   unmatched (differs-from-mode) rate by position:\n`);
  for (let p = 0; p <= 10; p++) { const e = buckets.get(p); if (!e) continue; w.write(`     pos ${p === 10 ? "10+" : p}  n=${String(e.n).padStart(5)}  differs=${(100 * e.u / e.n).toFixed(1)}%\n`); }

  // (b) re-run excluding pos < 5
  const kept = valid.filter((r) => r.pos >= 5);
  const filt = stats(kept, "pos>=5");
  w.write(`\n(b) purity with vs without early-position samples:\n`);
  w.write(`   all positions   n=${valid.length}  leaves=${full.leaves}  modal=${full.modal.toFixed(1)}%  leafH=${full.leafH.toFixed(3)}\n`);
  w.write(`   pos >= 5 only   n=${kept.length}  leaves=${filt.leaves}  modal=${filt.modal.toFixed(1)}%  leafH=${filt.leafH.toFixed(3)}\n`);
};

main();
