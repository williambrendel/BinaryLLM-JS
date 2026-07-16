"use strict";

// ============================================================================
// benchmark/leafInspect.js
//
// Build the GPT tree (L∪C → nextWord, r5), pick one moderate impure leaf, and
// print the actual sentence context for its samples: the left window + current
// word, and the (differing) next-word labels. Shows what's clustered and why the
// labels differ.
//
// Usage: node benchmark/leafInspect.js <dict> <corpus> [maxRecords]  (RADIUS via env)
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType } from "../src/core/parts/tokenize.js";
import { encodeWindowed } from "../src/core/signatures/encoder.js";
import { NO_LABEL } from "../src/core/signatures/signatureDataset.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";
import { buildCartTree } from "../src/core/predict/index.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };
const unionSorted = (a, b) => { const out = []; let i = 0, j = 0; while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } } while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out; };
const featOf = (r) => unionSorted(r.L, r.C), labOf = (r) => r.nextWord;

const [dictArg, corpusArg, nArg] = process.argv.slice(2);
const N = Number(nArg || 20000), radius = Number(process.env.RADIUS || 5);
const dict = loadDict(dictArg);
const vIdx = new Map(), vocab = [];
const wid = (w) => { let id = vIdx.get(w); if (id === undefined) { id = vocab.length; vIdx.set(w, id); vocab.push(w); } return id; };

const recs = [];
const text = fs.readFileSync(corpusArg, "utf8");
outer: for (const line of text.split("\n")) {
  for (const seg of segmentText(line)) {
    const toks = tokenizeStream(line.slice(seg.start, seg.end));
    const words = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => t.value);
    if (words.length < 2) continue;
    const sigs = encodeWindowed(dict, toks, radius);
    for (let i = 0; i < sigs.length; i++) {
      if (i < 5 || i > 8 || i + 1 >= words.length) continue; // target position 5–8, full left context
      const [L, C, R] = sigs[i];
      recs.push({ L, C, R, curWord: wid(words[i]), nextWord: wid(words[i + 1]), sent: words, at: i });
      if (recs.length >= N) break outer;
    }
  }
}
const valid = recs.filter((r) => labOf(r) !== NO_LABEL);
const tree = buildCartTree(valid, { featuresOf: featOf, labelOf: labOf, sMin: 20, D: 3, Dmax: 25 });

// group samples by leaf node
const routeLeaf = (r) => { const b = new Set(featOf(r)); let n = tree.root; while (n.part !== undefined) { let x = 0; for (const bb of n.part) if (b.has(bb)) x++; n = x >= n.k ? n.present : n.absent; } return n; };
const groups = new Map();
for (const r of valid) { const leaf = routeLeaf(r); let g = groups.get(leaf); if (!g) groups.set(leaf, (g = [])); g.push(r); }

// per-leaf path composition (contains vs lacks conditions)
const leafMeta = new Map();
(function walk(nd, nC, nL) { if (nd.part === undefined) { leafMeta.set(nd, { nC, nL }); return; } walk(nd.present, nC + 1, nL); walk(nd.absent, nC, nL + 1); })(tree.root, 0, 0);

// is r5's impurity flat across leaves? modal-purity distribution over leaves (size>=20)
const distL = [];
for (const [node, gg] of groups) { if (gg.length < 20) continue; let top = 0; const c = new Map(); for (const r of gg) c.set(labOf(r), (c.get(labOf(r)) || 0) + 1); for (const v of c.values()) if (v > top) top = v; distL.push(100 * top / gg.length); }
distL.sort((a, b) => a - b);
const dmean = distL.reduce((a, b) => a + b, 0) / distL.length;
const dstd = Math.sqrt(distL.reduce((a, b) => a + (b - dmean) ** 2, 0) / distL.length);
process.stdout.write(`leaf modal-purity across ${distL.length} leaves (size>=20): mean=${dmean.toFixed(1)}%  std=${dstd.toFixed(1)}  min=${distL[0].toFixed(0)}%  p50=${distL[distL.length >> 1].toFixed(0)}%  max=${distL[distL.length - 1].toFixed(0)}%\n\n`);

// pick the leaf with the MOST shared parts (highest contains-fraction), size >= 30
let best = null;
for (const [node, gg] of groups) { if (gg.length < 30) continue; const m = leafMeta.get(node); const frac = m.nC + m.nL ? m.nC / (m.nC + m.nL) : 0; const d = new Set(gg.map(labOf)).size; if (!best || frac > best.frac) best = { g: gg, d, frac, nC: m.nC, nL: m.nL }; }

const g = best.g;
const cnt = new Map(); for (const r of g) cnt.set(labOf(r), (cnt.get(labOf(r)) || 0) + 1);
let topW = -1, topC = 0; for (const [wv, c] of cnt) if (c > topC) { topC = c; topW = wv; }
const w = process.stdout;

// Do the leaf's samples actually have SIMILAR signatures? Within-leaf pairwise
// Jaccard vs random-pair Jaccard, and the path (conjunction of coverage tests)
// that routes them here.
const jac = (a, b) => { let i = 0, j = 0, x = 0; while (i < a.length && j < b.length) { if (a[i] < b[j]) i++; else if (a[i] > b[j]) j++; else { x++; i++; j++; } } const u = a.length + b.length - x; return u ? x / u : 0; };
const feats = g.map(featOf);
const rnd = (n) => Math.floor(Math.random() * n);
let inJ = 0; for (let s = 0; s < 3000; s++) { let i = rnd(g.length), j = rnd(g.length); while (j === i) j = rnd(g.length); inJ += jac(feats[i], feats[j]); } inJ /= 3000;
let rJ = 0; for (let s = 0; s < 3000; s++) { rJ += jac(featOf(valid[rnd(valid.length)]), featOf(valid[rnd(valid.length)])); } rJ /= 3000;
const meanSig = feats.reduce((s, f) => s + f.length, 0) / g.length;
// path taken by a representative sample
const b0 = new Set(featOf(g[0]));
let n = tree.root; const conds = [];
while (n.part !== undefined) { let x = 0; for (const bb of n.part) if (b0.has(bb)) x++; const fired = x >= n.k; conds.push({ bits: n.part, k: n.k, fired }); n = fired ? n.present : n.absent; }
const nFire = conds.filter((p) => p.fired).length;

w.write(`# leafInspect — MOST-SHARED-PARTS leaf (pos 5–8) | contains-frac=${best.frac.toFixed(2)} (${best.nC} contains / ${best.nL} lacks)  size=${g.length}  distinct=${best.d}  modal="${vocab[topW]}" (${topC}, ${(100 * topC / g.length).toFixed(0)}%)\n`);
w.write(`  mean |sig| = ${meanSig.toFixed(1)} bits\n`);
w.write(`  within-leaf pairwise Jaccard = ${inJ.toFixed(3)}   vs random-pair = ${rJ.toFixed(3)}   (ratio ${(inJ / rJ).toFixed(1)}×)\n`);
w.write(`  path length = ${conds.length} splits  (${nFire} "contains" / ${conds.length - nFire} "lacks")\n`);
w.write(`  path: ` + conds.map((p) => `${p.fired ? "" : "¬"}{${p.bits.join(",")}}≥${p.k}`).join(" ∧ ") + `\n`);
w.write(`  format:  … left-window …  [CURRENT] → NEXT-label\n\n`);
for (const r of g.slice(0, 28)) {
  const lo = Math.max(0, r.at - 5);
  const left = r.sent.slice(lo, r.at).join(" ");
  w.write(`  …${left.padStart(46).slice(-46)}  [${r.sent[r.at]}] → ${r.sent[r.at + 1]}\n`);
}
