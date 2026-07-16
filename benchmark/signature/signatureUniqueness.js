"use strict";

// ============================================================================
// benchmark/signatureUniqueness.js
//
// THE design-validating benchmark. Measures, for three word signatures:
//   - F        = fuzzy containment set (the NN key),
//   - [L,R]    = the peel's Start + End ids (boundary structure),
//   - [L∪C]    = the GPT-side union: Start ∪ Mids/fills ids,
// two things:
//   (a) UNIQUENESS at rest — fraction of distinct words whose signature is not
//       shared by any other word (collision-free);
//   (b) ROBUSTNESS under injected character corruption (0.5% … 25%):
//       * F   → NN recovery rate (corrupted F still resolves to the original);
//       * [L,R] → exact-stable rate (boundary ids unchanged);
//       * [L∪C] → mean Jaccard(original, corrupted).
//
// Expected ordering: F ≫ [L∪C] > [L,R]. That is *why* F is the fuzzy retrieval
// key and the peel runs on canonical words — the redundant containment tolerates
// noise the non-redundant peel does not.
//
// Usage: node benchmark/signatureUniqueness.js [corpus...]  (default: wiki.test)
// ============================================================================

import fs from "node:fs";
import { Kind } from "../../src/core/parts/kind.js";
import { wordFreq } from "../../src/utilities/corpus/wordFreq.js";
import { generateParts } from "../../src/core/parts/generation/index.js";
import { addBigramBackstop, fuzzyEncode, bpeEncode } from "../../src/core/parts/encoding/index.js";
import { buildInvertedIndex, nearestNeighbor } from "../../src/core/parts/normalization/index.js";
import { jaccard } from "../../src/core/math/sparse/jaccard.js";

const paths = process.argv.slice(2).length ? process.argv.slice(2) : ["data/corpora/wiki.test.txt"];
const bufs = paths.map((p) => new Uint8Array(fs.readFileSync(p)));
const freq = wordFreq(bufs);
const { dict, tracks } = generateParts(freq);
addBigramBackstop(dict);
const words = [...freq.keys()];
const index = buildInvertedIndex(words, words.map((x) => fuzzyEncode(dict, x)));

// Signature extractors → { LRkey (string), luc (sorted Uint32Array) }.
const structural = (x) => {
  const e = bpeEncode(dict, x, tracks);
  if (e.whole) return { LRkey: "W" + e.whole.id, lrIds: Uint32Array.of(e.whole.id), luc: Uint32Array.of(e.whole.id) };
  const lr = `${e.start ? e.start.id : "_"}|${e.end ? e.end.id : "_"}`;
  const lrSet = []; if (e.start) lrSet.push(e.start.id); if (e.end) lrSet.push(e.end.id);
  const set = new Set(); if (e.start) set.add(e.start.id); for (const c of e.mid) set.add(c.id);
  return { LRkey: lr, lrIds: Uint32Array.from([...new Set(lrSet)].sort((a, b) => a - b)), luc: Uint32Array.from([...set].sort((a, b) => a - b)) };
};

// Deterministic corruption.
let seed = 0x9e3779b9;
const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const subst = (x, p) => { let o = ""; for (const c of x) { if (rnd() < p) { let d; do { d = String.fromCharCode(97 + Math.floor(rnd() * 26)); } while (d === c); o += d; } else o += c; } return o; };

const sampleIdx = (() => { const out = []; const stride = Math.max(1, Math.floor(words.length / 3000)); for (let i = 0; i < words.length && out.length < 3000; i += stride) if (words[i].length >= 4) out.push(i); return out; })();
const w = process.stdout;
w.write(`\n# Signature uniqueness & robustness — ${paths.join(", ")}  (${words.length} words, sample ${sampleIdx.length})\n`);

// ---- (a) uniqueness at rest over the FULL vocabulary -----------------------
const fCnt = new Map(), lrCnt = new Map(), lucCnt = new Map();
for (const x of words) {
  const s = structural(x);
  const fk = fuzzyEncode(dict, x).join(",");
  fCnt.set(fk, (fCnt.get(fk) || 0) + 1);
  lrCnt.set(s.LRkey, (lrCnt.get(s.LRkey) || 0) + 1);
  lucCnt.set(s.luc.join(","), (lucCnt.get(s.luc.join(",")) || 0) + 1);
}
const uniqRate = (map) => { let u = 0, n = 0; for (const c of map.values()) { n += c; if (c === 1) u += c; } return 100 * u / n; };
w.write(`\n(a) uniqueness at rest (collision-free % of words):\n`);
w.write(`    F = ${uniqRate(fCnt).toFixed(1)}%   [L,R] = ${uniqRate(lrCnt).toFixed(1)}%   [L∪C] = ${uniqRate(lucCnt).toFixed(1)}%\n`);

// ---- (b) robustness under corruption ---------------------------------------
// Two views of the peel signatures:
//   RAW    = encode the corrupted word directly (no correction — worst case).
//   via-NN = canonicalize the corrupted word through the index first, then
//            encode (the REAL pipeline: the peel runs on the canonical word).
// The via-NN columns should track F recovery — showing the NN *restores* the
// brittle peel, which is the whole point of putting robustness upstream.
// Same lens for all three signatures (each is an id-set → mean JACCARD), shown
// two ways:
//   RAW    = encode the corrupted word directly — the INTRINSIC robustness,
//            where F (redundant) diverges from the peel (brittle).
//   via-NN = canonicalize first, then encode — the REAL pipeline, where the NN
//            restores the peel (all three converge on the recovery rate).
// F also reports NN recovery % (its operational role as the retrieval key).
const THETA = 0.5;
const jac = (a, b) => (a.length && b.length ? jaccard(a, b) : a.length === b.length ? 1 : 0);
const rows = [0.005, 0.05, 0.10, 0.15, 0.20, 0.25].map((p) => {
  let fJ = 0, lrRaw = 0, lucRaw = 0, fRec = 0, lrNN = 0, lucNN = 0, n = 0;
  for (const wi of sampleIdx) {
    const orig = words[wi], mut = subst(orig, p); n++;
    const a = structural(orig), Fo = fuzzyEncode(dict, orig);
    if (mut === orig) { fJ += 1; lrRaw += 1; lucRaw += 1; fRec++; lrNN += 1; lucNN += 1; continue; }
    const Fm = fuzzyEncode(dict, mut), raw = structural(mut);
    fJ += jac(Fo, Fm); lrRaw += jac(a.lrIds, raw.lrIds); lucRaw += jac(a.luc, raw.luc);
    const nn = nearestNeighbor(Fm, index);
    const canon = nn.wi >= 0 && nn.j >= THETA ? words[nn.wi] : mut; // NN + gate
    const c = structural(canon);
    if (canon === orig) fRec++;
    lrNN += jac(a.lrIds, c.lrIds); lucNN += jac(a.luc, c.luc);
  }
  return { p, fJ: fJ / n, lrRaw: lrRaw / n, lucRaw: lucRaw / n, fRec: 100 * fRec / n, lrNN: lrNN / n, lucNN: lucNN / n };
});
w.write(`\n(b) RAW mean Jaccard — encode corrupted directly (intrinsic robustness):\n`);
w.write(`| error % | F | [L,R] | [L∪C] |\n|---|---|---|---|\n`);
for (const r of rows) w.write(`| ${(100 * r.p).toFixed(1)} | ${r.fJ.toFixed(3)} | ${r.lrRaw.toFixed(3)} | ${r.lucRaw.toFixed(3)} |\n`);
w.write(`\n(c) via-NN — real pipeline, θ=${THETA} (F = NN recovery %; peel = mean Jaccard):\n`);
w.write(`| error % | F recovery % | [L,R] J | [L∪C] J |\n|---|---|---|---|\n`);
for (const r of rows) w.write(`| ${(100 * r.p).toFixed(1)} | ${r.fRec.toFixed(1)} | ${r.lrNN.toFixed(3)} | ${r.lucNN.toFixed(3)} |\n`);
