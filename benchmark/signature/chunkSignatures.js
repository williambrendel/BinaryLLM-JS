"use strict";

// ============================================================================
// benchmark/chunkSignatures.js
//
// The chunk-contextual [L,C,R] signature (the CORRECT model — see
// src/core/parts/signature/). Partitions the corpus into chunks at two levels
// (PARAGRAPHS and SENTENCES), computes [L,C,R] per word via the library's
// buildChunkSignatures, and reports:
//   (1) a demo chunk with |L| |C| |R| per word;
//   (2) COLLISION rates per signature — C / L / R / [L,R] / [L∪C] / [L,C,R] —
//       (uniqueness = % of positions whose signature no other position shares);
//   (3) PREDICTION consistency — the signatures that exist to predict a word
//       are judged by whether the context DETERMINES its target, not by bare
//       uniqueness. [L∪C] (GPT input) predicts the NEXT word; [L,R] (cloze
//       context) predicts the CURRENT word. A collision only hurts when the
//       same context maps to a *different* target; same context → same target
//       is benign. So we report, per shared context, how often the target agrees.
//   (4) [L∪C] collision examples showing the two OR-pooling effects:
//       order-invariance (cross-chunk) and prefix saturation (within-chunk).
//
// Usage: node benchmark/chunkSignatures.js [corpus...]  (default: English novels)
// ============================================================================

import fs from "node:fs";
import { wordFreq } from "../../src/utilities/corpus/wordFreq.js";
import { generateParts } from "../../src/core/parts/generation/index.js";
import { addBigramBackstop, bpeEncode, fuzzyEncode } from "../../src/core/parts/encoding/index.js";
import { buildChunkSignatures } from "../../src/core/parts/signature/index.js";
import { tokenizeStream, StreamTokenType } from "../../src/core/parts/tokenize.js";
import { or } from "../../src/core/math/sparse/or.js";

// Windowed causal union — OR of C over the last W words (incl. current). The
// OR-pool analogue of local attention; used to show the cap removes saturation.
const unionU32 = (a, b) => Uint32Array.from(or(a, b));
const windowLC = (C, i, W) => { let u = new Uint32Array(0); for (let j = Math.max(0, i - W + 1); j <= i; j++) u = unionU32(u, C[j]); return u; };
const rangeOr = (C, lo, hi) => { let u = new Uint32Array(0); for (let j = Math.max(0, lo); j <= Math.min(C.length - 1, hi); j++) u = unionU32(u, C[j]); return u; };
// Fixed-size window centred at i, radius r, shifted inward to stay full at edges.
const shiftWin = (i, k, r) => { if (k <= 2 * r + 1) return [0, k - 1]; let lo = i - r, hi = i + r; if (lo < 0) { hi -= lo; lo = 0; } else if (hi > k - 1) { lo -= hi - (k - 1); hi = k - 1; } return [lo, hi]; };
const CONTEXT_WINDOWS = [3, 5, 7];

const paths = process.argv.slice(2).length ? process.argv.slice(2) : ["data/corpora/alice_en.txt", "data/corpora/pride_en.txt"];
const texts = paths.map((p) => fs.readFileSync(p, "utf8"));
const enc = new TextEncoder();
const freq = wordFreq(texts.map((t) => enc.encode(t)));
const { dict } = generateParts(freq);
addBigramBackstop(dict);

// Encoder is configurable for testing: ENCODER=sparse (default) | fuzzy.
const ENCODER = process.env.ENCODER === "fuzzy" ? "fuzzy" : "sparse";
const encodeC = ENCODER === "fuzzy"
  ? (w) => [...new Set(fuzzyEncode(dict, w))].sort((a, b) => a - b).join(",")
  : (w) => [...new Set(bpeEncode(dict, w).parts.map((p) => p.id))].sort((a, b) => a - b).join(",");
// C uniqueness at the TYPE level: does a *different word* share the same sparse C?
// (Per token position C collides constantly — but that's just common words
// repeating, not a signature collision, so it is measured over the vocabulary.)
const vocab = [...freq.keys()];
const cByType = new Map();
for (const wd of vocab) { const k = encodeC(wd); cByType.set(k, (cByType.get(k) || 0) + 1); }
const cTypeUniq = 100 * [...cByType.values()].filter((c) => c === 1).length / vocab.length;
const wordsOf = (chunk) => [...tokenizeStream(enc.encode(chunk))].filter((t) => t.type === StreamTokenType.Word).map((t) => t.value);

const LEVELS = [
  ["PARAGRAPH", (t) => t.split(/\n\s*\n/)],
  ["SENTENCE", (t) => t.replace(/\s+/g, " ").split(/(?<=[.!?])\s+/)],
];
const TYPES = ["C", "L", "R", "[L,R]", "[L∪C]", "[L,C,R]"];
const keyOf = (s) => ({
  C: s.C.join(","),
  L: s.L.join(","),
  R: s.R.join(","),
  "[L,R]": `${s.L.join(",")}|${s.R.join(",")}`,
  "[L∪C]": s.LC.join(","),
  "[L,C,R]": `${s.L.join(",")}|${s.C.join(",")}|${s.R.join(",")}`,
});

// Target-determinism for a predictive signature: `nested` maps context-key →
// Map(target → count). A context "determines" its target when it only ever
// occurs with a single distinct target; a collision (context shared by ≥2
// positions) is BENIGN when the target still agrees, HARMFUL when it differs.
const consistency = (nested) => {
  let N = 0, determined = 0, colliding = 0, benign = 0;
  for (const targets of nested.values()) {
    let n = 0; for (const c of targets.values()) n += c;
    N += n;
    if (targets.size === 1) determined += n;
    if (n >= 2) { colliding += n; if (targets.size === 1) benign += n; }
  }
  return {
    distinct: nested.size, N,
    determinedPct: 100 * determined / N,       // context ⇒ unique target (incl. trivially-unique contexts)
    collidePct: 100 * colliding / N,           // context shared by ≥2 positions
    benignOfCollide: colliding ? 100 * benign / colliding : 100, // of those, target still agrees
  };
};
const bumpNested = (m, k, t) => { let tm = m.get(k); if (!tm) m.set(k, (tm = new Map())); tm.set(t, (tm.get(t) || 0) + 1); };

const w = process.stdout;
w.write(`# Chunk-contextual [L,C,R] — ${paths.map((p) => p.split("/").pop()).join(", ")}  [encoder=${ENCODER}]\n`);

for (const [name, split] of LEVELS) {
  const chunks = texts.flatMap(split).map((c) => c.replace(/\s+/g, " ").trim()).filter((c) => wordsOf(c).length >= 2);
  w.write(`\n\n======================== ${name} level (${chunks.length} chunks) ========================\n`);

  // (1) demo
  const demoChunk = chunks.find((c) => { const n = wordsOf(c).length; return n >= 5 && n <= 12; }) || chunks[0];
  const demoSig = buildChunkSignatures(wordsOf(demoChunk), { dict, encoder: ENCODER });
  w.write(`\n(1) demo — “${demoChunk.slice(0, 150)}”\n    ${demoSig.map((s) => `${s.word}(L${s.L.length}·C${s.C.length}·R${s.R.length})`).join(" ")}\n`);

  // (2)+(3)+(4): per-signature collision counts, prediction consistency, and
  // [L∪C] example groups — all in one pass.
  const counts = Object.fromEntries(TYPES.map((t) => [t, new Map()]));
  const lcNext = new Map();   // [L∪C] key → Map(nextWord → count)   (GPT: predict next word)
  const lrWord = new Map();   // [L,R]  key → Map(currWord → count)   (cloze: predict current word)
  const lcGroups = new Map(); // [L∪C] key → position records (for examples)
  const lrGroups = new Map(); // [L,R]  key → position records (for examples)
  const lcNextW = Object.fromEntries(CONTEXT_WINDOWS.map((W) => [W, new Map()])); // windowed [L∪C] → next word
  const lcByLenU = new Map(); // unbounded [L∪C] key → [prefix len i]  (next-word positions)
  const lcByLenW = new Map(); // window=5  [L∪C] key → [prefix len i]
  let positions = 0, lcPositions = 0;
  for (const chunk of chunks) {
    const words = wordsOf(chunk), sigs = buildChunkSignatures(words, { dict, encoder: ENCODER });
    const C = sigs.map((s) => s.C);
    for (let i = 0; i < sigs.length; i++) {
      positions++;
      const k = keyOf(sigs[i]);
      for (const t of TYPES) counts[t].set(k[t], (counts[t].get(k[t]) || 0) + 1);
      // [L,R] → current word (every position has a current word).
      bumpNested(lrWord, k["[L,R]"], sigs[i].word);
      let lr = lrGroups.get(k["[L,R]"]); if (!lr) lrGroups.set(k["[L,R]"], (lr = [])); lr.push({ chunk, i, word: sigs[i].word, words });
      // [L∪C] → next word (only positions that HAVE a next word).
      let a = lcGroups.get(k["[L∪C]"]); if (!a) lcGroups.set(k["[L∪C]"], (a = []));
      if (i < sigs.length - 1) {
        lcPositions++;
        const next = sigs[i + 1].word;
        bumpNested(lcNext, k["[L∪C]"], next);
        a.push({ chunk, i, word: sigs[i].word, words, next });
        // Windowed [L∪C] → next word, plus prefix-length collision keys.
        for (const W of CONTEXT_WINDOWS) {
          const wk = windowLC(C, i, W).join(",");
          bumpNested(lcNextW[W], wk, next);
          if (W === 5) { let g = lcByLenW.get(wk); if (!g) lcByLenW.set(wk, (g = [])); g.push(i); }
        }
        let gu = lcByLenU.get(k["[L∪C]"]); if (!gu) lcByLenU.set(k["[L∪C]"], (gu = [])); gu.push(i);
      } else {
        a.push({ chunk, i, word: sigs[i].word, words });
      }
    }
  }
  const uniquePct = (map) => { let u = 0; for (const c of map.values()) if (c === 1) u++; return 100 * u / positions; };
  w.write(`\n(2) collision rates:\n`);
  w.write(`  C  (per distinct WORD — does a *different* word share the same C=F?): ${cTypeUniq.toFixed(1)}% unique of ${vocab.length} words.\n`);
  w.write(`     (per token position C is only ${uniquePct(counts["C"]).toFixed(1)}% — that's the same common words repeating, NOT a signature collision.)\n`);
  w.write(`  Context signatures (per POSITION over ${positions} — does another position share this context?):\n`);
  w.write(`| signature | distinct | unique % |\n|---|---|---|\n`);
  for (const t of ["L", "R", "[L,R]", "[L∪C]", "[L,C,R]"]) w.write(`| ${t} | ${counts[t].size} | ${uniquePct(counts[t]).toFixed(1)} |\n`);

  // (3) prediction consistency — does the context DETERMINE its target?
  const cLC = consistency(lcNext), cLR = consistency(lrWord);
  w.write(`\n(3) prediction consistency (does the context determine its target?):\n`);
  w.write(`| context → target | positions | distinct ctx | determines target % | shares ctx % | …of which target agrees % |\n|---|---|---|---|---|---|\n`);
  w.write(`| [L∪C] → next word | ${cLC.N} | ${cLC.distinct} | ${cLC.determinedPct.toFixed(1)} | ${cLC.collidePct.toFixed(1)} | ${cLC.benignOfCollide.toFixed(1)} |\n`);
  w.write(`| [L,R] → current word | ${cLR.N} | ${cLR.distinct} | ${cLR.determinedPct.toFixed(1)} | ${cLR.collidePct.toFixed(1)} | ${cLR.benignOfCollide.toFixed(1)} |\n`);

  // Examples of HARMFUL collisions (same context, different target).
  const win2 = (words, i) => { const a = Math.max(0, i - 3), b = Math.min(words.length, i + 4); return (a ? "…" : "") + words.slice(a, b).map((x, j) => (a + j === i ? `⟦${x}⟧` : x)).join(" ") + (b < words.length ? "…" : ""); };
  const lcConflict = [...lcGroups.values()].filter((g) => new Set(g.filter((e) => e.next != null).map((e) => e.next)).size >= 2).sort((a, b) => a.length - b.length);
  for (const g of lcConflict.slice(0, 2)) {
    w.write(`\n  ── same [L∪C], DIFFERENT next word (positionless probe: intrinsic next-word entropy + order/saturation — a transformer adds a positional encoding + keeps the current token) ──\n`);
    const seen = new Set();
    for (const e of g.filter((x) => x.next != null)) { if (seen.has(e.next) || seen.size >= 3) continue; seen.add(e.next); w.write(`    predict “${e.next}”:  ${win2(e.words, e.i)}\n`); }
  }
  const lrConflict = [...lrGroups.values()].filter((g) => new Set(g.map((e) => e.word)).size >= 2).sort((a, b) => a.length - b.length);
  for (const g of lrConflict.slice(0, 2)) {
    w.write(`\n  ── same [L,R], DIFFERENT current word (cloze ambiguity — the boundary context fits several fillers) ──\n`);
    const seen = new Set();
    for (const e of g) { if (seen.has(e.word) || seen.size >= 3) continue; seen.add(e.word); w.write(`    fill “${e.word}”:  ${win2(e.words, e.i)}\n`); }
  }

  const coll = [...lcGroups.values()].filter((g) => g.length >= 2 && new Set(g.map((e) => e.word)).size >= 2);
  const cross = coll.filter((g) => new Set(g.map((e) => e.words)).size >= 2);
  const within = coll.filter((g) => new Set(g.map((e) => e.words)).size === 1);
  w.write(`\n(4) [L∪C] collisions: ${cross.length} cross-chunk (order-invariance), ${within.length} within-chunk (saturation, biggest ${within.reduce((m, g) => Math.max(m, g.length), 0)})\n`);
  const win = (words, i) => { const a = Math.max(0, i - 3), b = Math.min(words.length, i + 4); return (a ? "…" : "") + words.slice(a, b).map((x, j) => (a + j === i ? `⟦${x}⟧` : x)).join(" ") + (b < words.length ? "…" : ""); };
  for (const g of cross.filter((x) => x.length <= 4).slice(-4)) {
    w.write(`\n  ── same [L∪C] (different chunks) ──\n`);
    const seen = new Set();
    for (const e of g) { if (seen.has(e.chunk)) continue; seen.add(e.chunk); w.write(`    C=“${e.word}”@${e.i}:  ${win(e.words, e.i)}\n`); }
  }

  // (5) Capping the context to a sliding window removes saturation. Windows below
  // the prefix length are a no-op, so this only helps the long-prefix regime.
  w.write(`\n(5) context cap — [L∪C] → next word, determines-target % by window:\n`);
  w.write(`| window | determines next word % |\n|---|---|\n`);
  w.write(`| unbounded | ${consistency(lcNext).determinedPct.toFixed(1)} |\n`);
  for (const W of CONTEXT_WINDOWS) w.write(`| ${W} words | ${consistency(lcNextW[W]).determinedPct.toFixed(1)} |\n`);

  const buckets = [[0, 0], [1, 1], [2, 2], [3, 4], [5, 7], [8, 1e9]];
  const collideByLen = (map) => {
    const stat = new Map();
    for (const arrI of map.values()) {
      const shared = arrI.length >= 2;
      for (const i of arrI) {
        const b = buckets.find(([lo, hi]) => i >= lo && i <= hi), key = b.join("-");
        let s = stat.get(key); if (!s) stat.set(key, (s = { n: 0, c: 0 })); s.n++; if (shared) s.c++;
      }
    }
    return stat;
  };
  const su = collideByLen(lcByLenU), sw = collideByLen(lcByLenW);
  w.write(`\n    collision % by prefix length (unbounded vs window=5):\n`);
  w.write(`| prefix words | positions | collide% unbounded | collide% window=5 |\n|---|---|---|---|\n`);
  for (const [lo, hi] of buckets) {
    const key = `${lo}-${hi}`, u = su.get(key), wd = sw.get(key); if (!u) continue;
    const range = lo === hi ? `${lo + 1}` : hi > 1e8 ? `${lo + 1}+` : `${lo + 1}-${hi + 1}`;
    w.write(`| ${range} | ${u.n} | ${(100 * u.c / u.n).toFixed(1)} | ${(100 * wd.c / wd.n).toFixed(1)} |\n`);
  }

  // (6) Symmetric clamp: rebuild the level with window=5 on BOTH sides (L and R
  // each reach w-1=4 words) and compare every metric to unbounded.
  const wc = Object.fromEntries(TYPES.map((t) => [t, new Map()]));
  const wLcNext = new Map(), wLrWord = new Map();
  let wPos = 0, wLcPos = 0;
  for (const chunk of chunks) {
    const words = wordsOf(chunk), sigs = buildChunkSignatures(words, { dict, encoder: ENCODER, window: 5 });
    for (let i = 0; i < sigs.length; i++) {
      wPos++;
      const k = keyOf(sigs[i]);
      for (const t of TYPES) wc[t].set(k[t], (wc[t].get(k[t]) || 0) + 1);
      bumpNested(wLrWord, k["[L,R]"], sigs[i].word);
      if (i < sigs.length - 1) { wLcPos++; bumpNested(wLcNext, k["[L∪C]"], sigs[i + 1].word); }
    }
  }
  const uPct = (map, tot) => { let u = 0; for (const c of map.values()) if (c === 1) u++; return 100 * u / tot; };
  w.write(`\n(6) symmetric clamp — window=5 on BOTH L and R (4+⟦word⟧+4 frame):\n`);
  w.write(`| metric | unbounded | window=5 both sides |\n|---|---|---|\n`);
  for (const t of ["L", "R", "[L,R]", "[L∪C]", "[L,C,R]"]) w.write(`| ${t} uniqueness | ${uPct(counts[t], positions).toFixed(1)} | ${uPct(wc[t], wPos).toFixed(1)} |\n`);
  w.write(`| [L∪C] → next word | ${consistency(lcNext).determinedPct.toFixed(1)} | ${consistency(wLcNext).determinedPct.toFixed(1)} |\n`);
  w.write(`| [L,R] → current (cloze) | ${consistency(lrWord).determinedPct.toFixed(1)} | ${consistency(wLrWord).determinedPct.toFixed(1)} |\n`);

  // (7) ±5 frame (radius r=5) and boundary shift-pad. Truncated clamps at the
  // chunk edge; shift-pad keeps 2r total context by extending the far side. This
  // is BENCHMARK-ONLY analysis (the library keeps the [L∪C]=L∪C invariant); the
  // causal [L∪C] cannot borrow the future, so shift-pad is bidirectional only.
  const r = 5;
  const tLR = new Map(), tLCR = new Map(), sLR = new Map(), sLCR = new Map();
  const tCloze = new Map(), sCloze = new Map(), tNext = new Map();
  let p7 = 0;
  for (const chunk of chunks) {
    const words = wordsOf(chunk), sigs = buildChunkSignatures(words, { dict, encoder: ENCODER }), C = sigs.map((s) => s.C), kk = words.length;
    for (let i = 0; i < kk; i++) {
      p7++;
      const cj = C[i].join(","), cw = sigs[i].word;
      const tl = rangeOr(C, i - r, i - 1), tr = rangeOr(C, i + 1, i + r);      // truncated
      const [sl0, sh0] = shiftWin(i, kk, r);
      const sl = rangeOr(C, sl0, i - 1), sr = rangeOr(C, i + 1, sh0);          // shift-pad
      const tk = `${tl.join(",")}|${tr.join(",")}`, sk = `${sl.join(",")}|${sr.join(",")}`;
      tLR.set(tk, (tLR.get(tk) || 0) + 1); sLR.set(sk, (sLR.get(sk) || 0) + 1);
      tLCR.set(`${tl.join(",")}|${cj}|${tr.join(",")}`, (tLCR.get(`${tl.join(",")}|${cj}|${tr.join(",")}`) || 0) + 1);
      sLCR.set(`${sl.join(",")}|${cj}|${sr.join(",")}`, (sLCR.get(`${sl.join(",")}|${cj}|${sr.join(",")}`) || 0) + 1);
      bumpNested(tCloze, tk, cw); bumpNested(sCloze, sk, cw);
      if (i < kk - 1) bumpNested(tNext, rangeOr(C, i - r, i).join(","), sigs[i + 1].word);
    }
  }
  const uP = (m) => { let u = 0; for (const c of m.values()) if (c === 1) u++; return (100 * u / p7).toFixed(1); };
  w.write(`\n(7) ±5 frame (window=6) + boundary shift-pad — bidirectional metrics:\n`);
  w.write(`| metric | truncated (window=6) | shift-pad r=5 |\n|---|---|---|\n`);
  w.write(`| [L,R] uniqueness | ${uP(tLR)} | ${uP(sLR)} |\n`);
  w.write(`| [L,C,R] uniqueness | ${uP(tLCR)} | ${uP(sLCR)} |\n`);
  w.write(`| [L,R] → cloze | ${consistency(tCloze).determinedPct.toFixed(1)} | ${consistency(sCloze).determinedPct.toFixed(1)} |\n`);
  w.write(`| [L∪C] → next word | ${consistency(tNext).determinedPct.toFixed(1)} | n/a (causal) |\n`);
}
