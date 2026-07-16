"use strict";

// ============================================================================
// benchmark/probeSemantic.js — targeted semantic probe. Stop ranking; ask by name.
// Fixed word lists declared in advance, NO filter, NO truncation. For each probe
// WORD: does it co-occur with the target, in the window? Report every one incl n_A=0.
// The decisive quantity is COVERAGE = fraction of A whose window contains ANY probe
// word of a sense-list, and the UNION LIFT (treat the whole list as one feature) —
// the direct test of "pool the sense atoms into one gate" vs per-atom certification.
// Word-level co-occurrence (clean signal-existence; atom access is noisier, separate Q).
// Usage: node --max-old-space-size=8192 benchmark/probeSemantic.js <dict> <corpus>
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };
const katzLCB = (nA, nAbar, nInA, nInAbar, z) => { const a = nA + 0.5, b = (nInA - nA) + 0.5, c = nAbar + 0.5, d = (nInAbar - nAbar) + 0.5; return Math.exp(Math.log((a / (a + b)) / (c / (c + d))) - z * Math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))); };

const LISTS = {
  "merc/ELEM": "element metal liquid toxic mercuric compound thermometer poisoning vapor sulfur arsenic chemical".split(" "),
  "merc/PLANET": "orbit sun venus mars planet solar spacecraft astronomy crater orbital".split(" "),
  "jup/PLANET": "moons orbit saturn jovian rings galilean spacecraft atmosphere moon".split(" "),
  "jup/GOD": "roman god temple mythology deity jove worship".split(" "),
  "mars/PLANET": "rover surface orbit crater atmosphere spacecraft red planet water".split(" "),
  "mars/GOD": "roman god war mythology deity temple".split(" "),
  "CONTROL": "banana purple guitar elephant".split(" "),
};
const TARGET_LISTS = { mercury: ["merc/ELEM", "merc/PLANET", "CONTROL"], jupiter: ["jup/PLANET", "jup/GOD", "CONTROL"], mars: ["mars/PLANET", "mars/GOD", "CONTROL"] };

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  loadDict(dictArg);   // (dict not needed — word-level probe — but keep the arg contract)
  const wordList = new Map();   // word -> list name
  for (const [ln, ws] of Object.entries(LISTS)) for (const w of ws) wordList.set(w, ln);
  const targets = Object.keys(TARGET_LISTS);
  const tset = new Set(targets);

  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const toks = tokenizeStream(line.slice(seg.start, seg.end));
    const ws = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws);
  }
  const w = process.stdout;
  const RADII = [1, 5, Infinity];
  const rlab = (r) => (r === Infinity ? "∞" : String(r));

  // accumulators keyed by cfg = enc|radius
  const results = {};   // cfg -> {N, absA{t}, Mw{word}, nAw{t}{word}, anyList{list}, anyA{t}{list}}
  for (const enc of ["L|C", "L,R"]) for (const r of RADII) {
    const key = `${enc}@${rlab(r)}`;
    const R = { N: 0, absA: {}, Mw: new Map(), nAw: {}, anyList: {}, anyA: {} };
    for (const t of targets) { R.absA[t] = 0; R.nAw[t] = new Map(); R.anyA[t] = {}; }
    for (const s of sents) {
      const L = s.length;
      for (let p = 0; p < L; p++) {
        let lo, hi, skipP;
        if (enc === "L|C") { lo = r === Infinity ? 0 : Math.max(0, p - 1 - r); hi = p - 1; skipP = -1; }
        else { lo = r === Infinity ? 0 : Math.max(0, p - r); hi = r === Infinity ? L - 1 : Math.min(L - 1, p + r); skipP = p; }
        if (hi < lo) continue;
        const words = new Set(), lists = new Set();
        for (let q = lo; q <= hi; q++) { if (q === skipP) continue; const ww = s[q]; const ln = wordList.get(ww); if (ln) { words.add(ww); lists.add(ln); } }
        R.N++;
        const ti = tset.has(s[p]) ? s[p] : null;
        if (ti) R.absA[ti]++;
        for (const ww of words) { R.Mw.set(ww, (R.Mw.get(ww) || 0) + 1); if (ti) R.nAw[ti].set(ww, (R.nAw[ti].get(ww) || 0) + 1); }
        for (const ln of lists) { R.anyList[ln] = (R.anyList[ln] || 0) + 1; if (ti) R.anyA[ti][ln] = (R.anyA[ti][ln] || 0) + 1; }
      }
    }
    results[key] = R;
  }

  w.write(`# probeSemantic — ${path.basename(corpusArg)} | word-level co-occurrence | targets: ${targets.join(", ")}\n`);
  const anyR = results[`L|C@1`];
  w.write(`# |A|: ${targets.map((t) => t + "=" + anyR.absA[t]).join(" ")}   N≈${anyR.N}\n\n`);

  for (const t of targets) {
    w.write(`════════════════════ ${t} ════════════════════\n`);
    // DECISIVE MATRIX: coverage% and union-lift per (sense-list × radius × enc)
    for (const enc of ["L|C", "L,R"]) {
      w.write(`  [${enc}]  ` + RADII.map((r) => `r${rlab(r)}`.padStart(14)).join("") + `\n`);
      for (const ln of TARGET_LISTS[t]) {
        const cells = RADII.map((r) => {
          const R = results[`${enc}@${rlab(r)}`], A = R.absA[t], anyA = R.anyA[t][ln] || 0, tot = R.anyList[ln] || 0;
          const cov = 100 * anyA / A, pA = anyA / A, pB = (tot - anyA) / (R.N - A), lift = pB > 0 ? pA / pB : (pA > 0 ? Infinity : 0);
          return `${cov.toFixed(1)}%|${(lift === Infinity ? "∞" : lift.toFixed(0))}×`.padStart(14);
        });
        w.write(`    ${ln.padEnd(11)}` + cells.join("") + `\n`);
      }
    }
    // PER-WORD, by name, incl n_A=0 — at r=∞ [L,R] (widest reach: any sentence co-occurrence)
    for (const enc of ["L|C", "L,R"]) {
      const R = results[`${enc}@∞`], A = R.absA[t];
      w.write(`  ── per-word co-occurrence, [${enc}] r=∞ (n_A incl zeros, point-lift) ──\n`);
      for (const ln of TARGET_LISTS[t]) {
        const parts = LISTS[ln].map((word) => {
          const nA = R.nAw[t].get(word) || 0, M = R.Mw.get(word) || 0, nB = M - nA;
          const fA = nA / A, fB = nB / (R.N - A), pl = fB > 0 ? fA / fB : (fA > 0 ? Infinity : 0);
          return `${word}(${nA}${nA >= 3 ? "," + (pl === Infinity ? "∞" : pl.toFixed(0)) + "×" : ""})`;
        });
        const anyA = R.anyA[t][ln] || 0, tot = R.anyList[ln] || 0, pA = anyA / A, pB = (tot - anyA) / (R.N - A);
        w.write(`    ${ln.padEnd(11)} cov=${(100 * pA).toFixed(1)}% unionLift=${(pB > 0 ? (pA / pB).toFixed(0) : "∞")}×  ::  ${parts.join(" ")}\n`);
      }
    }
    w.write(`\n`);
  }
};

main();
