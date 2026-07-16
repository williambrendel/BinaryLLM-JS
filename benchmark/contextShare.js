"use strict";

// ============================================================================
// benchmark/contextShare.js — does a target's context set SHARE atoms, at what radius?
// No trainer, no model. Pure counting. Sweep radius × encoding × word.
//   A = {positions p : word[p]==target};  context = window atoms (fuzzy).
//   score(a) = Wilson-LCB[f_A] / f_corpus   (enrichment guaranteed at 95%).
// Tables (condensed): B=n_A≥3 by score (PRIMARY), C=raw-ratio singleton frac,
//   D=word-level Σ score over n_A≥3 atoms.  |A| reported first.
// Usage: node --max-old-space-size=8192 benchmark/contextShare.js <dict> <corpus>
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { fuzzyEncodeWord } from "../src/core/signatures/wordEncoder.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";
import { kindToString } from "../src/core/parts/kind.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };
const FUNCW = new Set("the of a an and or but in on at to for with from by as is are was were be been being it its this that these those he she they we you i his her their our not no s".split(" "));
// Katz log-relative-risk LCB with Haldane–Anscombe 0.5 correction — bounds BOTH rates.
// score = exp(logRR − z·SE), SE² = 1/a − 1/(a+b) + 1/c − 1/(c+d). The 1/n terms explode
// when EITHER count is small — a hapax (n_Ā=0) gets a wide interval instead of an exploding ratio.
const katzLCB = (nA, nAbar, nInA, nInAbar, z) => {
  const a = nA + 0.5, b = (nInA - nA) + 0.5, c = nAbar + 0.5, d = (nInAbar - nAbar) + 0.5;
  const logRR = Math.log((a / (a + b)) / (c / (c + d)));
  const se = Math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d));
  return Math.exp(logRR - z * se);
};

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg);
  const parts = dict.allParts(), F = dict.size();

  // ---- load sentences as arrays of {w, atoms} (fuzzy atoms cached per word) ----
  const cache = new Map();
  const atomsOf = (w) => { let a = cache.get(w); if (!a) { a = Int32Array.from(new Set(fuzzyEncodeWord(dict, w))); cache.set(w, a); } return a; };
  const sents = [];
  const text = fs.readFileSync(corpusArg, "utf8");
  for (const line of text.split("\n")) for (const seg of segmentText(line)) {
    const toks = tokenizeStream(line.slice(seg.start, seg.end));
    const ws = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws.map((w) => ({ w, atoms: atomsOf(w) })));
  }
  const totalWords = sents.reduce((s, x) => s + x.length, 0);

  // ---- pick the monosemous content word by largest |A| in this slice ---------
  const wc = new Map(); for (const s of sents) for (const t of s) wc.set(t.w, (wc.get(t.w) || 0) + 1);
  const targets = ["the", "bank", "state", "city"];   // the=control, bank/state=polysemous, city=collocational content
  const w = process.stdout;
  w.write(`# contextShare — dict=${path.basename(dictArg)} corpus=${path.basename(corpusArg)} | F=${F} | N_words=${totalWords} | sents=${sents.length}\n`);
  w.write(`# targets: ${targets.map((t) => t + "(|A|~" + (wc.get(t) || 0) + ")").join(" ")}   | statistic: Katz RR-LCB (Haldane 0.5), z=1.645\n\n`);

  // ---- one tally per (encoding, radius) via a stamp array (no Set alloc) ------
  const RADII = [1, 3, 5, 10, Infinity];
  const stamp = new Int32Array(F).fill(-1);
  const tag = (a) => { const p = parts[a]; return (kindToString(p.kind) === "whole" && FUNCW.has(p.value)) ? "·fn" : "  "; };
  const label = (a) => { const p = parts[a]; return `${p.value}:${kindToString(p.kind)}`; };

  for (const enc of ["L|C", "L,R"]) {
    for (const r of RADII) {
      const M = new Float64Array(F);                       // M_a: #positions whose context contains a
      const nA = targets.map(() => new Float64Array(F));   // per-target
      const absA = new Float64Array(targets.length);
      let N = 0, pid = 0;
      const tIdx = new Map(targets.map((t, i) => [t, i]));
      for (const s of sents) {
        const L = s.length;
        for (let p = 0; p < L; p++) {
          // window word indices for this encoding+radius (exclude target position p)
          let lo, hi, skipP;
          if (enc === "L|C") { lo = r === Infinity ? 0 : Math.max(0, p - 1 - r); hi = p - 1; skipP = -1; }
          else { lo = r === Infinity ? 0 : Math.max(0, p - r); hi = r === Infinity ? L - 1 : Math.min(L - 1, p + r); skipP = p; }
          if (hi < lo) continue;                           // no context
          pid++; let any = false;
          const ti = tIdx.get(s[p].w);
          for (let q = lo; q <= hi; q++) { if (q === skipP) continue; const atoms = s[q].atoms; for (let k = 0; k < atoms.length; k++) { const a = atoms[k]; if (stamp[a] !== pid) { stamp[a] = pid; M[a]++; if (ti !== undefined) nA[ti][a]++; any = true; } } }
          if (!any) continue;
          N++; if (ti !== undefined) absA[ti]++;
        }
      }
      // ---- per target: tables ------------------------------------------------
      w.write(`══ encoding=[${enc}]  radius=${r === Infinity ? "∞" : r}  N=${N} ══\n`);
      for (let ti = 0; ti < targets.length; ti++) {
        const A = absA[ti]; const nAt = nA[ti];
        w.write(`  ${targets[ti].padEnd(11)} |A|=${A}${A < 200 ? "  ⚠️<200 UNINTERPRETABLE" : ""}\n`);
        if (A < 1) continue;
        const rows = [];
        let sing = 0, cRawTop = 0;
        for (let a = 0; a < F; a++) {
          if (nAt[a] === 0) continue;
          const fcorp = M[a] / N, fA = nAt[a] / A, nAbar = M[a] - nAt[a], fAbar = nAbar / (N - A);
          const score = katzLCB(nAt[a], nAbar, A, N - A, 1.645), score1 = katzLCB(nAt[a], nAbar, A, N - A, 1.0);   // Katz RR-LCB
          const raw = (fA + fAbar) > 0 ? (fA - fAbar) / (fA + fAbar) : 0;
          rows.push({ a, nA: nAt[a], nAbar, fcorp, score, score1, raw });
          if (nAt[a] === 1) sing++;
        }
        // Table C: raw-ratio top-40 singleton fraction
        const byRaw = [...rows].sort((x, y) => y.raw - x.raw).slice(0, 40);
        const cSing = byRaw.filter((r) => r.nA === 1).length;
        // Table B: n_A>=3 by score, top 12 (condensed)
        const B = rows.filter((r) => r.nA >= 3).sort((x, y) => y.score - x.score).slice(0, 12);
        w.write(`    [B n_A≥3] ` + (B.length ? B.map((r) => `${label(r.a)}${tag(r.a)}(${r.nA},${r.score.toFixed(1)}×)`).join("  ") : "— none with n_A≥3 —") + `\n`);
        w.write(`    [C raw]   top-40 singleton(n_A=1) fraction: ${cSing}/40${cSing >= 38 ? "  ⇒ raw view ≈ pure artifact" : ""}\n`);
        // Table D: word-level Σ score over n_A≥3 atoms
        const scoreByAtom = new Float64Array(F); for (const r of B.length ? rows.filter((r) => r.nA >= 3) : []) scoreByAtom[r.a] = r.score;
        const wordScore = new Map();
        for (const [ww, atoms] of cache) { let sc = 0, cnt = 0; for (const a of atoms) if (scoreByAtom[a] > 0) { sc += scoreByAtom[a]; cnt++; } if (cnt >= 1 && sc > 0) wordScore.set(ww, [sc, cnt]); }
        const D = [...wordScore.entries()].sort((x, y) => y[1][0] - x[1][0]).slice(0, 8);
        w.write(`    [D word]  ` + D.map(([ww, [sc, cnt]]) => `${ww}(${sc.toFixed(1)}/${cnt}a)`).join("  ") + `\n`);
      }
      w.write(`\n`);
    }
  }
};

main();
