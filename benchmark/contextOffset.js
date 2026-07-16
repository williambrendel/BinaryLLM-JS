"use strict";

// ============================================================================
// benchmark/contextOffset.js — OFFSET-RESOLVED contrast (no pooling).
// For target w, measure the atom contrast at EACH offset d separately (word at p+d),
// instead of unioning a window. Reveals (1) whether a sense atom has a SHARP single-
// offset contrast that the pooled union washes out — the case for selection over
// pooling; (2) which side of the causal boundary each sense cluster sits on
// (d<0 = causal, reachable by [L|C]; d>0 = future, only via [L,R]/the encoder code).
//   score(a,d) = Katz RR-LCB (Haldane 0.5).
// Usage: node --max-old-space-size=8192 benchmark/contextOffset.js <dict> <corpus>
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
const FUNCW = new Set("the of a an and or but in on at to for with from by as is are was were be been being it its this that these those not no s".split(" "));
const katzLCB = (nA, nAbar, nInA, nInAbar, z) => { const a = nA + 0.5, b = (nInA - nA) + 0.5, c = nAbar + 0.5, d = (nInAbar - nAbar) + 0.5; return Math.exp(Math.log((a / (a + b)) / (c / (c + d))) - z * Math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))); };

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg);
  const parts = dict.allParts(), F = dict.size();
  const cache = new Map();
  const atomsOf = (w) => { let a = cache.get(w); if (!a) { a = Int32Array.from(new Set(fuzzyEncodeWord(dict, w))); cache.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const toks = tokenizeStream(line.slice(seg.start, seg.end));
    const ws = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws.map((x) => ({ w: x, atoms: atomsOf(x) })));
  }
  const targets = ["bank", "state"];
  const OFF = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5];
  const oi = new Map(OFF.map((d, i) => [d, i]));
  const M = OFF.map(() => new Float64Array(F)), Nd = new Float64Array(OFF.length);
  const nA = targets.map(() => OFF.map(() => new Float64Array(F))), absA = targets.map(() => new Float64Array(OFF.length));
  const tIdx = new Map(targets.map((t, i) => [t, i]));

  for (const s of sents) {
    const L = s.length;
    for (let p = 0; p < L; p++) {
      const ti = tIdx.get(s[p].w);
      for (const d of OFF) { const q = p + d; if (q < 0 || q >= L) continue; const k = oi.get(d); Nd[k]++; if (ti !== undefined) absA[ti][k]++; const at = s[q].atoms; for (let j = 0; j < at.length; j++) { const a = at[j]; M[k][a]++; if (ti !== undefined) nA[ti][k][a]++; } }
    }
  }

  const w = process.stdout;
  const label = (a) => { const p = parts[a]; return `${p.value}:${kindToString(p.kind)}`; };
  const isfn = (a) => { const p = parts[a]; return kindToString(p.kind) === "whole" && FUNCW.has(p.value); };
  w.write(`# contextOffset — ${path.basename(corpusArg)} | offset-resolved (no pooling) | Katz RR-LCB z=1.645\n`);
  w.write(`#   d<0 = LEFT (causal, [L|C] reachable)   d>0 = RIGHT (future, only [L,R]/encoder)\n\n`);

  // sense atoms to track across offsets (matched by value substring)
  const SENSE = { bank: ["orporat", "deposit", "swi", "iss", "danube", "ubs", "union", "west", "river", "ank"], state: ["xidati", "michig", "egislat", "highway", "epartm", "universi", "chigan"] };

  for (let ti = 0; ti < targets.length; ti++) {
    w.write(`══════════ ${targets[ti]} ══════════\n`);
    // per-offset top atoms (n_A>=3)
    for (const d of OFF) {
      const k = oi.get(d), A = absA[ti][k], nAt = nA[ti][k];
      const rows = [];
      for (let a = 0; a < F; a++) { if (nAt[a] < 3) continue; const sc = katzLCB(nAt[a], M[k][a] - nAt[a], A, Nd[k] - A, 1.645); rows.push([a, nAt[a], sc]); }
      rows.sort((x, y) => y[2] - x[2]);
      const side = d < 0 ? "L" : "R";
      w.write(`  d=${d >= 0 ? "+" + d : d} (${side}) |A|=${A}: ` + rows.slice(0, 8).map(([a, n, sc]) => `${label(a)}${isfn(a) ? "·fn" : ""}(${n},${sc.toFixed(0)}×)`).join("  ") + `\n`);
    }
    // sense-atom offset profile: peak offset for each tracked sense atom
    w.write(`  ── sense-atom offset profile (score by d; where does each sense cluster live?) ──\n`);
    const tracked = new Map();  // pick the highest-count atom id matching each substring
    for (const sub of SENSE[targets[ti]]) { let best = -1, bc = -1; for (let a = 0; a < F; a++) { if (!parts[a].value.includes(sub)) continue; let tot = 0; for (const d of OFF) tot += nA[ti][oi.get(d)][a]; if (tot > bc) { bc = tot; best = a; } } if (best >= 0 && bc >= 3) tracked.set(sub, best); }
    for (const [sub, a] of tracked) {
      const prof = OFF.map((d) => { const k = oi.get(d), A = absA[ti][k], n = nA[ti][k][a]; return n >= 1 ? katzLCB(n, M[k][a] - n, A, Nd[k] - A, 1.645) : 0; });
      let pk = 0; for (let i = 1; i < OFF.length; i++) if (prof[i] > prof[pk]) pk = i;
      w.write(`    ${label(a).padEnd(16)} peak@d=${OFF[pk] >= 0 ? "+" + OFF[pk] : OFF[pk]} (${prof[pk].toFixed(0)}×)  | ` + OFF.map((d, i) => `${d >= 0 ? "+" + d : d}:${prof[i].toFixed(0)}`).join(" ") + `\n`);
    }
    w.write(`\n`);
  }
};

main();
