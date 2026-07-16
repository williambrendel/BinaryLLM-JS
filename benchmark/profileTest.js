"use strict";

// ============================================================================
// benchmark/profileTest.js — PRECONDITION: is there a sense-space for layer 1?
// Aggregate context profile Π(w)[a] = Σ_{τ_i=w} 1[a ∈ context]. Model-free.
// T3 (the decisive test): split a polysemous word's contexts by sense (via the
// probe atoms already found), build aggregate profiles EXCLUDING the splitter
// atoms (or it's circular), and compare:
//   cos(fin,geo)  vs  cos(fin1,fin2) / cos(geo1,geo2)  vs  cos(unlab1,unlab2 = floor)
// cross-sense ≪ same-sense ⇒ a sense-space exists in the aggregate ⇒ build layer 2.
// cross-sense ≈ same-sense ⇒ indistinguishable once splitters removed ⇒ ceiling real.
// Usage: node --max-old-space-size=8192 benchmark/profileTest.js <dict> <corpus>
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { fuzzyEncodeWord } from "../src/core/signatures/wordEncoder.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };
const mkRng = (s) => () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const rng = mkRng(11);
const cos = (A, B) => { let d = 0, na = 0, nb = 0; const [s, l] = A.size < B.size ? [A, B] : [B, A]; for (const [k, v] of s) { const o = l.get(k); if (o) d += v * o; } for (const v of A.values()) na += v * v; for (const v of B.values()) nb += v * v; return na && nb ? d / Math.sqrt(na * nb) : 0; };

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
  const w = process.stdout;
  const val = (a) => parts[a].value;

  // splitter roots per sense (substring match on atom value) — used to CLASSIFY and to EXCLUDE
  const CFG = {
    bank: { A: ["swi", "iss", "ubs", "union", "orporat", "corpora"], B: ["danube", "nbur", "enbur", "west", "right", "opposit"], nameA: "FIN", nameB: "GEO" },
    state: { A: ["xidati", "oxidat"], B: ["michig", "ichigan", "egislat", "legisl"], nameA: "CHEM", nameB: "POL" },
  };

  // whole-sentence context atom set for a target occurrence at position p (excl p, excl splitter atoms)
  const contextProfileInto = (Π, s, p, excl) => { const L = s.length; for (let q = 0; q < L; q++) { if (q === p) continue; for (const a of s[q].atoms) if (!excl.has(a)) Π.set(a, (Π.get(a) || 0) + 1); } };

  for (const [tgt, cfg] of Object.entries(CFG)) {
    // precompute splitter atom-id sets
    const setA = new Set(), setB = new Set();
    for (let a = 0; a < F; a++) { const v = val(a); if (cfg.A.some((r) => v.includes(r))) setA.add(a); if (cfg.B.some((r) => v.includes(r))) setB.add(a); }
    const excl = new Set([...setA, ...setB]);   // exclude ALL splitter atoms from profiles (anti-circularity)

    // collect target occurrences, classify by presence of a splitter atom in the sentence
    const occ = { A: [], B: [], U: [] };   // arrays of [sentIndex, pos]
    sents.forEach((s, si) => { for (let p = 0; p < s.length; p++) { if (s[p].w !== tgt) continue; let hasA = false, hasB = false; for (let q = 0; q < s.length; q++) { if (q === p) continue; for (const a of s[q].atoms) { if (setA.has(a)) hasA = true; if (setB.has(a)) hasB = true; } } if (hasA && !hasB) occ.A.push([si, p]); else if (hasB && !hasA) occ.B.push([si, p]); else if (!hasA && !hasB) occ.U.push([si, p]); } });

    const buildΠ = (list) => { const Π = new Map(); for (const [si, p] of list) contextProfileInto(Π, sents[si], p, excl); return Π; };
    const half = (list) => { const a = [], b = []; for (const x of list) (rng() < 0.5 ? a : b).push(x); return [a, b]; };
    const [A1, A2] = half(occ.A), [B1, B2] = half(occ.B), [U1, U2] = half(occ.U);
    const ΠA = buildΠ(occ.A), ΠB = buildΠ(occ.B);
    const cAB = cos(ΠA, ΠB), cAA = cos(buildΠ(A1), buildΠ(A2)), cBB = cos(buildΠ(B1), buildΠ(B2)), cUU = cos(buildΠ(U1), buildΠ(U2));
    // cross-word floor: this target's unlabelled vs a shuffled control (random content sample)
    w.write(`════ ${tgt} (T3: cross-sense vs same-sense; splitter atoms EXCLUDED) ════\n`);
    w.write(`   sense sizes:  ${cfg.nameA}=${occ.A.length}  ${cfg.nameB}=${occ.B.length}  unlabelled=${occ.U.length}\n`);
    w.write(`   cos(${cfg.nameA},${cfg.nameB})  cross-sense = ${cAB.toFixed(3)}   ← want LOW\n`);
    w.write(`   cos(${cfg.nameA}₁,${cfg.nameA}₂) same-sense  = ${cAA.toFixed(3)}   ← noise ceiling (if low, |A| too small)\n`);
    w.write(`   cos(${cfg.nameB}₁,${cfg.nameB}₂) same-sense  = ${cBB.toFixed(3)}\n`);
    w.write(`   cos(unlab₁,unlab₂) same-word floor = ${cUU.toFixed(3)}\n`);
    const gap = ((cAA + cBB) / 2) - cAB;
    w.write(`   → same-sense − cross-sense = ${gap.toFixed(3)}  ⇒ ${cAA < 0.15 || cBB < 0.15 ? "|A| TOO SMALL — split-half noisy, do not read cross-sense" : gap > 0.05 ? "SENSE-SPACE EXISTS (cross ≪ same) — build layer 2" : "senses INDISTINGUISHABLE once splitters removed — ceiling real"}\n\n`);
  }
};

main();
