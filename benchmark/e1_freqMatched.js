"use strict";

// ============================================================================
// benchmark/e1_freqMatched.js  —  BLOCKING experiment E-1
//
// The §1.3 slot-ceiling table (ant lift ≥ syn lift, decaying with radius) was
// computed against a random baseline sampled UNIFORMLY from the pool. Antonyms
// skew frequent; frequent words have denser profiles and higher cosine with
// everything — so the antonym-over-synonym result may be a frequency artifact.
//
// E-1 re-runs the table with a FREQUENCY-MATCHED random baseline: for each real
// pair (a,b), the matched-random comparison draws (a',b') from the SAME log2-freq
// bins as a and b. Then lift = mean pair-cosine / mean matched-random-cosine.
//
// Reports, per radius × band, BOTH baselines side by side, so we can see whether
// ant ≥ syn survives the control. If antonym lift collapses to/below synonym lift
// under matching, H0 is dead on arrival (and r=1 is not forced by this argument).
//
// Usage: node benchmark/e1_freqMatched.js <dict> <corpus>   (MINOCC via env)
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { streamSignatureChunks } from "../src/core/signatures/streamSignatures.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const mkRng = (seed) => () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const rng = mkRng(98765);
const rand = (n) => Math.floor(rng() * n);
const cosine = (a, b, na, nb) => { let d = 0; const [s, l] = a.size < b.size ? [a, b] : [b, a]; for (const [k, v] of s) { const o = l.get(k); if (o) d += v * o; } return na && nb ? d / (na * nb) : 0; };

// Curated pairs (same as contextSemantics.js). None are morphological variants
// (no X / unX, X / Xs), so the spelling-collision guard is a no-op here — asserted below.
const SYN = [["also", "additionally"], ["however", "although"], ["first", "initial"], ["large", "major"], ["area", "region"], ["city", "town"], ["war", "conflict"], ["group", "team"], ["known", "called"], ["used", "employed"], ["made", "created"], ["began", "started"], ["many", "several"], ["big", "large"], ["important", "significant"], ["often", "frequently"], ["build", "construct"], ["end", "finish"], ["near", "close"], ["old", "ancient"], ["new", "modern"], ["fast", "quick"], ["show", "display"], ["help", "aid"]];
const ANT = [["first", "last"], ["high", "low"], ["old", "new"], ["large", "small"], ["long", "short"], ["more", "less"], ["before", "after"], ["north", "south"], ["east", "west"], ["increase", "decrease"], ["early", "late"], ["win", "loss"], ["male", "female"], ["good", "bad"], ["major", "minor"]];

// morphological-variant guard: one member is the other + a short affix, or they share a long stem
const isMorphPair = ([a, b]) => { const [s, l] = a.length <= b.length ? [a, b] : [b, a]; if (l.startsWith(s) || l.endsWith(s)) return l.length - s.length <= 3; return false; };

const [dictArg, corpusArg] = process.argv.slice(2);
const MINOCC = Number(process.env.MINOCC || 20);
const dict = loadDict(dictArg);
const radii = [1, 2, 3, 5, 10, null];

// Build per-word L and R profiles at a given radius (single stream pass).
const buildProfiles = async (radius) => {
  const vocabIndex = new Map(), vocab = [];
  const profL = new Map(), profR = new Map(), wcount = new Map();
  for await (const { records } of streamSignatureChunks(dict, corpusArg, { chunkSize: 20000, radius, vocabIndex, vocab })) {
    for (const r of records) {
      const w = r.curWord;
      wcount.set(w, (wcount.get(w) || 0) + 1);
      let mL = profL.get(w); if (!mL) profL.set(w, (mL = new Map()));
      for (const b of r.L) mL.set(b, (mL.get(b) || 0) + 1);
      let mR = profR.get(w); if (!mR) profR.set(w, (mR = new Map()));
      for (const b of r.R) mR.set(b, (mR.get(b) || 0) + 1);
    }
  }
  const idOf = new Map(); vocab.forEach((v, i) => idOf.set(v, i));
  return { profL, profR, wcount, idOf };
};

const l2 = (m) => { let n = 0; for (const v of m.values()) n += v * v; return Math.sqrt(n); };

const evalBand = (prof, wcount, idOf, pairsSyn, pairsAnt, pool, poolByBin, binOf) => {
  const norm = new Map(); for (const w of pool) norm.set(w, l2(prof.get(w)));
  const cosPair = (a, b) => cosine(prof.get(a), prof.get(b), norm.get(a), norm.get(b));
  const setCos = (pairs) => pairs.reduce((s, [a, b]) => s + cosPair(a, b), 0) / pairs.length;

  // UNMATCHED baseline: uniform random pairs from pool
  let ru = 0; const RU = 5000; for (let k = 0; k < RU; k++) { const a = pool[rand(pool.length)]; let b = pool[rand(pool.length)]; while (b === a) b = pool[rand(pool.length)]; ru += cosPair(a, b); } ru /= RU;

  // MATCHED baseline: per real pair, draw comparison words from the same freq bins
  const matchedFor = (pairs) => {
    let acc = 0, cnt = 0;
    for (const [a, b] of pairs) {
      const ba = poolByBin.get(binOf(a)), bb = poolByBin.get(binOf(b));
      if (!ba || !bb) continue;
      let s = 0; const K = 300;
      for (let k = 0; k < K; k++) { let x = ba[rand(ba.length)]; let y = bb[rand(bb.length)]; let tries = 0; while ((x === y) && tries++ < 5) y = bb[rand(bb.length)]; s += cosPair(x, y); }
      acc += s / K; cnt++;
    }
    return acc / cnt;
  };
  const synC = setCos(pairsSyn), antC = setCos(pairsAnt);
  const synM = matchedFor(pairsSyn), antM = matchedFor(pairsAnt);
  return { synC, antC, ru, synM, antM };
};

const main = async () => {
  const badMorph = [...SYN, ...ANT].filter(isMorphPair);
  const out = [];
  out.push(`# E-1 frequency-matched r-sweep — dict=${path.basename(dictArg)} corpus=${path.basename(corpusArg)} | MINOCC=${MINOCC}`);
  out.push(`# morphological-variant pairs excluded: ${badMorph.length ? badMorph.map((p) => p.join("/")).join(", ") : "none"}`);
  out.push(`# lift = mean pair-cosine / mean random-cosine.  UNMATCHED = uniform pool baseline (old §1.3).  MATCHED = same-freq-bin baseline (the control).`);
  out.push("");

  for (const radius of radii) {
    const { profL, profR, wcount, idOf } = await buildProfiles(radius);
    const pool = [...wcount.entries()].filter(([, c]) => c >= MINOCC).map(([w]) => w);
    const okW = (s) => { const w = idOf.get(s); return w !== undefined && (wcount.get(w) || 0) >= MINOCC ? w : -1; };
    const qual = (list) => list.filter((p) => !isMorphPair(p)).map(([a, b]) => [okW(a), okW(b)]).filter(([a, b]) => a >= 0 && b >= 0);
    const synP = qual(SYN), antP = qual(ANT);

    // frequency bins (log2 of occurrence count)
    const binOf = (w) => Math.floor(Math.log2(wcount.get(w) || 1));
    const poolByBin = new Map(); for (const w of pool) { const b = binOf(w); let g = poolByBin.get(b); if (!g) poolByBin.set(b, (g = [])); g.push(w); }

    // frequency skew, to show the confound is real
    const mfreq = (ids) => ids.reduce((s, [a, b]) => s + wcount.get(a) + wcount.get(b), 0) / (2 * ids.length);
    const poolMeanFreq = pool.reduce((s, w) => s + wcount.get(w), 0) / pool.length;

    const rlabel = radius === null ? "∞" : String(radius);
    for (const [band, prof] of [["L", profL], ["R", profR]]) {
      const r = evalBand(prof, wcount, idOf, synP, antP, pool, poolByBin, binOf);
      out.push(
        `r${rlabel.padEnd(2)} ${band}  | syn=${r.synC.toFixed(3)} ant=${r.antC.toFixed(3)}  ` +
        `| UNMATCHED rand=${r.ru.toFixed(3)} → synLift=${(r.synC / r.ru).toFixed(2)}× antLift=${(r.antC / r.ru).toFixed(2)}×  ` +
        `| MATCHED randSyn=${r.synM.toFixed(3)} randAnt=${r.antM.toFixed(3)} → synLift=${(r.synC / r.synM).toFixed(2)}× antLift=${(r.antC / r.antM).toFixed(2)}×`
      );
    }
    out.push(`      (syn|ant|pool mean freq: ${mfreq(synP).toFixed(0)} | ${mfreq(antP).toFixed(0)} | ${poolMeanFreq.toFixed(0)}  — n_syn=${synP.length} n_ant=${antP.length} pool=${pool.length})`);
    process.stderr.write(`  r=${rlabel} done\n`);
  }
  process.stdout.write(out.join("\n") + "\n");
};

main();
