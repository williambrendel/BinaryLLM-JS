"use strict";

// ============================================================================
// benchmark/synonymCollision.js
//
// Is the CONTEXT signature (L∪R, target excluded) organized by target semantics,
// or is it a spelling-driven near-hash? Two directions:
//
//   A. target ⇒ signature (recall): do records that share a target — or whose
//      targets are synonyms / antonyms — have SIMILAR signatures? Compared to
//      same-word and to random pairs. C-spelling Jaccard shows the atom is spelling.
//   B. signature ⇒ target (precision): among the most SIMILAR signature pairs, how
//      often is the target the same / related, vs the base rate?
//
// The fork this resolves: if synonym-pair signature similarity ≈ random, the atom
// is spelling-based (fix the atom). If synonym ≫ random, context is already
// semantic-ish and the over-uniqueness is in the join (fix the composition).
//
// Usage: node benchmark/synonymCollision.js [dataset.bin]   (MINOCC, SAMPLE via env)
// ============================================================================

import fs from "node:fs";
import { readDataset } from "../src/core/signatures/signatureDataset.js";

const unionSorted = (a, b) => { const out = []; let i = 0, j = 0; while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } } while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out; };
const jaccard = (a, b) => { let i = 0, j = 0, inter = 0; while (i < a.length && j < b.length) { if (a[i] < b[j]) i++; else if (a[i] > b[j]) j++; else { inter++; i++; j++; } } const u = a.length + b.length - inter; return u === 0 ? 0 : inter / u; };
const rand = (n) => Math.floor(Math.random() * n);

const SYN = [["also", "additionally"], ["however", "although"], ["first", "initial"], ["large", "major"], ["area", "region"], ["city", "town"], ["war", "conflict"], ["group", "team"], ["known", "called"], ["used", "employed"], ["made", "created"], ["began", "started"], ["many", "several"], ["big", "large"], ["important", "significant"], ["often", "frequently"], ["build", "construct"], ["end", "finish"], ["near", "close"], ["old", "ancient"], ["new", "modern"], ["fast", "quick"], ["show", "display"], ["help", "aid"]];
const ANT = [["first", "last"], ["high", "low"], ["old", "new"], ["large", "small"], ["long", "short"], ["more", "less"], ["before", "after"], ["north", "south"], ["east", "west"], ["increase", "decrease"], ["early", "late"], ["began", "ended"], ["win", "loss"], ["male", "female"], ["good", "bad"], ["up", "down"], ["open", "closed"], ["major", "minor"]];

const p = process.argv[2] || "data/signatures/wiki_valid_sigs.bin";
const MINOCC = Number(process.env.MINOCC || 40);
const S = Number(process.env.SAMPLE || 40000);
const ds = readDataset(new Uint8Array(fs.readFileSync(p)));
const recs = ds.records.filter((r) => r.scale === 0);
const id = new Map(); ds.vocab.forEach((wd, i) => id.set(typeof wd === "string" ? wd : Buffer.from(wd).toString("latin1"), i));

// Precompute context signatures L∪R (target excluded) and index by target word.
const sig = recs.map((r) => unionSorted(r.L, r.R));
const cband = new Map(); // curWord -> its own C spelling bag (occurrence-invariant)
const byWord = new Map();
for (let i = 0; i < recs.length; i++) { const w = recs[i].curWord; let g = byWord.get(w); if (!g) byWord.set(w, (g = [])); g.push(i); if (!cband.has(w)) cband.set(w, recs[i].C); }
const okWord = (wd) => { const w = id.get(wd); return w !== undefined && (byWord.get(w)?.length || 0) >= MINOCC ? w : -1; };
const pairs = (list) => list.map(([a, b]) => [okWord(a), okWord(b)]).filter(([a, b]) => a >= 0 && b >= 0);
const synP = pairs(SYN), antP = pairs(ANT);
const pool = [...byWord.entries()].filter(([, g]) => g.length >= MINOCC).map(([w]) => w);

// mean signature Jaccard over sampled cross pairs of two word-groups.
const crossSig = (wa, wb, s) => { const A = byWord.get(wa), B = byWord.get(wb); let sum = 0; for (let k = 0; k < s; k++) sum += jaccard(sig[A[rand(A.length)]], sig[B[rand(B.length)]]); return sum / s; };
const meanOver = (wps, s) => { let sum = 0; for (const [a, b] of wps) sum += crossSig(a, b, s); return sum / wps.length; };

const w = process.stdout;
w.write(`# synonymCollision — ${p.split("/").pop()} | records=${recs.length} | MINOCC=${MINOCC} | sig=L∪R\n`);
w.write(`  qualifying pairs: ${synP.length}/${SYN.length} synonym, ${antP.length}/${ANT.length} antonym; pool=${pool.length} words\n\n`);

// --- A. target ⇒ signature -------------------------------------------------
const perWordS = Math.max(200, Math.floor(S / Math.max(1, pool.length)));
let sameW = 0; for (const wd of pool) sameW += crossSig(wd, wd, 400); sameW /= pool.length; // same word (diff occurrences)
let rndW = 0; for (let k = 0; k < S; k++) { const a = pool[rand(pool.length)]; let b = pool[rand(pool.length)]; while (b === a) b = pool[rand(pool.length)]; rndW += jaccard(sig[byWord.get(a)[rand(byWord.get(a).length)]], sig[byWord.get(b)[rand(byWord.get(b).length)]]); } rndW /= S;
const synS = meanOver(synP, 600), antS = meanOver(antP, 600);
const cJac = (wps) => { let s = 0; for (const [a, b] of wps) s += jaccard(cband.get(a), cband.get(b)); return s / wps.length; };

w.write(`A. target ⇒ signature   (mean L∪R Jaccard of paired records)\n`);
w.write(`   same word        ${sameW.toFixed(3)}\n`);
w.write(`   synonym targets  ${synS.toFixed(3)}   (C-spelling Jaccard ${cJac(synP).toFixed(3)})\n`);
w.write(`   antonym targets  ${antS.toFixed(3)}   (C-spelling Jaccard ${cJac(antP).toFixed(3)})\n`);
w.write(`   random targets   ${rndW.toFixed(3)}\n`);
w.write(`   → synonym/random lift = ${(synS / rndW).toFixed(2)}×   same/random = ${(sameW / rndW).toFixed(2)}×\n\n`);

// --- B. signature ⇒ target -------------------------------------------------
// Sample random record pairs; bucket by signature Jaccard; P(same target) per bucket.
const relSet = new Set([...synP, ...antP].map(([a, b]) => (a < b ? a + "," + b : b + "," + a)));
const buckets = [[0.5, 1.01], [0.3, 0.5], [0.15, 0.3], [0, 0.15]];
const cnt = buckets.map(() => ({ n: 0, same: 0, rel: 0 }));
let baseN = 0, baseSame = 0;
for (let k = 0; k < S * 4; k++) {
  const i = rand(recs.length); let j = rand(recs.length); if (i === j) continue;
  const jc = jaccard(sig[i], sig[j]); const wi = recs[i].curWord, wj = recs[j].curWord;
  baseN++; if (wi === wj) baseSame++;
  for (let b = 0; b < buckets.length; b++) if (jc >= buckets[b][0] && jc < buckets[b][1]) { const c = cnt[b]; c.n++; if (wi === wj) c.same++; else if (relSet.has(wi < wj ? wi + "," + wj : wj + "," + wi)) c.rel++; break; }
}
const base = baseSame / baseN;
w.write(`B. signature ⇒ target   (random record pairs; P(same target) by signature-similarity bucket)\n`);
w.write(`   base rate P(same target) = ${(100 * base).toFixed(2)}%\n`);
buckets.forEach(([lo, hi], b) => { const c = cnt[b]; if (c.n < 20) { w.write(`   Jaccard [${lo}-${hi})  n=${c.n} (too few)\n`); return; } w.write(`   Jaccard [${lo}-${hi})  n=${c.n}  P(same)=${(100 * c.same / c.n).toFixed(1)}%  lift=${((c.same / c.n) / base).toFixed(1)}×  P(syn/ant)=${(100 * c.rel / c.n).toFixed(2)}%\n`); });
