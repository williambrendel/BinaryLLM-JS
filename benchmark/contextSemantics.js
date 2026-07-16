"use strict";

// ============================================================================
// benchmark/contextSemantics.js
//
// Does L / R carry distributional SEMANTICS, or only a syntactic language-prior —
// and how does that trade against uniqueness as the window grows? For the L band
// and the R band SEPARATELY, per dataset (= per window):
//
//   UNIQUENESS  — same-word vs random per-occurrence Jaccard (clustering by target),
//     and out-of-class collision rate at match tolerances θ (fraction of random
//     cross-word pairs that "match" at Jaccard ≥ θ). Low θ = loose match.
//   SEMANTICS   — aggregate per-word profiles (distributional vectors); cosine lift
//     of SYNONYM and ANTONYM pairs vs random. Lift ≫ 1 ⇒ similar band ⇒ similar
//     meaning; lift ≈ 1 ⇒ the band is spelling/syntax, not meaning.
//
// Run across windows (r1…r∞): if synonym lift RISES with the window, wider radius
// buys semantics (extend it, with loose matching to keep collisions); if it stays
// flat, L/R is just a language prior and no radius rescues it.
//
// Usage: node benchmark/contextSemantics.js <dataset.bin>   (MINOCC via env)
// ============================================================================

import fs from "node:fs";
import { readDataset } from "../src/core/signatures/signatureDataset.js";

const jaccard = (a, b) => { let i = 0, j = 0, x = 0; while (i < a.length && j < b.length) { if (a[i] < b[j]) i++; else if (a[i] > b[j]) j++; else { x++; i++; j++; } } const u = a.length + b.length - x; return u === 0 ? 0 : x / u; };
const cosine = (a, b, na, nb) => { let d = 0; const [s, l] = a.size < b.size ? [a, b] : [b, a]; for (const [k, v] of s) { const o = l.get(k); if (o) d += v * o; } return na && nb ? d / (na * nb) : 0; };
const rand = (n) => Math.floor(Math.random() * n);

const SYN = [["also", "additionally"], ["however", "although"], ["first", "initial"], ["large", "major"], ["area", "region"], ["city", "town"], ["war", "conflict"], ["group", "team"], ["known", "called"], ["used", "employed"], ["made", "created"], ["began", "started"], ["many", "several"], ["big", "large"], ["important", "significant"], ["often", "frequently"], ["build", "construct"], ["end", "finish"], ["near", "close"], ["old", "ancient"], ["new", "modern"], ["fast", "quick"], ["show", "display"], ["help", "aid"]];
const ANT = [["first", "last"], ["high", "low"], ["old", "new"], ["large", "small"], ["long", "short"], ["more", "less"], ["before", "after"], ["north", "south"], ["east", "west"], ["increase", "decrease"], ["early", "late"], ["win", "loss"], ["male", "female"], ["good", "bad"], ["major", "minor"]];

const p = process.argv[2] || "data/signatures/wiki_valid_r5.bin";
const MINOCC = Number(process.env.MINOCC || 20);
const S = Number(process.env.SAMPLE || 30000);
const ds = readDataset(new Uint8Array(fs.readFileSync(p)));
const recs = ds.records.filter((r) => r.scale === 0);
const id = new Map(); ds.vocab.forEach((wd, i) => id.set(typeof wd === "string" ? wd : Buffer.from(wd).toString("latin1"), i));

const byWord = new Map();
for (let i = 0; i < recs.length; i++) { const wd = recs[i].curWord; let g = byWord.get(wd); if (!g) byWord.set(wd, (g = [])); g.push(i); }
const okWord = (wd) => { const w = id.get(wd); return w !== undefined && (byWord.get(w)?.length || 0) >= MINOCC ? w : -1; };
const qual = (list) => list.map(([a, b]) => [okWord(a), okWord(b)]).filter(([a, b]) => a >= 0 && b >= 0);
const synP = qual(SYN), antP = qual(ANT);
const pool = [...byWord.entries()].filter(([, g]) => g.length >= MINOCC).map(([w]) => w);

const analyze = (bandOf, name, out) => {
  // per-occurrence uniqueness + collision at tolerances
  const occJac = (wa, wb, s) => { const A = byWord.get(wa), B = byWord.get(wb); let sum = 0, m3 = 0, m5 = 0; for (let k = 0; k < s; k++) { const jc = jaccard(bandOf(recs[A[rand(A.length)]]), bandOf(recs[B[rand(B.length)]])); sum += jc; if (jc >= 0.3) m3++; if (jc >= 0.5) m5++; } return { j: sum / s, m3: m3 / s, m5: m5 / s }; };
  let sameJ = 0; for (const w of pool) sameJ += occJac(w, w, 200).j; sameJ /= pool.length;
  let rj = 0, rm3 = 0, rm5 = 0; for (let k = 0; k < S; k++) { const a = pool[rand(pool.length)]; let b = pool[rand(pool.length)]; while (b === a) b = pool[rand(pool.length)]; const jc = jaccard(bandOf(recs[byWord.get(a)[rand(byWord.get(a).length)]]), bandOf(recs[byWord.get(b)[rand(byWord.get(b).length)]])); rj += jc; if (jc >= 0.3) rm3++; if (jc >= 0.5) rm5++; }
  rj /= S; rm3 /= S; rm5 /= S;

  // aggregate per-word distributional profiles (normalized count vectors)
  const need = new Set([...pool]);
  const prof = new Map(); const norm = new Map();
  for (const w of need) prof.set(w, new Map());
  for (let i = 0; i < recs.length; i++) { const w = recs[i].curWord; const m = prof.get(w); if (!m) continue; for (const b of bandOf(recs[i])) m.set(b, (m.get(b) || 0) + 1); }
  for (const [w, m] of prof) { let n = 0; for (const v of m.values()) n += v * v; norm.set(w, Math.sqrt(n)); }
  const cosPairs = (wps) => { let s = 0; for (const [a, b] of wps) s += cosine(prof.get(a), prof.get(b), norm.get(a), norm.get(b)); return s / wps.length; };
  let rc = 0; for (let k = 0; k < 4000; k++) { const a = pool[rand(pool.length)]; let b = pool[rand(pool.length)]; while (b === a) b = pool[rand(pool.length)]; rc += cosine(prof.get(a), prof.get(b), norm.get(a), norm.get(b)); } rc /= 4000;
  const synC = cosPairs(synP), antC = cosPairs(antP);

  out.push(`  ${name}:  uniq same/rand Jac=${sameJ.toFixed(3)}/${rj.toFixed(3)} (ratio ${(sameJ / rj).toFixed(1)}×)   cross-collide θ.3=${(100 * rm3).toFixed(1)}% θ.5=${(100 * rm5).toFixed(1)}%`);
  out.push(`        profile cosine  syn=${synC.toFixed(3)}  ant=${antC.toFixed(3)}  rand=${rc.toFixed(3)}   → syn/rand=${(synC / rc).toFixed(2)}×  ant/rand=${(antC / rc).toFixed(2)}×`);
};

const out = [];
out.push(`# contextSemantics — ${p.split("/").pop()} | records=${recs.length} MINOCC=${MINOCC} | syn=${synP.length} ant=${antP.length} pool=${pool.length}`);
analyze((r) => r.L, "L", out);
analyze((r) => r.R, "R", out);
process.stdout.write(out.join("\n") + "\n");
