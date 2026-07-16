"use strict";

// ============================================================================
// benchmark/sigUniqueness.js
//
// Are the signatures too unique for the data — and unique in the WRONG way?
//   - OUT-OF-CLASS uniqueness (different target words → different sigs) is GOOD:
//     it's what lets the model discriminate.
//   - IN-CLASS uniqueness (same target word → still-unique sigs) is FATAL: there
//     is no common part to learn for that word, so nothing generalizes.
// We want LOW in-class distinctness (high in-class sharing) AND HIGH out-of-class
// distinctness. Reports collision structure, train/test overlap, in- vs
// out-of-class pairwise bit-Jaccard, and a common-part metric (does each word own
// a highly-shared, globally-rare bit — shared AND discriminative).
//
// Usage: node benchmark/sigUniqueness.js [dataset.bin]   (CHUNK, KMIN via env)
// ============================================================================

import fs from "node:fs";
import { readDataset, NO_LABEL } from "../src/core/signatures/signatureDataset.js";

const unionSorted = (a, b) => { const out = []; let i = 0, j = 0; while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } } while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out; };
const jaccard = (a, b) => { let i = 0, j = 0, inter = 0; while (i < a.length && j < b.length) { if (a[i] < b[j]) i++; else if (a[i] > b[j]) j++; else { inter++; i++; j++; } } return a.length + b.length - inter === 0 ? 0 : inter / (a.length + b.length - inter); };
const randInt = (n) => Math.floor(Math.random() * n);

const p = process.argv[2] || "data/signatures/wiki_valid_sigs.bin";
const CHUNK = Number(process.env.CHUNK || 60000);
const KMIN = Number(process.env.KMIN || 20); // min instances for a "class" to count
const ds = readDataset(new Uint8Array(fs.readFileSync(p)));
const F = ds.F;
const recs = ds.records.filter((r) => r.scale === 0).slice(0, CHUNK);
const cut = Math.floor(recs.length * 0.8);
const train = recs.slice(0, cut), test = recs.slice(cut);

const heads = {
  "L∪C (GPT→next)": { feat: (r) => unionSorted(r.L, r.C), lab: (r) => r.nextWord },
  "[L|R] (BERT→cur)": { feat: (r) => [...r.L, ...Array.from(r.R, (b) => b + F)], lab: (r) => r.curWord },
};

const w = process.stdout;
w.write(`# sigUniqueness — ${p.split("/").pop()} | F=${F} | train=${train.length} test=${test.length} | class KMIN=${KMIN}\n\n`);

for (const [name, { feat, lab }] of Object.entries(heads)) {
  const fa = train.map(feat);           // sorted feature arrays
  const labs = train.map(lab);

  // --- collision structure + test overlap ---------------------------------
  const sigCount = new Map(); const bitFreq = new Map(); let bitTotal = 0;
  for (let i = 0; i < fa.length; i++) { const f = fa[i]; bitTotal += f.length; sigCount.set(f.join(","), (sigCount.get(f.join(",")) || 0) + 1); for (const b of f) bitFreq.set(b, (bitFreq.get(b) || 0) + 1); }
  let singletons = 0; for (const c of sigCount.values()) if (c === 1) singletons++;
  let exactSeen = 0; for (const r of test) if (sigCount.has(feat(r).join(","))) exactSeen++;

  // --- in-class vs out-of-class pairwise Jaccard --------------------------
  const byLabel = new Map();
  for (let i = 0; i < labs.length; i++) { const l = labs[i]; if (l === NO_LABEL) continue; let g = byLabel.get(l); if (!g) byLabel.set(l, (g = [])); g.push(i); }
  const bigs = [...byLabel.values()].filter((g) => g.length >= KMIN);
  const S = 20000;
  let inSum = 0; for (let s = 0; s < S; s++) { const g = bigs[randInt(bigs.length)]; let x = randInt(g.length), y = randInt(g.length); while (y === x) y = randInt(g.length); inSum += jaccard(fa[g[x]], fa[g[y]]); }
  let outSum = 0; for (let s = 0; s < S; s++) { const i = randInt(fa.length); let j = randInt(fa.length); while (labs[j] === labs[i]) j = randInt(fa.length); outSum += jaccard(fa[i], fa[j]); }
  const inJac = inSum / S, outJac = outSum / S;

  // --- common-part metric: per class, its most-shared bit's in-class coverage
  //     and how globally rare that bit is (lift = shared AND discriminative). ---
  let covSum = 0, liftSum = 0, nc = 0;
  for (const g of bigs.slice(0, 300)) {
    const cnt = new Map(); for (const i of g) for (const b of fa[i]) cnt.set(b, (cnt.get(b) || 0) + 1);
    let best = 0, bestBit = -1; for (const [b, c] of cnt) if (c > best) { best = c; bestBit = b; }
    const cov = best / g.length;                         // fraction of the class sharing that bit
    const glob = bitFreq.get(bestBit) / fa.length;       // that bit's global prevalence
    covSum += cov; liftSum += cov / glob; nc++;
  }

  w.write(`== ${name} ==\n`);
  w.write(`  mean |sig|=${(bitTotal / fa.length).toFixed(1)}  distinct=${(100 * sigCount.size / fa.length).toFixed(1)}%  singletons=${(100 * singletons / sigCount.size).toFixed(1)}%  test-seen-in-train=${(100 * exactSeen / test.length).toFixed(1)}%\n`);
  w.write(`  in-class Jaccard=${inJac.toFixed(3)}   out-class Jaccard=${outJac.toFixed(3)}   ratio(in/out)=${(inJac / outJac).toFixed(2)}   ← want ≫ 1\n`);
  w.write(`  common part: mean top-bit in-class coverage=${(100 * covSum / nc).toFixed(1)}%   mean lift vs global=${(liftSum / nc).toFixed(1)}×   (${nc} classes)\n\n`);
}
