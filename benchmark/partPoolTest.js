"use strict";

// ============================================================================
// benchmark/partPoolTest.js  (the r1-parts-as-r5-pool experiment)
//
// Build a tree on r1 and on r5 (same records, both encodings). Then build a tree
// on r5 whose ONLY candidate splitters are r1's discovered parts. For each, report
// avg split IG (mean per-node entropy reduction) and global IG (H(root) − weighted
// mean leaf entropy). If r5 restricted to r1's parts beats r5-native, discovery —
// not the parts — is the failure: the good parts were available and greedy on r5's
// own landscape didn't pick them.
//
// Usage: node benchmark/partPoolTest.js [maxRecords]   (SMIN, DMAX via env)
// ============================================================================

import fs from "node:fs";
import { readDataset, NO_LABEL } from "../src/core/signatures/signatureDataset.js";
import { buildCartTree, buildPoolTree } from "../src/core/predict/index.js";

const unionSorted = (a, b) => { const out = []; let i = 0, j = 0; while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } } while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out; };
const featOf = (r) => unionSorted(r.L, r.C);
const labOf = (r) => r.nextWord;
const N = Number(process.argv[2] || 20000);
const SMIN = Number(process.env.SMIN || 20), DMAX = Number(process.env.DMAX || 25);
const load = (f) => readDataset(new Uint8Array(fs.readFileSync(`data/signatures/${f}.bin`))).records.filter((r) => r.scale === 0).slice(0, N);
const r1 = load("wiki_valid_r1"), r5 = load("wiki_valid_r5");

const H = (hist) => { let h = 0; for (const c of hist.counts.values()) { const p = c / hist.total; h -= p * Math.log(p); } return h; };
const collectParts = (node, pool) => { if (node.part === undefined) return; pool.push({ bits: node.part, k: node.k }); collectParts(node.present, pool); collectParts(node.absent, pool); };

// avg split IG (per internal node) and global IG (= Σ weighted split IG = Hroot − leafH).
const metrics = (tree) => {
  const igs = []; const leaves = [];
  const walk = (n) => {
    if (n.part === undefined) { leaves.push(n.hist); return; }
    const w = n.present.hist.total / n.hist.total, w2 = n.absent.hist.total / n.hist.total;
    igs.push(H(n.hist) - w * H(n.present.hist) - w2 * H(n.absent.hist));
    walk(n.present); walk(n.absent);
  };
  walk(tree.root);
  const rootTotal = tree.root.hist.total;
  const Hroot = H(tree.root.hist);
  const leafH = leaves.reduce((s, l) => s + (l.total / rootTotal) * H(l), 0);
  let modal = 0; for (const l of leaves) { let top = 0; for (const c of l.counts.values()) if (c > top) top = c; modal += top / l.total; }
  return { splits: igs.length, avgSplitIG: igs.reduce((a, b) => a + b, 0) / igs.length, globalIG: Hroot - leafH, leafH, modal: 100 * modal / leaves.length };
};

const tR1 = buildCartTree(r1, { featuresOf: featOf, labelOf: labOf, sMin: SMIN, D: 3, Dmax: DMAX });
const tR5 = buildCartTree(r5, { featuresOf: featOf, labelOf: labOf, sMin: SMIN, D: 3, Dmax: DMAX });
const poolR1 = []; collectParts(tR1.root, poolR1);
const tR5pool = buildPoolTree(r5, { featuresOf: featOf, labelOf: labOf, pool: poolR1, sMin: SMIN, Dmax: DMAX });

const row = (name, t, m) => `${name.padEnd(28)}  leaves=${String(t.leaves).padStart(4)}  splits=${String(m.splits).padStart(4)}  avgSplitIG=${m.avgSplitIG.toFixed(4)}  globalIG=${m.globalIG.toFixed(3)}  leafH=${m.leafH.toFixed(3)}  modal=${m.modal.toFixed(1)}%`;
process.stdout.write(`# partPoolTest — records=${r1.length} sMin=${SMIN} | grower=IG/α0 | r1 pool=${poolR1.length} parts\n`);
process.stdout.write(row("r1 tree (r1 parts, r1 data)", tR1, metrics(tR1)) + "\n");
process.stdout.write(row("r5 tree (r5 parts, r5 data)", tR5, metrics(tR5)) + "\n");
process.stdout.write(row("r5 tree (r1 POOL, r5 data)", tR5pool, metrics(tR5pool)) + "\n");
