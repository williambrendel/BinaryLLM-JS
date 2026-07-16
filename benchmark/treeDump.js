"use strict";

// ============================================================================
// benchmark/treeDump.js
//
// Investigate ONE CART tree on ONE data chunk (a partition — later one per tree,
// ~100 trees). The split objective is the per-class binary-entropy gain with
// class-weight exponent α (w_l ∝ n_l^{−α}). We sweep α to SEE how the weighting
// shifts which bits are chosen and how balanced/deep the tree is, and we print the
// terminal-reason breakdown (why each leaf stopped) — the §5 data-starvation check.
//
//   SPLIT part={bit ids} k=k*/|p| (coverage)  fire/not sizes  bal=balance%  gain
//   LEAF  n=size  top=majority word  purity%
//
// Usage: node benchmark/treeDump.js [dataset.bin]   (D, Dmax, CHUNK, HEAD via env)
// ============================================================================

import fs from "node:fs";
import { readDataset, NO_LABEL } from "../src/core/signatures/signatureDataset.js";
import { buildCartTree } from "../src/core/predict/index.js";

const unionSorted = (a, b) => {
  const out = []; let i = 0, j = 0;
  while (i < a.length && j < b.length) { const x = a[i], y = b[j]; if (x < y) { out.push(x); i++; } else if (y < x) { out.push(y); j++; } else { out.push(x); i++; j++; } }
  while (i < a.length) out.push(a[i++]); while (j < b.length) out.push(b[j++]); return out;
};

const path = process.argv[2] || "data/signatures/alice_sigs.bin";
const D = Number(process.env.D || 3);
const Dmax = Number(process.env.DMAX || 4);
const CHUNK = Number(process.env.CHUNK || 8000);
const HEAD = process.env.HEAD === "bert" ? "bert" : "gpt";
const ALPHAS = (process.env.ALPHA || "0,1").split(",").map(Number);

const ds = readDataset(new Uint8Array(fs.readFileSync(path)));
const F = ds.F;
const chunk = ds.records.filter((r) => r.scale === 0).slice(0, CHUNK);
// GPT = L∪C (union, F). BERT = [L|R] concatenated (2F): R offset by F so a part
// bit is direction-tagged — "piece b on the left" (b) ≠ "on the right" (b+F).
const head = HEAD === "bert"
  ? { name: "BERT [L|R] → curWord", featuresOf: (r) => [...r.L, ...Array.from(r.R, (b) => b + F)], labelOf: (r) => r.curWord }
  : { name: "GPT [L∪C] → nextWord", featuresOf: (r) => unionSorted(r.L, r.C), labelOf: (r) => r.nextWord };

// tree stats: depth, leaf sizes, mean split balance.
const stats = (node) => {
  if (node.part === undefined) return { depth: 0, leaves: 1, leafSizes: [node.hist.total], balances: [] };
  const a = stats(node.present), b = stats(node.absent);
  const bal = Math.min(node.fireSize, node.notSize) / (node.fireSize + node.notSize) * 2; // 1=50/50, 0=lopsided
  return { depth: 1 + Math.max(a.depth, b.depth), leaves: a.leaves + b.leaves, leafSizes: [...a.leafSizes, ...b.leafSizes], balances: [bal, ...a.balances, ...b.balances] };
};

const dump = (node, depth, tag, out) => {
  const pad = "  ".repeat(depth);
  if (node.part === undefined) {
    let bw = -1, bc = -1; for (const [l, c] of node.hist.counts) if (c > bc) { bc = c; bw = l; }
    out.push(`${pad}${tag}LEAF  n=${node.hist.total}  top=w#${bw}(${bc})  purity=${(100 * bc / node.hist.total).toFixed(0)}%`);
    return;
  }
  const bal = Math.min(node.fireSize, node.notSize) / (node.fireSize + node.notSize) * 2;
  out.push(`${pad}${tag}SPLIT part={${node.part.join(",")}} k=${node.k}/${node.part.length} (cov≥${node.threshold.toFixed(2)})  fire=${node.fireSize} not=${node.notSize}  bal=${(100 * bal).toFixed(0)}%  gain=${node.gain.toFixed(3)}`);
  dump(node.present, depth + 1, "F ", out);
  dump(node.absent, depth + 1, "N ", out);
};

const w = process.stdout;
w.write(`# treeDump — ${path.split("/").pop()} | ${head.name} | chunk=${chunk.length} | D=${D} Dmax=${Dmax}\n`);
for (const alpha of ALPHAS) {
  const tree = buildCartTree(chunk, { featuresOf: head.featuresOf, labelOf: head.labelOf, sMin: 20, alpha, D, Dmax, gainType: process.env.GAIN || "hb" });
  const s = stats(tree.root);
  const meanBal = s.balances.length ? s.balances.reduce((x, y) => x + y, 0) / s.balances.length : 0;
  const sizes = s.leafSizes.slice().sort((a, b) => a - b);
  const tr = tree.terminalReasons;
  w.write(`\n===== α=${alpha}  =====\n`);
  w.write(`leaves=${tree.leaves}  depth=${s.depth}  mean split balance=${(100 * meanBal).toFixed(0)}%  leaf sizes[min/med/max]=${sizes[0]}/${sizes[sizes.length >> 1]}/${sizes[sizes.length - 1]}\n`);
  w.write(`terminal reasons: pure=${tr.pure} smin=${tr.smin} depth=${tr.depth} nosplit=${tr.nosplit}\n`);
  const out = []; dump(tree.root, 0, "", out); w.write(out.join("\n") + "\n");
}
