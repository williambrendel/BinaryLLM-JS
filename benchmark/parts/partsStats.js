"use strict";

/**
 * @file partsStats.js
 * @brief Distribution stats for trained part dictionaries — counts per
 * (kind × length), the short-part share, and how much of the 2/3-letter
 * substring SPACE each dictionary actually covers. Used to test whether one
 * method simply stocks more short filler parts than another.
 *
 * Usage: node scripts/partsStats.js <dictA.txt> <dictB.txt> [...]
 */

import fs from "node:fs";
import { Kind } from "../../src/core/parts/kind.js";
import { loadDictText } from "../../src/core/parts/dictionary.js";

const KINDS = [Kind.Whole, Kind.Start, Kind.Mid, Kind.End];
const NAME = { [Kind.Whole]: "whole", [Kind.Start]: "start", [Kind.Mid]: "mid", [Kind.End]: "end" };

/** @description Trained word parts only (len ≥ 2; Letters/Delims/singletons excluded). */
const trained = (dict) =>
  dict.allParts().filter((p) => KINDS.includes(p.kind) && p.value.length >= 2);

/** @description Distinct values of a kind at a given length. */
const distinctAt = (parts, kind, len) => {
  const s = new Set();
  for (const p of parts) if (p.kind === kind && p.value.length === len) s.add(p.value);
  return s.size;
};

const analyze = (label, path) => {
  const parts = trained(loadDictText(fs.readFileSync(path, "latin1")));
  const maxLen = parts.reduce((m, p) => Math.max(m, p.value.length), 0);
  const grid = {}; for (const k of KINDS) grid[k] = new Array(maxLen + 1).fill(0);
  for (const p of parts) grid[p.kind][p.value.length]++;

  console.log(`\n=== ${label}  (${path.split("/").pop()}) — ${parts.length} trained parts ===`);
  let head = "  kind    "; for (let L = 2; L <= maxLen; L++) head += String(L).padStart(6); head += "   total    %len2";
  console.log(head);
  for (const k of KINDS) {
    let row = "  " + NAME[k].padEnd(6); let tot = 0;
    for (let L = 2; L <= maxLen; L++) { row += String(grid[k][L]).padStart(6); tot += grid[k][L]; }
    const pct2 = tot ? (100 * grid[k][2] / tot).toFixed(1) : "0.0";
    console.log(row + String(tot).padStart(8) + String(pct2).padStart(9));
  }
  const total = parts.length;
  const n2 = parts.filter((p) => p.value.length === 2).length;
  const n3 = parts.filter((p) => p.value.length === 3).length;
  console.log(`  2-letter parts: ${n2} (${(100 * n2 / total).toFixed(1)}%)   ≤3-letter: ${n2 + n3} (${(100 * (n2 + n3) / total).toFixed(1)}%)`);
  // Coverage of the lowercase substring space (26^L) for the tiling-relevant kinds.
  const sp2 = 26 * 26, sp3 = 26 * 26 * 26;
  for (const k of [Kind.Start, Kind.Mid, Kind.End]) {
    const c2 = distinctAt(parts, k, 2), c3 = distinctAt(parts, k, 3);
    console.log(`  ${NAME[k]} space coverage: 2-letter ${c2}/${sp2} (${(100 * c2 / sp2).toFixed(1)}%)   3-letter ${c3}/${sp3} (${(100 * c3 / sp3).toFixed(2)}%)`);
  }
  return { label, total, n2, grid, parts };
};

const files = process.argv.slice(2);
if (files.length < 1) { console.error("usage: node scripts/partsStats.js <dictA> <dictB> [...]"); process.exit(2); }
const results = files.map((f) => analyze(f.split("/").pop().replace(".txt", ""), f));

if (results.length === 2) {
  const [a, b] = results;
  console.log(`\n=== ${a.label} vs ${b.label} — 2-letter parts per type ===`);
  console.log("  type     " + a.label.padStart(10) + b.label.padStart(12) + "     ratio");
  for (const k of KINDS) {
    const av = a.grid[k][2] || 0, bv = b.grid[k][2] || 0;
    const ratio = av ? (bv / av).toFixed(1) : (bv ? "∞" : "—");
    console.log("  " + NAME[k].padEnd(6) + String(av).padStart(11) + String(bv).padStart(12) + String(ratio).padStart(10) + "×");
  }
  const ratioAll = a.n2 ? (b.n2 / a.n2).toFixed(2) : "—";
  console.log(`  ALL 2-letter: ${a.label}=${a.n2}  ${b.label}=${b.n2}  ratio=${ratioAll}×`);
}
