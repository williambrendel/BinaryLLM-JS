"use strict";
// TEMP: per-kind counts + part-length distribution for dict text files.
//   node scripts/_lenStat.mjs <A.txt> <B.txt> ...
import fs from "node:fs";
import { loadDictText } from "../src/core/parts/dictionary.js";
import { Kind, kindToString } from "../src/core/parts/kind.js";

const CONN = new Set(["-", ",", ".", "$"]);
const isAug = (kind, v) =>
  kind === Kind.Letter ||
  (kind === Kind.Whole && v.length === 1 &&
    ((v >= "a" && v <= "z") || (v >= "0" && v <= "9") || CONN.has(v))) ||
  kind === Kind.Delimiter;

const dicts = process.argv.slice(2).map((p) => {
  const d = loadDictText(fs.readFileSync(p, "latin1"));
  const kindCount = {}; for (const k of Object.values(Kind)) kindCount[k] = 0;
  // len[kind][length] for start/mid/end; wholeLen[length]
  const len = { [Kind.Start]: {}, [Kind.Mid]: {}, [Kind.End]: {} };
  const wholeLen = {};
  for (const { kind, value } of d.allParts()) {
    if (isAug(kind, value)) continue;
    kindCount[kind]++;
    if (kind === Kind.Start || kind === Kind.Mid || kind === Kind.End)
      len[kind][value.length] = (len[kind][value.length] || 0) + 1;
    if (kind === Kind.Whole) wholeLen[value.length] = (wholeLen[value.length] || 0) + 1;
  }
  return { name: p.split("/").pop(), kindCount, len, wholeLen };
});

const pad = (s, w) => String(s).padStart(w);
const col = 14;

// --- #2 kind totals ---
process.stdout.write("\n=== #2  KIND TOTALS (trained, excl. augmented) ===\n");
process.stdout.write("kind".padEnd(9) + dicts.map((d) => pad(d.name, col)).join("") + "\n");
for (const k of [Kind.Whole, Kind.Start, Kind.Mid, Kind.End, Kind.Letter, Kind.Delimiter])
  process.stdout.write(kindToString(k).padEnd(9) + dicts.map((d) => pad(d.kindCount[k], col)).join("") + "\n");
process.stdout.write("TOTAL".padEnd(9) +
  dicts.map((d) => pad(Object.values(d.kindCount).reduce((a, b) => a + b, 0), col)).join("") + "\n");

// --- #3 length distribution for start / mid / end ---
for (const [kind, kname] of [[Kind.Start, "START"], [Kind.Mid, "MID"], [Kind.End, "END"]]) {
  process.stdout.write(`\n=== #3  ${kname} parts by length ===\n`);
  process.stdout.write("length".padEnd(9) + dicts.map((d) => pad(d.name, col)).join("") + "\n");
  for (let L = 7; L >= 2; L--)
    process.stdout.write((L + "-char").padEnd(9) + dicts.map((d) => pad(d.len[kind][L] || 0, col)).join("") + "\n");
}

// combined start+mid+end by length
process.stdout.write("\n=== #3  START+MID+END combined by length ===\n");
process.stdout.write("length".padEnd(9) + dicts.map((d) => pad(d.name, col)).join("") + "\n");
for (let L = 7; L >= 2; L--)
  process.stdout.write((L + "-char").padEnd(9) +
    dicts.map((d) => pad((d.len[Kind.Start][L] || 0) + (d.len[Kind.Mid][L] || 0) + (d.len[Kind.End][L] || 0), col)).join("") + "\n");

// whole length distribution (2..16)
process.stdout.write("\n=== WHOLE atoms by length (context) ===\n");
process.stdout.write("length".padEnd(9) + dicts.map((d) => pad(d.name, col)).join("") + "\n");
const maxW = Math.max(...dicts.flatMap((d) => Object.keys(d.wholeLen).map(Number)), 2);
for (let L = 2; L <= maxW; L++) {
  if (!dicts.some((d) => d.wholeLen[L])) continue;
  process.stdout.write((L + "-char").padEnd(9) + dicts.map((d) => pad(d.wholeLen[L] || 0, col)).join("") + "\n");
}
