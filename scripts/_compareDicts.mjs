"use strict";
// TEMP: compare two dict text files — per-Kind counts, overlap (Jaccard),
// and sample parts unique to each. Trained (non-augmented) parts only.
//   node scripts/_compareDicts.mjs <A.txt> <B.txt>
import fs from "node:fs";
import { loadDictText } from "../src/core/parts/dictionary.js";
import { Kind, kindToString } from "../src/core/parts/kind.js";

const CONN = new Set(["-", ",", ".", "$"]);
const isAug = (kind, v) =>
  kind === Kind.Letter ||
  (kind === Kind.Whole && v.length === 1 &&
    ((v >= "a" && v <= "z") || (v >= "0" && v <= "9") || CONN.has(v))) ||
  kind === Kind.Delimiter;

const bucketize = (path) => {
  const dict = loadDictText(fs.readFileSync(path, "latin1"));
  const b = {};
  for (const k of Object.values(Kind)) b[k] = new Set();
  for (const { kind, value } of dict.allParts()) if (!isAug(kind, value)) b[kind].add(value);
  return b;
};

const [pa, pb] = process.argv.slice(2);
const A = bucketize(pa), B = bucketize(pb);
const nameA = pa.split("/").pop(), nameB = pb.split("/").pop();
const sample = (set, other, n = 8) =>
  [...set].filter((v) => !other.has(v)).slice(0, n).map((v) => JSON.stringify(v)).join(", ");

let tA = 0, tB = 0, tI = 0;
process.stdout.write(`\nA = ${nameA}\nB = ${nameB}\n\n`);
process.stdout.write(
  "kind".padEnd(9) + "|A|".padStart(8) + "|B|".padStart(8) +
  "shared".padStart(8) + "A-only".padStart(8) + "B-only".padStart(8) + "  Jaccard\n");
for (const k of Object.values(Kind)) {
  const a = A[k], b = B[k];
  let inter = 0; for (const v of a) if (b.has(v)) inter++;
  const uni = a.size + b.size - inter;
  const j = uni ? (inter / uni) : 1;
  tA += a.size; tB += b.size; tI += inter;
  process.stdout.write(
    kindToString(k).padEnd(9) + String(a.size).padStart(8) + String(b.size).padStart(8) +
    String(inter).padStart(8) + String(a.size - inter).padStart(8) +
    String(b.size - inter).padStart(8) + "  " + j.toFixed(3) + "\n");
}
const uniAll = tA + tB - tI;
process.stdout.write(
  "TOTAL".padEnd(9) + String(tA).padStart(8) + String(tB).padStart(8) +
  String(tI).padStart(8) + String(tA - tI).padStart(8) + String(tB - tI).padStart(8) +
  "  " + (tI / uniAll).toFixed(3) + "\n");

process.stdout.write(`\nSample parts in A(${nameA}) but NOT B:\n`);
for (const k of Object.values(Kind)) {
  const s = sample(A[k], B[k]); if (s) process.stdout.write(`  ${kindToString(k)}: ${s}\n`);
}
process.stdout.write(`\nSample parts in B(${nameB}) but NOT A:\n`);
for (const k of Object.values(Kind)) {
  const s = sample(B[k], A[k]); if (s) process.stdout.write(`  ${kindToString(k)}: ${s}\n`);
}
