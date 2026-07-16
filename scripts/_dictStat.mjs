"use strict";
// TEMP: report part counts (total, trained-only, per-kind) for dict text files.
import fs from "node:fs";
import { loadDictText } from "../src/core/parts/dictionary.js";
import { Kind, kindToString } from "../src/core/parts/kind.js";

const CONN = new Set(["-", ",", ".", "$"]);
const isAug = (kind, v) =>
  kind === Kind.Letter ||
  (kind === Kind.Whole && v.length === 1 &&
    ((v >= "a" && v <= "z") || (v >= "0" && v <= "9") || CONN.has(v))) ||
  kind === Kind.Delimiter;

for (const path of process.argv.slice(2)) {
  const dict = loadDictText(fs.readFileSync(path, "latin1"));
  const total = {}, trained = {};
  for (const k of Object.values(Kind)) { total[k] = 0; trained[k] = 0; }
  let nTotal = 0, nTrained = 0;
  for (const { kind, value } of dict.allParts()) {
    total[kind]++; nTotal++;
    if (!isAug(kind, value)) { trained[kind]++; nTrained++; }
  }
  const per = (o) => Object.values(Kind)
    .map((k) => `${kindToString(k)}=${o[k]}`).join(" ");
  process.stdout.write(
    `${path}\n  total F=${nTotal}  trained(excl. augmented)=${nTrained}\n` +
    `  total:   ${per(total)}\n  trained: ${per(trained)}\n\n`);
}
