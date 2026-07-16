"use strict";

// ============================================================================
// benchmark/e0_contrastTail.js  —  BLOCKING experiment E0 (prior-only slice, v2)
//
// P0: for a residual-aligned set A, do CONTEXT atoms concentrate in A vs Ā?
//
// v2 fixes two flaws in v1:
//   (1) DIRECTIONS are now TARGETED PRODUCTIVE AFFIXES (ing:End, un:Start, …), not
//       max-q(1−q) atoms. max-variance ≠ predictive; end:s being flat says nothing
//       about end:ing. The slot-channel claim rests on affixes, so probe affixes.
//   (2) The tail is measured by a Z-SCORE against a SIZE-MATCHED NULL, not a fixed
//       |cst| threshold. For a direction with |A|=k, draw R random k-subsets, take
//       null_sd[a] = std of cst over draws, z[a] = cst[a]/null_sd[a]. This is
//       scale-free (context-atom vs target-atom scale stops mattering) and |A|-aware
//       (a small A has a wide null, so 0.35 @ |A|=135 is correctly discounted).
//
// A "tail" is z ≫ 0, not |cst| > θ. T-01 becomes a significance test.
//
// Usage: node benchmark/e0_contrastTail.js <dict> <corpus>   (FUZZY, RADIUS, R via env)
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { streamSignatureChunks } from "../src/core/signatures/streamSignatures.js";
import { encodeWord, fuzzyEncodeWord } from "../src/core/signatures/wordEncoder.js";
import { NO_LABEL } from "../src/core/signatures/signatureDataset.js";
import { Kind, kindToString } from "../src/core/parts/kind.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };
const mkRng = (s) => () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const rng = mkRng(2024);

const MCOUNT = 30;   // an atom must appear in >= this many samples globally to be a z-candidate
const NCAND = 300;   // rank this many atoms (by |cst|) as z-candidates
const AFFIX_END = ["ing", "ed", "tion", "ly", "er", "ness", "ment", "ity", "al", "ous", "ive", "ers", "ate", "ance", "ism"];
const AFFIX_START = ["un", "re", "de", "pre", "dis", "over", "sub", "inter", "trans", "anti", "mis", "non", "con", "ex"];

const main = async () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const FUZZY = process.env.FUZZY === "1";
  const radius = Number(process.env.RADIUS || 1);
  const R = Number(process.env.R || 100);
  const dict = loadDict(dictArg);
  const F = dict.size();
  const bagOf = FUZZY ? fuzzyEncodeWord : encodeWord;
  const parts = dict.allParts();
  const idByKV = new Map(); parts.forEach((p, i) => idByKV.set(`${p.kind}:${p.value}`, i));

  // ---- stream records ---------------------------------------------------------
  const vocabIndex = new Map(), vocab = [];
  const recs = [];
  for await (const { records } of streamSignatureChunks(dict, corpusArg, { chunkSize: 20000, radius, fuzzy: FUZZY, vocabIndex, vocab })) {
    for (const r of records) if (r.nextWord !== NO_LABEL) recs.push({ L: r.L, C: r.C, tgt: r.nextWord });
  }
  const N = recs.length;
  const tgtBag = vocab.map((w) => bagOf(dict, Buffer.from(w).toString("latin1").toLowerCase()));
  const cnt = new Float64Array(vocab.length);
  for (const r of recs) cnt[r.tgt]++;

  // ---- context features of [L | C]  (C offset by F) ---------------------------
  const ctxArr = recs.map((r) => { const a = []; for (const x of r.L) a.push(x); for (const x of r.C) a.push(x + F); return a; });
  const ctxLabel = (f) => { const p = parts[f < F ? f : f - F]; return `${f < F ? "L" : "C"}:${p ? kindToString(p.kind) + ":" + p.value : "?"}`; };
  // global count M_a per context feature (presence, over all N)
  const M = new Float64Array(2 * F);
  for (const a of ctxArr) for (const f of a) M[f]++;

  // ---- enrichment z vs a size-matched null (exact hypergeometric) -------------
  // For atom a: obs = in-A count, exp = M_a·k/N; z = (obs−exp)/sd_hypergeom.
  // Admissibility: exp ≥ 5 (a small A is only testable against COMMON atoms — the
  // |A|-awareness). This is the analytic form of "draw R random k-subsets".
  const EXPMIN = 5, ZSIG = 3;
  const analyze = (Aidx) => {
    const k = Aidx.length, nB = N - k;
    const cA = new Float64Array(2 * F); for (const i of Aidx) for (const f of ctxArr[i]) cA[f]++;
    const rows = [];
    for (let f = 0; f < 2 * F; f++) {
      const Ma = M[f]; if (Ma === 0) continue;
      const exp = Ma * k / N; if (exp < EXPMIN) continue;         // expected-count floor
      const p = Ma / N, varH = k * p * (1 - p) * (N - k) / (N - 1);
      const z = (cA[f] - exp) / Math.sqrt(Math.max(varH, 1e-12));
      const cst = cA[f] / k - (Ma - cA[f]) / nB;
      rows.push({ f, cst, z });
    }
    rows.sort((a, b) => b.z - a.z);
    const sig = rows.filter((r) => r.z >= ZSIG);
    const distinctVals = new Set(sig.map((r) => { const p = parts[r.f < F ? r.f : r.f - F]; return p ? p.value : r.f; }));
    return { k, nTestable: rows.length, nSig: sig.length, nSigDistinct: distinctVals.size, rows };
  };

  const dirs = [];
  // targeted affixes
  for (const v of AFFIX_START) { const id = idByKV.get(`${Kind.Start}:${v}`); if (id !== undefined) dirs.push({ src: "affix", tag: `tgt Start:${v}`, id }); }
  for (const v of AFFIX_END) { const id = idByKV.get(`${Kind.End}:${v}`); if (id !== undefined) dirs.push({ src: "affix", tag: `tgt End:${v}`, id }); }
  for (const d of dirs) { const A = []; for (let i = 0; i < N; i++) if (tgtBag[recs[i].tgt].includes(d.id)) A.push(i); d.A = A; }
  // label directions: 2 frequent + 3 content, to test the z-inversion bet
  const wid = (s) => vocabIndex.get(s);
  for (const s of ["the", "of", "government", "east", "day"]) { const c = wid(s); if (c === undefined) continue; const A = []; for (let i = 0; i < N; i++) if (recs[i].tgt === c) A.push(i); dirs.push({ src: "label", tag: `next="${s}"`, A }); }

  const w = process.stdout;
  w.write(`# E0 v2 (0 gates, targeted affixes, hypergeometric enrichment z) — dict=${path.basename(dictArg)} corpus=${path.basename(corpusArg)} | enc=${FUZZY ? "FUZZY" : "sparse"} r=${radius} | N=${N}\n`);
  w.write(`# z = (obs−exp)/sd, exp=M_a·|A|/N ≥ 5 (else atom not testable at this |A|).  TAIL = # atoms z≥3.  distinct = # distinct VALUES among them (collinearity check).\n\n`);
  for (const d of dirs) {
    if (!d.A.length) { w.write(`[${d.src}] ${d.tag}: |A|=0 (affix absent in corpus targets under this encoding)\n\n`); continue; }
    const { k, nTestable, nSig, nSigDistinct, rows } = analyze(d.A);
    const top = rows.slice(0, 5).map((r) => `${ctxLabel(r.f)}[cst=${r.cst.toFixed(2)},z=${r.z.toFixed(1)}]`).join("  ");
    w.write(`[${d.src.padEnd(5)}] ${d.tag.padEnd(16)}  |A|=${String(k).padStart(6)} (${(100 * k / N).toFixed(1)}%)   testable=${nTestable}   TAIL(z≥3)=${nSig} (distinct=${nSigDistinct})   maxZ=${rows.length ? rows[0].z.toFixed(1) : "—"}\n`);
    w.write(`         ${top || "(no testable atoms — |A| too small for exp≥5)"}\n\n`);
  }
};

main();
