"use strict";

// ============================================================================
// benchmark/wordMask.js — WORD-level masking, PART-level representation.
// A word is kept/dropped by its own UNIGRAM frequency (≤ δ = keep); if kept, ALL its
// sub-word parts enter the feature (so masonry → mas·on·ry survives for morphological
// generalization; the → dropped whole). Never mask at the part level.
// Bands: [L | L1∪L2 | C]  — C=word t-1; middle = OR(t-2,t-3); far L = OR(t-4..t-D).
// Metrics: masked coverage (in-sample + held-out), masked perfect-coverage detector + FP samples.
// Usage: TARGET=bank R=5 node --max-old-space-size=8192 benchmark/wordMask.js english.txt <corpus>
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { encodeWord, fuzzyEncodeWord } from "../src/core/signatures/wordEncoder.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGET = process.env.TARGET || "bank";
const R = Number(process.env.R || 5);
const D = R + 1;
// band layouts: how offsets d=1..D map to bands (d=1 is t-1, the immediate predecessor)
const LAYOUT = process.env.LAYOUT || "L1L2";
// band(d) returns the LIST of band indices a word at offset d belongs to (overlap ⇒ >1 band)
const LAYOUTS = {
  LC:   { n: 2, band: (d) => (d === 1 ? [1] : [0]) },                        // [L | C]
  L1:   { n: 3, band: (d) => (d === 1 ? [2] : d === 2 ? [1] : [0]) },        // [L | L1 | C]
  L1L2: { n: 3, band: (d) => (d === 1 ? [2] : d <= 3 ? [1] : [0]) },         // [L | L1∪L2 | C]
  OVL:  { n: 3, band: (d) => (d === 1 ? [2] : d === 2 ? [2, 1] : d === 3 ? [1, 0] : [0]) }, // [L∪L2 | L2∪L1 | L1∪C]
};
const { n: NB, band: bandFn } = LAYOUTS[LAYOUT];

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size(), DIM = NB * F;
  const ENC = process.env.ENCODER === "fuzzy" ? fuzzyEncodeWord : encodeWord;
  const pcache = new Map();
  const partsOf = (w) => { let a = pcache.get(w); if (!a) { a = [...new Set(ENC(dict, w))]; pcache.set(w, a); } return a; };
  const sents = [];
  const cnt = new Map(); let Ntok = 0;
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const toks = tokenizeStream(line.slice(seg.start, seg.end));
    const ws = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) { sents.push(ws); for (const x of ws) { cnt.set(x, (cnt.get(x) || 0) + 1); Ntok++; } }
  }
  const wf = (w) => (cnt.get(w) || 0) / Ntok;   // WORD unigram frequency (the mask key)

  const bandOf = bandFn;
  // feature = per-context-word entries {wf, ids(band-offset parts)}; masking applied later by δ on wf
  const feat = (words, t) => { const e = []; for (let d = 1; d <= D; d++) { const q = t - d; if (q < 0) break; const w = words[q], parts = partsOf(w), ids = []; for (const b of bandOf(d)) { const off = b * F; for (const a of parts) ids.push(a + off); } e.push({ wf: wf(w), ids, w }); } return e; };

  const POS = [], NEG = []; let negSeen = 0, negTot = 0;
  for (const words of sents) for (let t = 1; t < words.length; t++) if (words[t] !== TARGET) negTot++;
  const step = Math.max(1, Math.floor(negTot / 3000));
  for (const words of sents) for (let t = 1; t < words.length; t++) { const f = feat(words, t); if (words[t] === TARGET) POS.push(f); else { if (negSeen % step === 0) NEG.push(f); negSeen++; } }

  const stampY = new Int32Array(DIM).fill(-1); let ypid = 0;
  const rareIds = (cand, d) => { ypid++; const y = []; for (const e of cand) if (e.wf <= d) for (const id of e.ids) if (stampY[id] !== ypid) { stampY[id] = ypid; y.push(id); } return y; };
  const buildP = (SET, d) => { const p = new Uint8Array(DIM); let n = 0; for (const cand of SET) for (const e of cand) if (e.wf <= d) for (const id of e.ids) if (!p[id]) { p[id] = 1; n++; } return { p, n }; };
  const covMasked = (SET, d, P) => { let s = 0, s2 = 0, nn = 0, empty = 0; for (const cand of SET) { const y = rareIds(cand, d); if (y.length === 0) { empty++; continue; } let k = 0; for (const id of y) if (P[id]) k++; const c = k / y.length; s += c; s2 += c * c; nn++; } const m = nn ? s / nn : 0; return { m, sd: nn ? Math.sqrt(Math.max(0, s2 / nn - m * m)) : 0, empty, tot: nn + empty }; };

  const w = process.stdout;
  w.write(`# wordMask LAYOUT=${LAYOUT} (${NB} bands) — ${path.basename(corpusArg)} | word-level mask, part repr | F=${F} DIM=${DIM}\n`);
  w.write(`# POS n=${POS.length}  NEG n=${NEG.length}   Ntok=${Ntok}  |vocab|=${cnt.size}\n`);

  const DELS = [Infinity, 0.01, 0.005, 0.002, 0.001, 0.0005];
  w.write(`\n── masked coverage (in-sample, δ on WORD freq) ──\n   δ       |p'|   cov pos   cov neg (rare)     ratio\n`);
  for (const d of DELS) { const { p, n } = buildP(POS, d); const cp = covMasked(POS, d, p), cn = covMasked(NEG, d, p); w.write(`   ${(d === Infinity ? "∞" : d).toString().padEnd(7)} ${String(n).padStart(6)}   ${cp.m.toFixed(4)}   ${cn.m.toFixed(4)} ± ${cn.sd.toFixed(4)}   ${(cp.m / (cn.m || 1e-9)).toFixed(2)}\n`); }

  const Pb = POS.filter((_, i) => i % 2 === 0), Pt = POS.filter((_, i) => i % 2 === 1);
  w.write(`\n── HELD-OUT (p from ${Pb.length} pos) ──\n   δ       cov pos-HO   %∅   cov neg      %∅   ratio\n`);
  for (const d of DELS) { const { p } = buildP(Pb, d); const cp = covMasked(Pt, d, p), cn = covMasked(NEG, d, p); w.write(`   ${(d === Infinity ? "∞" : d).toString().padEnd(7)}  ${cp.m.toFixed(4)}   ${(100 * cp.empty / cp.tot).toFixed(0).padStart(3)}%  ${cn.m.toFixed(4)}   ${(100 * cn.empty / cn.tot).toFixed(0).padStart(3)}%   ${(cp.m / (cn.m || 1e-9)).toFixed(2)}\n`); }

  // masked perfect-coverage detector at δ=0.001 + FP samples (which RARE WORDS matched)
  const DD = 0.001; const { p: pDet } = buildP(POS, DD);
  let mF = 0, mTP = 0, mU = 0, mN = 0; const fp = [];
  for (const words of sents) for (let t = 1; t < words.length; t++) {
    const cand = feat(words, t); mN++; const y = rareIds(cand, DD);
    if (y.length === 0) { mU++; continue; }
    let allIn = true; for (const id of y) if (!pDet[id]) { allIn = false; break; }
    if (allIn) { mF++; if (words[t] === TARGET) mTP++; else if (fp.length < 12 && t % 7 === 0) { const rw = cand.filter((e) => e.wf <= DD).map((e) => e.w); const ctx = words.slice(Math.max(0, t - D), t).join(" "); fp.push({ ctx, tgt: words[t], rw }); } }
  }
  w.write(`\n── MASKED perfect-coverage detector (δ=${DD} on word freq) N=${mN} ──\n`);
  w.write(`  fired=${mF} (${(100 * mF / mN).toFixed(3)}%)  TP=${mTP}/${POS.length} (recall ${(100 * mTP / POS.length).toFixed(1)}%)  FALSE POSITIVES=${mF - mTP}  precision=${(100 * mTP / mF).toFixed(3)}%  undefined=${(100 * mU / mN).toFixed(1)}%\n`);
  for (const e of fp) w.write(`  {${e.ctx}} → [${e.tgt}]  (rare words: ${e.rw.join(" ")})\n`);
};

main();
