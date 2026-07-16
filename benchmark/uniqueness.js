"use strict";

// ============================================================================
// benchmark/uniqueness.js — plain bit-overlap uniqueness test for one TARGET word.
// POS = context signatures y_i at positions where word==TARGET.  NEG = strided sample of the rest.
// Overlap(A,B) = popcount(A ∧ B) = |shared atoms|.  Report mean±std for pos-pos / pos-neg / neg-neg,
// raw and Jaccard. Then treat p = ⋁ POS (the OR-ed part) as one signature and report |p∧pos|, |p∧neg|.
// Usage: TARGET=bank ENC='L|C' R=5 node --max-old-space-size=8192 benchmark/uniqueness.js <dict> <corpus>
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { fuzzyEncodeWord, encodeWord } from "../src/core/signatures/wordEncoder.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGET = process.env.TARGET || "bank";
const ENC = process.env.ENC || "L|C";
const R = process.env.R === "inf" ? Infinity : Number(process.env.R || 5);
const NEG_N = Number(process.env.NEG || 3000);   // number of sampled negatives

const win = (enc, r, p, L) => enc === "L|C" ? [r === Infinity ? 0 : Math.max(0, p - 1 - r), p - 1, -1] : [r === Infinity ? 0 : Math.max(0, p - r), r === Infinity ? L - 1 : Math.min(L - 1, p + r), p];

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size();
  const ENCODER = process.env.ENCODER === "sparse" ? encodeWord : fuzzyEncodeWord;
  const cache = new Map();
  const atomsOf = (w) => { let a = cache.get(w); if (!a) { a = Int32Array.from(new Set(ENCODER(dict, w))); cache.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const toks = tokenizeStream(line.slice(seg.start, seg.end));
    const ws = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws.map((x) => ({ w: x, atoms: atomsOf(x) })));
  }

  // build signature (dedup atom list) for a window
  const stamp = new Int32Array(F).fill(-1); let pid = 0;
  const sig = (s, lo, hi, sk) => { pid++; const y = []; for (let q = lo; q <= hi; q++) { if (q === sk) continue; for (const a of s[q].atoms) if (stamp[a] !== pid) { stamp[a] = pid; y.push(a); } } return Int32Array.from(y); };

  // ---- collect POS and total negatives count for stride ----
  let negTotal = 0;
  for (const s of sents) { const L = s.length; for (let p = 0; p < L; p++) { const [lo, hi] = win(ENC, R, p, L); if (hi < lo) continue; if (s[p].w !== TARGET) negTotal++; } }
  const step = Math.max(1, Math.floor(negTotal / NEG_N));

  const POS = [], NEG = [], POSW = [], NEGW = []; let negIdx = 0;
  const words = (s, lo, hi, sk, cw) => { const ctx = []; for (let q = lo; q <= hi; q++) { if (q === sk) continue; ctx.push(s[q].w); } return `[${cw}] ⟵ {${ctx.join(" ")}}`; };
  for (const s of sents) { const L = s.length; for (let p = 0; p < L; p++) { const [lo, hi, sk] = win(ENC, R, p, L); if (hi < lo) continue; if (s[p].w === TARGET) { POS.push(sig(s, lo, hi, sk)); if (POSW.length < 6) POSW.push(words(s, lo, hi, sk, s[p].w)); } else { if (negIdx % step === 0) { NEG.push(sig(s, lo, hi, sk)); if (NEGW.length < 6) NEGW.push(words(s, lo, hi, sk, s[p].w)); } negIdx++; } } }

  // OR-ed part p
  const inP = new Uint8Array(F); let psz = 0;
  for (const y of POS) for (const a of y) if (!inP[a]) { inP[a] = 1; psz++; }

  // overlap via membership stamp
  const overlap = (A, B) => { pid++; for (const a of A) stamp[a] = pid; let c = 0; for (const b of B) if (stamp[b] === pid) c++; return c; };
  const stats = () => ({ n: 0, s: 0, s2: 0, js: 0, js2: 0 });
  const add = (st, ov, la, lb) => { st.n++; st.s += ov; st.s2 += ov * ov; const j = ov / (la + lb - ov || 1); st.js += j; st.js2 += j * j; };
  const rep = (st) => { const m = st.s / st.n, sd = Math.sqrt(Math.max(0, st.s2 / st.n - m * m)); const jm = st.js / st.n, jsd = Math.sqrt(Math.max(0, st.js2 / st.n - jm * jm)); return { m, sd, jm, jsd }; };
  const meanLen = (arr) => arr.reduce((a, y) => a + y.length, 0) / arr.length;

  const PP = stats(), PN = stats(), NN = stats();
  for (let i = 0; i < POS.length; i++) for (let j = 0; j < POS.length; j++) if (i !== j) { const ov = overlap(POS[i], POS[j]); add(PP, ov, POS[i].length, POS[j].length); }
  for (let i = 0; i < POS.length; i++) for (let j = 0; j < NEG.length; j++) { const ov = overlap(POS[i], NEG[j]); add(PN, ov, POS[i].length, NEG[j].length); }
  const nnCap = Math.min(NEG.length, 2000);
  for (let i = 0; i < nnCap; i++) for (let j = 0; j < nnCap; j++) if (i !== j) { const ov = overlap(NEG[i], NEG[j]); add(NN, ov, NEG[i].length, NEG[j].length); }

  // OR-ed part vs pos / neg — coverage = |p∩y| / |y|  (normalize by the COMPARED sample, not the union)
  const cov = () => ({ n: 0, s: 0, s2: 0, rs: 0, rs2: 0 });
  const addc = (st, ov, ly) => { st.n++; const c = ov / (ly || 1); st.s += c; st.s2 += c * c; st.rs += ov; st.rs2 += ov * ov; };
  const repc = (st) => { const m = st.s / st.n, sd = Math.sqrt(Math.max(0, st.s2 / st.n - m * m)); const rm = st.rs / st.n, rsd = Math.sqrt(Math.max(0, st.rs2 / st.n - rm * rm)); return { m, sd, rm, rsd }; };
  const pPos = cov(), pNeg = cov();
  for (const y of POS) { let ov = 0; for (const a of y) if (inP[a]) ov++; addc(pPos, ov, y.length); }
  for (const y of NEG) { let ov = 0; for (const a of y) if (inP[a]) ov++; addc(pNeg, ov, y.length); }

  const w = process.stdout;
  w.write(`# uniqueness — ${path.basename(corpusArg)} | TARGET=${TARGET} ENC=[${ENC}] r=${R === Infinity ? "∞" : R}\n`);
  w.write(`# POS n=${POS.length} mean|y|=${meanLen(POS).toFixed(1)}   NEG n=${NEG.length} (stride ${step}) mean|y|=${meanLen(NEG).toFixed(1)}   |p|=${psz}\n`);
  w.write(`# sample POSITIVE contexts (real signatures):\n${POSW.map((x) => "#   " + x).join("\n")}\n`);
  w.write(`# sample NEGATIVE contexts (real signatures, strided sample):\n${NEGW.map((x) => "#   " + x).join("\n")}\n\n`);
  const line = (name, st) => { const r = rep(st); w.write(`  ${name.padEnd(10)} raw |A∩B| = ${r.m.toFixed(2).padStart(6)} ± ${r.sd.toFixed(2).padStart(6)}     Jaccard = ${r.jm.toFixed(4)} ± ${r.jsd.toFixed(4)}\n`); };
  w.write(`── pairwise signature overlap ──\n`);
  line("pos–pos", PP);
  line("pos–neg", PN);
  line("neg–neg", NN);
  w.write(`\n── OR-ed part p as a single positive (|p|=${psz}) — coverage = |p∩y| / |y| ──\n`);
  const rp = repc(pPos), rn = repc(pNeg);
  w.write(`  p vs pos   coverage = ${rp.m.toFixed(4)} ± ${rp.sd.toFixed(4)}     raw |p∩y| = ${rp.rm.toFixed(2)} ± ${rp.rsd.toFixed(2)}   (pos⊆p ⇒ coverage=1)\n`);
  w.write(`  p vs neg   coverage = ${rn.m.toFixed(4)} ± ${rn.sd.toFixed(4)}     raw |p∩y| = ${rn.rm.toFixed(2)} ± ${rn.rsd.toFixed(2)}\n`);
  w.write(`\n  → coverage ratio pos/neg = ${(rp.m / rn.m).toFixed(3)}   (a random context is ${(100 * rn.m).toFixed(1)}% inside p vs 100% for positives)\n`);

  // ---- fire coverage ≥ THRESH over the WHOLE corpus ----
  const THRESH = Number(process.env.THRESH || rn.m);
  let N = 0, fired = 0, tp = 0, firedBits = 0, fpBits = 0;
  const orNeg = new Uint8Array(F); let orNegBits = 0;   // OR of the FALSE-POSITIVE (center≠TARGET) fired windows
  for (const s of sents) { const L = s.length; for (let p = 0; p < L; p++) { const [lo, hi, sk] = win(ENC, R, p, L); if (hi < lo) continue; N++; pid++; let k = 0, yl = 0; const ys = []; for (let q = lo; q <= hi; q++) { if (q === sk) continue; for (const a of s[q].atoms) if (stamp[a] !== pid) { stamp[a] = pid; yl++; ys.push(a); if (inP[a]) k++; } } if (yl > 0 && k / yl >= THRESH) { fired++; firedBits += yl; if (s[p].w === TARGET) tp++; else { fpBits += yl; for (const a of ys) if (!orNeg[a]) { orNeg[a] = 1; orNegBits++; } } } } }
  const fp = fired - tp;
  w.write(`\n── DETECTOR: coverage ≥ ${THRESH.toFixed(4)} over whole corpus (N=${N}) ──\n`);
  w.write(`  fired=${fired} (${(100 * fired / N).toFixed(1)}% of corpus)   TP=${tp}/${POS.length} (recall ${(100 * tp / POS.length).toFixed(1)}%)   FALSE POSITIVES=${fp}   precision=${(100 * tp / fired).toFixed(4)}%\n`);
  w.write(`  mean bits of FIRED windows = ${(firedBits / fired).toFixed(2)}   mean bits of FIRED false-positives = ${(fpBits / fp).toFixed(2)}   (vs corpus mean|y|=${meanLen(NEG).toFixed(1)}, positive mean=${meanLen(POS).toFixed(1)})\n`);
  let inBoth = 0; for (let a = 0; a < F; a++) if (orNeg[a] && inP[a]) inBoth++;
  w.write(`  OR of the ${fp} false-positive negatives = ${orNegBits} bits  (= ${(100 * orNegBits / psz).toFixed(1)}% of p's ${psz} bits; ${inBoth}/${psz} of p's atoms are reconstructed by non-${TARGET} contexts)\n`);
};

main();
