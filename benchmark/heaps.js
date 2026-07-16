"use strict";

// benchmark/heaps.js — §9.2 Heaps sweep. Subsample the target's positives at fixed fractions,
// run 6a (seq, emergent K) at a FIXED λ, and fit K ≈ c·|A|^β log-log. β>0 with good r² ⇒
// finding parts (part count grows with contexts); flat K ⇒ collapsing to senses (§9.2/§11).
// Held recall is measured on a FIXED held set so the curves are comparable across |A|.
// Usage: TARGET=bank R=5 REP=fuzzy LAMBDA=0.002 node --max-old-space-size=8192 benchmark/heaps.js english.txt <corpus>

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { encodeWord } from "../src/core/signatures/wordEncoder.js";
import { fuzzyEncode } from "../src/core/parts/encoding/fuzzyEncode.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";
import buildHardNegatives from "../src/core/discovery/hardNegatives.js";
import discoverSeq from "../src/core/discovery/discoverSeq.js";
import makeReadout from "../src/core/discovery/maxReadout.js";
import poolMetrics from "../src/core/discovery/metrics.js";
import { heapsFit, subsamplePrefix } from "../src/core/discovery/heaps.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGET = process.env.TARGET || "bank";
const R = Number(process.env.R || 5), D = R + 1, DD = Number(process.env.DELTA || 0.001);
const REP = process.env.REP || "fuzzy";
const LAMBDA = Number(process.env.LAMBDA || 0.002);       // FIXED across all fractions (mandatory, §9.2)
const KAPPA = Number(process.env.KAPPA || 1.5);
const MINHARD = Number(process.env.MINHARD || 2000);
const NEGSAMPLE = Number(process.env.NEGSAMPLE || 20000);

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size(), DIM = 3 * F;
  const enc = REP === "sparse" ? encodeWord : fuzzyEncode;
  const pc = new Map(); const partsOf = (w) => { let a = pc.get(w); if (!a) { a = [...new Set(enc(dict, w))]; pc.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const ws = tokenizeStream(line.slice(seg.start, seg.end)).filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws);
  }
  const band = (dd) => (dd === 1 ? 2 : dd <= 3 ? 1 : 0);
  const stmp = new Int32Array(DIM).fill(-1); let sp = 0;
  const feat = (s, t) => { sp++; const y = []; for (let dd = 1; dd <= D; dd++) { const q = t - dd; if (q < 0) break; const off = band(dd) * F; for (const a of partsOf(s[q])) { const b = a + off; if (stmp[b] !== sp) { stmp[b] = sp; y.push(b); } } } return y.sort((a, b) => a - b); };

  const M = new Float64Array(DIM); let Ntot = 0, negTot = 0;
  const A = [], negS = []; let ni = 0;
  for (const s of sents) for (let t = 1; t < s.length; t++) {
    const f = feat(s, t); Ntot++; for (const b of f) M[b]++;
    if (s[t] === TARGET) A.push(f);
    else { negTot++; if (ni++ % Math.max(1, Math.floor((sents.length * 8) / NEGSAMPLE)) === 0) negS.push(f); }
  }
  const Mglob = new Set(); for (let b = 0; b < DIM; b++) if (M[b] / Ntot > DD) Mglob.add(b);

  // fixed held set = last 30% of A; subsample the pool (first 70%) at fractions.
  const hcut = Math.floor(A.length * 0.7);
  let pool = A.slice(0, hcut); const Aheld = A.slice(hcut);
  const poolBase = pool;
  const DRAWS = Number(process.env.DRAWS || 1);                  // random draws per fraction; K averaged
  const FIXEDAH = Number(process.env.FIXEDAH || 0);             // exact |Ā_h| to hold the pool constant
  const SEED0 = Number(process.env.SEED || 12345);
  const broadSample = negS.filter((_, j) => j % Math.max(1, Math.floor(negS.length / 6000)) === 0);
  // thrNeg: broad FP surface (curbs ensemble FP at high K); THRHARD adds Ā_h so t_k
  // controls BOTH the easy and the confusable surface.
  const mkThrNeg = (AbarH) => (Number(process.env.THRNEG || 0) ? (Number(process.env.THRHARD || 0) ? [...AbarH, ...broadSample] : broadSample) : null);
  const shuffled = (seed) => { const p = poolBase.slice(); let s = seed >>> 0; const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 0x100000000; for (let i = p.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; } return p; };
  const drawPool = (d) => (DRAWS > 1 || Number(process.env.SHUFFLE || 0) ? shuffled(SEED0 + d * 7919) : poolBase);

  const runFraction = (poolDraw, fr) => {
    const Af = subsamplePrefix(poolDraw, fr);
    const { Dtau, AbarH, t0 } = buildHardNegatives(Af, negS, Mglob, FIXEDAH > 0 ? { size: FIXEDAH } : { minSize: MINHARD });
    const { gates } = discoverSeq(Af, AbarH, Dtau, { lambda: LAMBDA, m: 2, kappa: KAPPA, sMin: 5, maxK: 400, minCover: Number(process.env.MINCOVER || 2), thrNeg: mkThrNeg(AbarH) });
    const mp = poolMetrics((x) => makeReadout(gates, Dtau).fires(x), Aheld, negS);
    const meanPm = gates.length ? gates.reduce((a, g) => a + g.pMinus.length, 0) / gates.length : 0;
    return { n: Af.length, K: gates.length, recall: mp.recall, fp: mp.fpRate, AbarHlen: AbarH.length, t0, meanPm, cov: gates.reduce((a, g) => a + g.Q.length, 0) };
  };

  const fracs = (process.env.FRACS || "0.15,0.3,0.5,0.7,1.0").split(",").map(Number);
  const w = process.stdout;
  w.write(`# heaps — ${path.basename(corpusArg)} | TARGET=${TARGET} REP=${REP} λ=${LAMBDA} κ=${KAPPA} DRAWS=${DRAWS} | |A|=${A.length} pool=${poolBase.length} held=${Aheld.length} negS=${negS.length}\n\n`);
  w.write(`  |A_f|   |Ā_h|      K±sd     mean|p⁻|  held-rec  held-FP%   ΣQ\n`);

  const mean = (rs, f) => rs.reduce((a, r) => a + f(r), 0) / rs.length;
  const points = [];
  for (const fr of fracs) {
    const rs = []; for (let d = 0; d < DRAWS; d++) rs.push(runFraction(drawPool(d), fr));
    const mK = mean(rs, (r) => r.K), sK = Math.sqrt(mean(rs, (r) => (r.K - mK) ** 2));
    const mN = Math.round(mean(rs, (r) => r.n));
    points.push({ n: mN, K: mK });
    w.write(`  ${String(mN).padStart(5)}  ${String(Math.round(mean(rs, (r) => r.AbarHlen))).padStart(5)}  ${mK.toFixed(1).padStart(5)}±${sK.toFixed(1).padStart(4)}   ${mean(rs, (r) => r.meanPm).toFixed(1).padStart(6)}   ${(100 * mean(rs, (r) => r.recall)).toFixed(1).padStart(5)}%   ${(100 * mean(rs, (r) => r.fp)).toFixed(3).padStart(6)}%   ${String(Math.round(mean(rs, (r) => r.cov))).padStart(4)}\n`);
  }

  const { beta, c, r2, n } = heapsFit(points);
  w.write(`\n# Heaps fit (K ≈ c·|A|^β) on ${DRAWS > 1 ? `${DRAWS}-draw means` : "single draw"}:  β=${beta.toFixed(3)}  c=${c.toFixed(2)}  r²=${r2.toFixed(3)}  (n=${n} points)\n`);
  w.write(beta > 0.15 && r2 > 0.7 ? `# → β>0 with good r²: part count GROWS with |A| — finding parts (Heaps), not senses.\n`
    : Number.isNaN(beta) ? `# → too few non-empty points to fit (K stayed ~0).\n`
    : `# → flat/weak β: suspect collapse to a fixed sense count, OR saturation of a finite inventory — inspect the curve.\n`);
};

main();
