"use strict";

// ============================================================================
// benchmark/megaPart.js — can ONE per-word union-part separate its target?
// E1: p = ⋁_{i∈A} y_i (raw union).  E2: p = union \ {a : freq_Ā(a) > δ}, sweep δ.
// NO ranking, NO top-n, NO cap on |p| — E2 keeps the WEAK MANY (mortgage n_A=1 stays);
// only the ubiquitous atoms are dropped. Deliverable is the popcount histogram
// (A vs Ā), overlap fraction, best-t F1, corr(k,|y|) [length control], and the MASK.
// Usage: node --max-old-space-size=8192 benchmark/megaPart.js <dict> <corpus>
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { fuzzyEncodeWord } from "../src/core/signatures/wordEncoder.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";
import { kindToString } from "../src/core/parts/kind.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };
const DEL = [Infinity, 0.5, 0.2, 0.1, 0.05, 0.01];   // δ[0]=E1 (raw union)
const NAMIN = Number(process.env.NAMIN || 1);   // E3: keep only atoms with n_A ≥ NAMIN (recurrence filter — strips the n_A=1 length-noise)
const HB = 384;   // histogram cap

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const parts = dict.allParts(), F = dict.size();
  const cache = new Map();
  const atomsOf = (w) => { let a = cache.get(w); if (!a) { a = Int32Array.from(new Set(fuzzyEncodeWord(dict, w))); cache.set(w, a); } return a; };
  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const toks = tokenizeStream(line.slice(seg.start, seg.end));
    const ws = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws.map((x) => ({ w: x, atoms: atomsOf(x) })));
  }
  const targets = ["bank", "state", "city", "the"];
  const tIdx = new Map(targets.map((t, i) => [t, i]));
  const w = process.stdout;
  const stamp = new Int32Array(F).fill(-1);

  // window word-index range for (enc, radius) at position p
  const win = (enc, r, p, L) => enc === "L|C" ? [r === Infinity ? 0 : Math.max(0, p - 1 - r), p - 1, -1] : [r === Infinity ? 0 : Math.max(0, p - r), r === Infinity ? L - 1 : Math.min(L - 1, p + r), p];

  for (const enc of ["L|C", "L,R"]) for (const r of [5, Infinity]) {
    // ---- PASS 1: M_a, nA[t], |A[t]|, N ----
    const M = new Float64Array(F), nA = targets.map(() => new Float64Array(F)), absA = new Float64Array(targets.length); let N = 0, pid = 0;
    for (const s of sents) { const L = s.length; for (let p = 0; p < L; p++) { const [lo, hi, sk] = win(enc, r, p, L); if (hi < lo) continue; pid++; let any = false; const ti = tIdx.get(s[p].w); for (let q = lo; q <= hi; q++) { if (q === sk) continue; for (const a of s[q].atoms) if (stamp[a] !== pid) { stamp[a] = pid; M[a]++; if (ti !== undefined) nA[ti][a]++; any = true; } } if (any) { N++; if (ti !== undefined) absA[ti]++; } } }
    // freq_Ā[t][a] = (M-nA)/(N-|A|)
    const fB = targets.map((_, ti) => { const f = new Float32Array(F), den = N - absA[ti]; for (let a = 0; a < F; a++) if (nA[ti][a] > 0) f[a] = (M[a] - nA[ti][a]) / den; return f; });

    // ---- PASS 2: popcount histograms for every (target, δ) ----
    const T = targets.length, D = DEL.length;
    const hA = new Float64Array(T * D * HB), hB = new Float64Array(T * D * HB);
    const cn = new Float64Array(T * D), sk = new Float64Array(T * D), sy = new Float64Array(T * D), sky = new Float64Array(T * D), skk = new Float64Array(T * D), syy = new Float64Array(T * D);
    for (const s of sents) {
      const L = s.length;
      for (let p = 0; p < L; p++) {
        const [lo, hi, skp] = win(enc, r, p, L); if (hi < lo) continue;
        // build y_i (dedup atoms) and its length
        pid++; const ylist = []; for (let q = lo; q <= hi; q++) { if (q === skp) continue; for (const a of s[q].atoms) if (stamp[a] !== pid) { stamp[a] = pid; ylist.push(a); } }
        const yLen = ylist.length; if (!yLen) continue;
        const selfTi = tIdx.get(s[p].w);
        for (let ti = 0; ti < T; ti++) {
          const nAt = nA[ti], fBt = fB[ti];
          // kByDelta: count y-atoms in union[ti] with freq_Ā ≤ δ_j, for each δ_j
          const kd = new Int32Array(D);
          for (const a of ylist) { if (nAt[a] < NAMIN) continue; const f = fBt[a]; for (let j = 0; j < D; j++) if (f <= DEL[j]) kd[j]++; }
          const isA = selfTi === ti;
          for (let j = 0; j < D; j++) { const k = kd[j], idx = ti * D + j; if (isA) hA[idx * HB + Math.min(k, HB - 1)]++; else { hB[idx * HB + Math.min(k, HB - 1)]++; cn[idx]++; sk[idx] += k; sy[idx] += yLen; sky[idx] += k * yLen; skk[idx] += k * k; syy[idx] += yLen * yLen; } }
        }
      }
    }

    // ---- report ----
    const stat = (h, off) => { let n = 0, s1 = 0, s2 = 0; for (let k = 0; k < HB; k++) { const c = h[off + k]; n += c; s1 += c * k; s2 += c * k * k; } const m = n ? s1 / n : 0; return { n, m, sd: n ? Math.sqrt(Math.max(0, s2 / n - m * m)) : 0 }; };
    const overlapAndF1 = (ti, j) => {
      const oA = (ti * D + j) * HB, oB = (ti * D + j) * HB, sA = stat(hA, oA), sB = stat(hB, oB);
      let ov = 0; for (let k = 0; k < HB; k++) ov += Math.min(sA.n ? hA[oA + k] / sA.n : 0, sB.n ? hB[oB + k] / sB.n : 0);
      // best-t F1 (predict A iff k>=t)
      let bF = 0, bt = 0, bp = 0, br = 0; let cumA = sA.n, cumB = sB.n;
      for (let t = 0; t <= HB; t++) { const tp = cumA, fp = cumB; const P = tp + fp ? tp / (tp + fp) : 0, R = sA.n ? tp / sA.n : 0, f1 = P + R ? 2 * P * R / (P + R) : 0; if (f1 > bF) { bF = f1; bt = t; bp = P; br = R; } if (t < HB) { cumA -= hA[oA + t]; cumB -= hB[oB + t]; } }
      return { sA, sB, ov, bF, bt, bp, br };
    };
    w.write(`\n════════ [${enc}]  r=${r === Infinity ? "∞" : r} ════════   (N=${N})\n`);
    for (let ti = 0; ti < T; ti++) {
      w.write(`  ${targets[ti]}  |A|=${absA[ti]}\n`);
      w.write(`    δ        |p|      meanA   meanĀ   overlap   F1(t,P,R)          corr(k,|y|)Ā\n`);
      let bestJ = 0, bestF = -1;
      for (let j = 0; j < D; j++) {
        let psz = 0; for (let a = 0; a < F; a++) if (nA[ti][a] >= NAMIN && fB[ti][a] <= DEL[j]) psz++;
        const R = overlapAndF1(ti, j), idx = ti * D + j;
        const n = cn[idx], cr = n > 1 ? (n * sky[idx] - sk[idx] * sy[idx]) / (Math.sqrt(Math.max(1e-9, n * skk[idx] - sk[idx] * sk[idx])) * Math.sqrt(Math.max(1e-9, n * syy[idx] - sy[idx] * sy[idx]))) : 0;
        if (R.bF > bestF) { bestF = R.bF; bestJ = j; }
        w.write(`    ${(DEL[j] === Infinity ? "E1/∞" : DEL[j]).toString().padEnd(6)} ${String(psz).padStart(7)}   ${R.sA.m.toFixed(1).padStart(6)}  ${R.sB.m.toFixed(1).padStart(6)}   ${R.ov.toFixed(3)}    ${R.bF.toFixed(2)}(t${R.bt},P${R.bp.toFixed(2)},R${R.br.toFixed(2)})   ${cr.toFixed(2)}\n`);
      }
      // mask dump at best δ
      const δ = DEL[bestJ]; const mask = []; for (let a = 0; a < F; a++) if (nA[ti][a] >= NAMIN && fB[ti][a] <= δ) mask.push(a);
      const kc = { whole: 0, start: 0, end: 0, mid: 0 }; let n1 = 0; for (const a of mask) { kc[kindToString(parts[a].kind)] = (kc[kindToString(parts[a].kind)] || 0) + 1; if (nA[ti][a] === 1) n1++; }
      mask.sort((x, y2) => nA[ti][y2] - nA[ti][x]);
      const show = mask.filter((a) => /mortgag|teller|deposit|lend|swi|ubs|orporat|danub|river|oxidat|michig|egislat|manches|money|credit|financ|investm/.test(parts[a].value)).slice(0, 14);
      w.write(`    → best δ=${δ === Infinity ? "E1" : δ}: |p|=${mask.length}  Kind{W:${kc.whole} S:${kc.start} E:${kc.end} M:${kc.mid}}  n_A=1 atoms: ${n1} (${(100 * n1 / mask.length).toFixed(0)}%)\n`);
      w.write(`      sense-atoms in mask: ${show.length ? show.map((a) => `${parts[a].value}:${kindToString(parts[a].kind)}(${nA[ti][a]})`).join(" ") : "—"}\n`);
    }
  }
};

main();
