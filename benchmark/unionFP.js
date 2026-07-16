"use strict";

// ============================================================================
// benchmark/unionFP.js — DIAGNOSTIC: what does a target's union-part actually detect?
// Build p = ⋁_{i∈A} y_i (raw union of TARGET's context signatures). Set the threshold to the
// minimal viable bar t_p = min_{i∈A} popcount(y_i ∧ p) — and since every y_i ⊆ p, that is just
// min_{i∈A} |y_i| (the shortest context). Recall on A is 100% by construction. Then fire p over
// the WHOLE corpus. A FALSE POSITIVE = a window that clears t_p whose CENTER word ≠ TARGET.
//   Report FP center words ranked by COUNT and by LIFT (fire-rate enrichment vs base rate),
//   plus the fraction of FP mass landing in TARGET's semantic field.
//   Random FPs (the/and/was) ⇒ length detector. Field-clustered FPs (credit/loan/river) ⇒ field detector.
// Usage: TARGET=bank ENC='L|C' R=5 node --max-old-space-size=8192 benchmark/unionFP.js <dict> <corpus>
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadDictText, loadDictBinary } from "../src/core/parts/dictionary.js";
import { tokenizeStream, StreamTokenType, asciiLowercase } from "../src/core/parts/tokenize.js";
import { fuzzyEncodeWord } from "../src/core/signatures/wordEncoder.js";
import { segmentText } from "../src/utilities/textSegmentation/segmentText.js";

const REPO = path.resolve(fileURLToPath(import.meta.url), "..", "..");
const DICT_DIR = path.join(REPO, "data", "dictionaries");
const resolveIn = (p, d) => (path.dirname(p) === "." && !path.isAbsolute(p) ? path.join(d, p) : p);
const loadDict = (a) => { const p = resolveIn(a, DICT_DIR); return p.endsWith(".bin") ? loadDictBinary(new Uint8Array(fs.readFileSync(p))) : loadDictText(fs.readFileSync(p, "latin1")); };

const TARGET = process.env.TARGET || "bank";
const ENC = process.env.ENC || "L|C";
const R = process.env.R === "inf" ? Infinity : Number(process.env.R || 5);
const MINC = Number(process.env.MINC || 30);   // min FP count to appear in the lift-ranked list (kills hapax lift noise)

// TARGET-specific semantic field word lists (for the reverse view). Only bank is curated; others → skip field view.
const FIELDS = {
  bank: {
    fin: "banks banking bank's credit loan loans lend lender lending borrow financial finance deposit deposits depositor mortgage money investment investor account accounts monetary currency capital fund funds cash savings interest debt bond bonds securities teller atm branch".split(" "),
    geo: "river rivers riverbank bank's shore shores water waters stream streams lake creek valley bay coast estuary flood embankment".split(" "),
  },
  state: { gov: "government federal national province provincial county municipal republic nation kingdom territory jurisdiction constitution legislature governor senate parliament sovereign".split(" "), phys: "solid liquid gas plasma matter phase energy equilibrium quantum".split(" ") },
  city: { urban: "town village municipality metropolis capital urban downtown suburb borough district county population mayor council".split(" ") },
};

const win = (enc, r, p, L) => enc === "L|C"
  ? [r === Infinity ? 0 : Math.max(0, p - 1 - r), p - 1, -1]
  : [r === Infinity ? 0 : Math.max(0, p - r), r === Infinity ? L - 1 : Math.min(L - 1, p + r), p];

const main = () => {
  const [dictArg, corpusArg] = process.argv.slice(2);
  const dict = loadDict(dictArg); const F = dict.size();
  const cache = new Map();
  const atomsOf = (w) => { let a = cache.get(w); if (!a) { a = Int32Array.from(new Set(fuzzyEncodeWord(dict, w))); cache.set(w, a); } return a; };

  const sents = [];
  for (const line of fs.readFileSync(corpusArg, "utf8").split("\n")) for (const seg of segmentText(line)) {
    const toks = tokenizeStream(line.slice(seg.start, seg.end));
    const ws = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => asciiLowercase(t.value));
    if (ws.length >= 2) sents.push(ws.map((x) => ({ w: x, atoms: atomsOf(x) })));
  }

  const inP = new Uint8Array(F);
  const stamp = new Int32Array(F).fill(-1); let pid = 0;

  // ---- PASS 1a: build p = union of TARGET's context atoms ----
  let absA = 0, psz = 0;
  for (const s of sents) { const L = s.length; for (let p = 0; p < L; p++) { if (s[p].w !== TARGET) continue; const [lo, hi, sk] = win(ENC, R, p, L); if (hi < lo) continue; absA++; for (let q = lo; q <= hi; q++) { if (q === sk) continue; for (const a of s[q].atoms) if (!inP[a]) { inP[a] = 1; psz++; } } } }

  // ---- PASS 1b: t_p = min over TARGET windows of popcount(y ∧ p)  (== min |y_i| since y_i ⊆ p) ----
  let tP = Infinity;
  for (const s of sents) { const L = s.length; for (let p = 0; p < L; p++) { if (s[p].w !== TARGET) continue; const [lo, hi, sk] = win(ENC, R, p, L); if (hi < lo) continue; pid++; let k = 0; for (let q = lo; q <= hi; q++) { if (q === sk) continue; for (const a of s[q].atoms) if (inP[a] && stamp[a] !== pid) { stamp[a] = pid; k++; } } if (k > 0 && k < tP) tP = k; } }
  if (tP === Infinity) tP = 1;

  // ---- PASS 2: fire p over the whole corpus, tally center words + per-group popcount histograms ----
  const KMAX = 256;
  const field = FIELDS[TARGET];
  const finSet = new Set(field ? (field.fin || field.gov || field.urban || []) : []);
  const geoSet = new Set(field ? (field.geo || field.phys || []) : []);
  const CTL_FN = new Set("the of and to in a was is that as on for with by".split(" "));
  const CTL_CONTENT = new Set("film music army game season league team party church century election player album song championship battle soldier military war".split(" "));
  const hist = { ALL: new Float64Array(KMAX), TARGET: new Float64Array(KMAX), fin: new Float64Array(KMAX), geo: new Float64Array(KMAX), ctlFn: new Float64Array(KMAX), ctlCt: new Float64Array(KMAX) };
  let N = 0, fired = 0, tpFired = 0;
  const occ = new Map();       // center word -> valid positions
  const fpFired = new Map();   // center word (≠TARGET) -> fired count at t_p
  for (const s of sents) {
    const L = s.length;
    for (let p = 0; p < L; p++) {
      const [lo, hi, sk] = win(ENC, R, p, L); if (hi < lo) continue;
      N++; const cw = s[p].w; occ.set(cw, (occ.get(cw) || 0) + 1);
      pid++; let k = 0;
      for (let q = lo; q <= hi; q++) { if (q === sk) continue; for (const a of s[q].atoms) if (inP[a] && stamp[a] !== pid) { stamp[a] = pid; k++; } }
      const kb = k < KMAX ? k : KMAX - 1;
      hist.ALL[kb]++;
      if (cw === TARGET) hist.TARGET[kb]++;
      else if (finSet.has(cw)) hist.fin[kb]++;
      else if (geoSet.has(cw)) hist.geo[kb]++;
      else if (CTL_FN.has(cw)) hist.ctlFn[kb]++;
      else if (CTL_CONTENT.has(cw)) hist.ctlCt[kb]++;
      if (k >= tP) { fired++; if (cw === TARGET) tpFired++; else fpFired.set(cw, (fpFired.get(cw) || 0) + 1); }
    }
  }
  // suffix sums: fireCount(group, t) = Σ_{k≥t} hist[group][k]
  const suffix = (h) => { const s = new Float64Array(KMAX + 1); for (let k = KMAX - 1; k >= 0; k--) s[k] = s[k + 1] + h[k]; return s; };
  const suf = Object.fromEntries(Object.entries(hist).map(([g, h]) => [g, suffix(h)]));
  const tot = Object.fromEntries(Object.entries(hist).map(([g, h]) => [g, h.reduce((a, b) => a + b, 0)]));

  const w = process.stdout;
  const fr = fired / N;                       // overall fire rate
  const liftOf = (cnt, o) => (o > 0 ? (cnt / o) / fr : 0);
  w.write(`# unionFP — ${path.basename(corpusArg)} | TARGET=${TARGET}  ENC=[${ENC}]  r=${R === Infinity ? "∞" : R}\n`);
  w.write(`# |A|=${absA}  |p|=${psz} atoms  t_p=${tP} (shortest ctx)  N=${N}\n`);
  w.write(`# fired=${fired} (${(100 * fr).toFixed(2)}% of corpus)  recall=${(100 * tpFired / absA).toFixed(1)}%  precision=${(100 * tpFired / fired).toFixed(3)}%  → ${fired - tpFired} false positives\n\n`);

  // ---- THRESHOLD SWEEP: does field enrichment EMERGE as the bar rises above the saturated min? ----
  w.write(`── THRESHOLD SWEEP: fire-rate + group LIFT vs threshold t (lift=1 ⇒ tracks length; lift≫1 ⇒ field detector) ──\n`);
  w.write(`    t      overallFire   recall(A)   fin-lift   geo-lift   ctlFn-lift   ctlContent-lift\n`);
  const rate = (g, t) => (tot[g] ? suf[g][Math.min(t, KMAX)] / tot[g] : 0);
  const THS = [tP, 5, 8, 12, 16, 20, 25, 30, 40, 50, 65, 80, 100, 130];
  for (const t of THS) {
    const oR = rate("ALL", t); if (oR === 0) break;
    const lg = (g) => (oR > 0 ? (rate(g, t) / oR).toFixed(2) : "—");
    w.write(`    ${String(t).padStart(4)}   ${(100 * oR).toFixed(2).padStart(8)}%   ${(100 * rate("TARGET", t)).toFixed(1).padStart(7)}%   ${lg("fin").padStart(7)}   ${lg("geo").padStart(7)}   ${lg("ctlFn").padStart(8)}   ${lg("ctlCt").padStart(12)}\n`);
  }
  w.write(`    (fin/geo = bank's finance/river field words; ctlFn = function words; ctlContent = film/army/game/battle… off-field content)\n\n`);

  const fpArr = [...fpFired.entries()].map(([word, cnt]) => ({ word, cnt, occ: occ.get(word) || 0, fr: cnt / (occ.get(word) || 1), lift: liftOf(cnt, occ.get(word) || 0) }));

  const fmtRow = (r) => `    ${r.word.padEnd(16)} fired=${String(r.cnt).padStart(6)}  base=${String(r.occ).padStart(6)}  fireRate=${(100 * r.fr).toFixed(1).padStart(5)}%  lift=${r.lift.toFixed(2).padStart(6)}`;

  w.write(`── TOP 40 FALSE-POSITIVE TARGETS BY COUNT (who clears the bar most often) ──\n`);
  for (const r of [...fpArr].sort((a, b) => b.cnt - a.cnt).slice(0, 40)) w.write(fmtRow(r) + "\n");

  w.write(`\n── TOP 40 FALSE-POSITIVE TARGETS BY LIFT (count ≥ ${MINC}; enrichment vs base rate) ──\n`);
  for (const r of fpArr.filter((r) => r.cnt >= MINC).sort((a, b) => b.lift - a.lift).slice(0, 40)) w.write(fmtRow(r) + "\n");

  // ---- REVERSE VIEW: what fraction of FP mass lands in TARGET's semantic field? ----
  if (field) {
    const totalFP = fpArr.reduce((s, r) => s + r.cnt, 0);
    w.write(`\n── SEMANTIC-FIELD REVERSE VIEW (share of FP mass in field vs base rate) ──\n`);
    for (const [gname, words] of Object.entries(field)) {
      const wset = new Set(words);
      let gFP = 0, gOcc = 0; for (const r of fpArr) if (wset.has(r.word)) { gFP += r.cnt; gOcc += r.occ; }
      const shareFP = totalFP ? gFP / totalFP : 0, shareBase = gOcc / N, gFr = gOcc ? gFP / gOcc : 0;
      w.write(`  ${gname.padEnd(5)}: FP-share=${(100 * shareFP).toFixed(2)}%  base-share=${(100 * shareBase).toFixed(2)}%  enrichment=${shareBase > 0 ? (shareFP / shareBase).toFixed(1) : "∞"}×   groupFireRate=${(100 * gFr).toFixed(1)}%  groupLift=${liftOf(gFP, gOcc).toFixed(2)}×\n`);
      const members = words.map((word) => ({ word, cnt: fpFired.get(word) || 0, occ: occ.get(word) || 0 })).filter((m) => m.occ > 0).sort((a, b) => b.cnt - a.cnt);
      w.write(`    members: ${members.map((m) => `${m.word}(${m.cnt}/${m.occ}${m.occ >= 10 ? ",L" + liftOf(m.cnt, m.occ).toFixed(0) : ""})`).join(" ")}\n`);
    }
  }
};

main();
