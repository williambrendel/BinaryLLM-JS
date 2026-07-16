# Phase-1 StQP + AdaBoost — Implementation & Validation Report

**Formulation:** weak-classifier part discovery via inhomogeneous StQP (Pavan–Pelillo replicator on a
signed PMI-difference bit-graph with a unary log-odds tilt) + AdaBoost sample reweighting.
**Scope:** find `{p⁺_i}` parts only (`p⁻` = ∅, deferred to phase 2). Representation = 3F canon
`[L|L1∪L2|C]`. Head = α-weighted sum.
**Deployed entry point:** `src/core/phase1/fit.js` `fitClass(A, negPool, Mglob, opts)`.
**Validation:** `benchmark/phase1.js` on `bank`, `state`, `time` (wiki_4m, english.txt dict). Structural
diagnostics (T1/T2/T7) on a single split; all **operating-point decisions are 5-fold CV** (interleaved
folds; per fold, internal train/val split for early-stop + θ, test = held fold). `|A|` = 500 / 2600 / 4978.

---

## 1. What was built

| module | § | role |
|---|---|---|
| `negSet.js` | §2 | confusable `Neg` (single-pass masked content-fraction + δ band); decoupled SELECT vs NODE mask; frozen `n⁻`, `j⁻` |
| `affinity.js` | §1.1–1.2 | `w`-weighted `u` (log-odds) + `M` (PMI-diff sparse edge map), rebuilt/round; per-channel common-bit coeffs `commonCoefU`/`commonCoefM` |
| `replicator.js` | §4–§5 | direct payoff `π = u + Mx − ρx`; std / exp / hybrid; active-set; support extraction |
| `mfit.js` | §6 | m-of-n fit, base-rate-correct precision at weighted-recall floor 0.5 |
| `boost.js` | §7 | AdaBoost reweight loop (train-pure), dual-bound stop, `α_k=½log(rec/(1−rec))`, ε-clip |
| `head.js` | §7.2 | α-weighted sum `S_τ=Σα_k h_k > θ`; max-pool readout for the §11 A/B |
| **`fit.js`** | **§8** | **deployed: split → keep-Neg → boost (recall-based weak extraction) → val early-stop → val θ-tune** |

No dense `n×n` object is ever allocated; the payoff is one sparse matvec `Mx`.

---

## 2. Decisions (§14 record)

| # | decision | resolved value | evidence |
|---|---|---|---|
| **T1** | `M` data structure | **sparse edge map** | post-floor density 2.33% (bank) / 1.30% (state); nnz 25.4k / 154k |
| **T2** | replicator solver | **exp** | equal support (Jaccard 1.000, same `f`); exp 208 matvecs vs std/hybrid 500 |
| **T7** | `ρ` → part size | **confirmed; ship ρ=80** | \|Q\| = 5→8→38→219 as ρ = 0→5→20→100; admissible bundle regime ρ≈40–80 |
| **T8** | deployed head | **α-sum** (≫ max-pool on FP) | 3–13× lower confFP than max-pool on the same parts |
| **D1** | Neg selection | **single-pass** masked content-fraction, top-\|A\|, **δ=0.05** band | two-pass (local re-mask) **retired** — no useful θ_loc regime (mild = no-op, aggressive = collapse) |
| **D2** | global mask (SELECT) | **mild, DD≈0.05** (masks ~12 near-universal bits) | aggressive mask strips pairwise-load-bearing bits → discovery fails; mild de-saturates the neg score |
| **D3** | NODE mask (affinity nodes) | **keep (maskNodes=∅) for all targets** | CV: `ban` fragile (state 18.1±22.1%); per-target ban/keep **retired** (bank's ban-preference was slice noise) |
| **D4** | common-bit channel | **keep both** (unary carries recall, edges scaffold precision) | `banU-keepE`=ban, `keepU-banE`=keep; dropping edges raises FP under early-stop |
| **D5** | overfit control | **validation prefix early-stop** (train-pure boosting) | keep(full) FP blows up (bank 28.7±17.9%); early-stop reins it to 9.3% |
| **T9** | threshold `θ` | **val-tuned, `fpWeight` λ=1 default** | strict win over ½Σα — stabilizes (bank 58±8.9 → 61.4±2.3); λ is the precision knob |
| **D6** | support extraction | **recall-based weak (`recallTau`, ~50%)** | generic `tauSupp` extracts a STRONG part (66–80%) → AdaBoost just re-finds it; recall-calibration yields a genuine weak learner (§8) |
| T3/T4/T5 | precision basis / soft m-of-n / `a⁻:a⁺` | `<unresolved>` — defaults (w-weighted / hard / 1:1) | not yet ablated |
| T11 | cross-formulation agreement (vs seq/parallel) | `<unresolved>` | not yet run |

**Retired approaches** (tested, closed — do not relitigate): per-target ban/keep selector (slice noise);
`ban` node mask (fragile on state); two-pass local-mask Neg (no regime); scaffold = edges-only bridging
(≈ ban, refutes the connectivity hypothesis); global soft-coefficient `c` (opposite optima per target).

---

## 3. Structural diagnostics (single split)

### T1 — the affinity is sparse and the mask self-zeroes
`M` density **1.3–2.3%** (well under 5%). Signed structure is real: bank has 11.7k attractive (`M>0`) and
**13.7k repulsive** (`M<0`) edges — repulsion (precision pressure) is the majority. `u` ranges ±3.7/±4.3,
with **216/729 bits at `u≈0`** — the common bits self-zeroing on the unary channel.

### T7 — the `−ρI` size knob works
Higher ρ ⇒ larger/coarser parts (Pavan–Pelillo direction): `|Q| = 5 → 8 → 38 → 219` as `ρ = 0 → 5 → 20 →
100` (bank, round 1). Small ρ gives tight cliques that fail the recall floor; the admissible **bundle**
regime is ρ ≈ 40–80.

### T2 — exp solver ships
All three solvers converge to identical support (Jaccard 1.000) at the same `f`; exp reaches it in **208**
matvecs vs **500** for std/hybrid.

---

## 4. The Neg / mask / node arc (the substance)

### 4.1 Neg selection & mask (D1, D2)
`Neg` = the confusables `p⁺` implicates, selected single-pass by masked content-fraction
`|P⁺∧y| / |y∖D_τ|`, top-`|A|`, skipping a δ=0.05 near-duplicate band (always positive; inert at the sweet
spot but retained as a saturation safeguard). The mask `D_τ` is a **mild** global frequency ladder
(DD≈0.05): aggressive masking removes bits whose *pairwise* co-occurrence is load-bearing → `PMI⁺≈PMI⁻` →
`M` cancels → discovery fails; mild masking only de-saturates the selection score. A **two-pass** variant
(re-run the mask mechanism locally over `{pos ∪ Pass-1 neg}`) was built and **closed**: no useful `θ_loc`
(mild = masks nothing, contexts too diffuse; aggressive = same collapse as an aggressive global mask).

### 4.2 Node mask: ban vs keep, under CV (D3)
A common bit self-cancels in a PMI-difference affinity (`PMI⁺≈PMI⁻≈log(1/f)`, `u≈0`, `−ρx` pushes it out),
so it need not be banned to avoid re-discovery — but banning still shrinks the graph and cuts noise. The
question is whether *keeping* it as a node helps. 5-fold CV (ρ=80, δ=0.05), test-rec / confFP (mean±sd):

| target | ban | keep | **keep+earlystop** |
|---|---|---|---|
| bank | 52.8±3.3 / 7.5±1.1 | 76.6±10.2 / **28.7±17.9** | **58.0±8.9 / 9.3±5.7** |
| state | **18.1±22.1** / 2.8±3.4 | 77.8±1.9 / 15.1±0.8 | **77.7±0.9 / 15.3±0.7** |
| time | 81.6±1.0 / 12.0±0.8 | 82.1±1.8 / 10.8±2.3 | **81.5±1.2 / 9.3±0.7** |

`ban` is **fragile** — on state it fails in most folds (18.1 ± 22.1%), because masking strips load-bearing
structure. `keep` has the best recall but its FP **blows up on bank** (28.7 ± 17.9%). The earlier
single-split "bank prefers ban" (44 vs 32%) was **slice noise**: on CV bank's preference is unstable and the
per-target ban/keep selector is retired. Only **keep+earlystop** is strong *and* stable on all three.

### 4.3 Channel decomposition: recall is unary (D4)
Scaling the common bits' unary (`commonCoefU`) and edge (`commonCoefM`) channels independently isolates
where the keep-effect lives:

| target | ban | keep | banU-keepE (edges only) | keepU-banE (unary only) |
|---|---|---|---|---|
| state | 18.1±22.1 / 2.8 | 77.8 / 15.1 | **18.1±22.1 / 2.8** (= ban) | **77.3 / 14.7** (≈ keep) |
| time | 81.6 / 12.0 | 82.1 / 10.8 | **81.6 / 12.0** (= ban) | 82.9 / 13.2 |

`banU-keepE` is **bit-identical to ban** on state/time — the common bits' **edges carry ~zero recall
signal** (they self-cancel in `M`). All the keep-effect flows through the **unary log-odds**. But edges are
not free to drop: under early-stop `keep+es` (both channels) holds FP at 9.3% (bank, time) vs
`keepU-banE+es` at 14.9 / 12.2% — the edges **scaffold precision**. So keep both.

### 4.4 Why only state gains from keep
State's positive class is a **union of disconnected sub-patterns** (heterogeneous senses of "state"); its
discriminative bits fragment into many small clusters. The common bits, entering the m-of-n gate via their
unary, fire broadly enough to raise per-part coverage — state finishes in **K=4 broad parts** (vs 23 narrow
ones under ban), which generalize (73% vs 53%). Bank doesn't fragment (K unchanged) so keeping only adds
overfit; time is already connected (indifferent). Confirmed by the scaffold refutation (§Retired): keeping
common bits *only in `M`, not in `Q`* reverts to ban — the value is the gate feature, not graph bridging.

### 4.5 AdaBoost consistency
The early-stop is **outer** only: replicator, `mfit`, `α`, and reweighting all stay on **train**, so the
boosting objective is untouched and the guarantee holds. Validation picks *where to cut* the coverage-greedy
sequence (frequent-generalizable parts first, rare-memorizing tail last), leveraging AdaBoost's ordering
rather than overriding it. Pushing held-out *into* the α/acceptance would break consistency — hence it is
confined to the stopping rule.

---

## 5. The FP floor (T9)

Per-part breakdown (`FPDIAG=1`) of a deployed head: **`m≥2` parts are clean (partFP 1–2%)**, **`m=1` parts
are the drivers (10–22%)**, and dropping any single part barely moves head FP (the false positives are
redundant across parts). So the part set can't fix it — only the α-sum **threshold θ** can. `m=1` is forced
by the recall floor on heterogeneous classes, so it cannot simply be forbidden. Fix = **tune θ on the
validation ROC** (same `recall − λ·FP` objective), `fpWeight = λ`. 5-fold CV:

| config | bank | state | time |
|---|---|---|---|
| fixed θ = ½Σα | 58.0±8.9 / 9.3±5.7 | 77.7 / 15.3 | 81.5 / 9.3 |
| **θtune λ=1 (default)** | **61.4±2.3 / 9.8±2.3** | 77.2 / 14.9 | 81.7 / 9.5 |
| θtune λ=2 | 57.8 / **7.6** | 75.6 / 13.9 | 78.4 / **7.2** |
| θtune λ=3 | 50.6 / 4.9 | 52.9±15.2 / 6.3 | 76.3 / 6.7 |

**λ=1 is a strict win** over the fixed heuristic — same-or-better recall with a much tighter operating point
(bank variance ±8.9 → ±2.3), now the `fitClass` default. **λ is the precision knob**: λ=2 trims ~2pts FP for
~2–3pts recall; λ=3 goes lower but state destabilizes. The floor is a genuine recall/FP trade — the
confusables are boundary-hard by construction (`p_min = 0.5`) — not eliminable, but honestly val-tunable.

---

## 6. Part redundancy, extraction, and class separability

### Metric legend (units for every table in §6)
| symbol | unit | meaning |
|---|---|---|
| parts discovered / kept | count | dominant sets AdaBoost extracts / kept after val early-stop |
| \|Q\| | bits (count) | part size = bits in the m-of-n gate |
| m | bits (count) | how many of \|Q\| bits must co-occur to fire (`m1`=OR) |
| α | dimensionless | head vote weight `½ln((1−ε)/ε)`; **α-cv**=std/mean (<0.3 flat, >0.7 skewed) |
| intra/cross-Jaccard | 0–1 | mean pairwise `|A∩B|/|A∪B|` of part bit-sets (1=identical, 0=disjoint) |
| containment | 0–1 | `|A∩B|/min(|A|,|B|)` (1=nested) |
| union/Σ\|Q\| | 0–1 | `|⋃parts|/Σ|Qk|` (1=disjoint, →0=fully redundant) |
| retention \|Q\|/⋁A | % | part size as fraction of the OR-ed p⁺ (`⋁A`) bit-set |
| train / held-rec | % | positive contexts the head fires on (train slice / held test fold) |
| confFP | % | confusable negatives the head fires on |
| gap | percentage points | train-rec − held-rec (overfit) |
| per-band % | % | share of a bit-set in each 3F band `[F2-adj dd1 / F1-near dd2-3 / F0-far dd≥4]` |
| λ (`fpWeight`) | dimensionless | FP penalty in val selection `recall−λ·FP`; auto `clamp(1500/|A|,1,3)` |

### 6.-1 FINAL deployed config + the diversity trade (`E2E`, `PEEL=1`, 5-fold)
Deployed = keep-Neg → recall-based weak extraction (`tauFloor=0.6`) → AdaBoost → val early-stop → val θ-tune,
with auto FP-penalty **λ=clamp(1500/|A|,1,3)** (bank 3 / state,time 1). Stats:

| metric | bank | state | time |
|---|---|---|---|
| λ (auto) | 3 | 1 | 1 |
| parts disc / kept (count) | 8.8 / 4.6 | 15.2 / 14.6 | 21.4 / 20.8 |
| \|Q\| med (bits) | 94 | 277 | 286 |
| α mean, cv | 0.26, ~0.4 | 0.23, 0.23 | 0.22, 0.17 |
| intra-Jaccard (0–1) | 0.43 | 0.34 | 0.28 |
| held-rec (%) | 48.6 | 76.0 | 80.8 |
| confFP (%) | **7.7** | 12.6 | 8.9 |
| gap (pp) | 8.0 | 18.1 | 12.0 |

**Bank FP fixed & stabilized: 18.9%±10 → 7.7%±3.1** (auto-λ=3), at a held-recall cost (65→49%) — the inherent
FP/recall trade for a data-poor class. **Bit-peel** (remove accepted parts' bits from the affinity → force a new
clique) achieves *perfect disjointness* but collapses recall and part count:

| config | intra-Jaccard | held-rec b/s/t (%) | parts kept | proves |
|---|---|---|---|---|
| generic strong | 0.80/0.45/0.47 | 61/77/82 | 5–25 | one bundle re-found each round |
| recallTau@0.5 | 0.99/0.70/0.76 | 46/70/58 | 3–29 | weakness ≠ diversity |
| **recallTau@0.6 (deployed)** | 0.43/0.34/0.28 | 49/76/81 | 5–21 | **sweet spot: diverse-ish, full recall, flat α** |
| + bit-peel | **0.00/0.00/0.00** | 18/70/60 | 1–3 | disjointness → recall collapse |

⇒ **The classes are irreducibly ~1–3 dominant bundles** (state ~3 distinct disjoint parts at 70%; bank/time ~1).
No rich disjoint decomposition exists. Deployed keeps the sweet spot (`tauFloor=0.6`, no peel); peel is retained as
a flag — the *proof* that fuller disjointness costs recall.

### 6.0 Deployed stats (recallTau, tauFloor=0.6 — `E2E`/`DIST`/`XCLASS`, 5-fold)

| stat | bank | state | time |
|---|---|---|---|
| parts discovered / kept | 8.8 / 8.6 | 15.2 / 14.6 | 21.4 / 20.8 |
| \|Q\| med [min–max] | 102 [61–133] | 277 [63–456] | 286 [28–477] |
| α mean, cv | 0.26, 0.44 | 0.23, **0.23** | 0.22, **0.17** (flat, all ≥0.20) |
| intra-class Jaccard | **0.43** | **0.34** | **0.28** |
| cross-class Jaccard | ~0.01 | ~0.01 | ~0.01 |
| retention \|Q\|/⋁A · union/⋁A | 10% · 19% | 8% · 18% | 6% · 15% |
| train / held-rec · confFP | 81.1 / 65.4 · 18.9% | 94.1 / 76.0 · 12.6% | 92.8 / 80.8 · 8.9% |

vs prior configs — **the tauFloor=0.6 default improves every axis** the redundancy work exposed:

| axis | generic (strong) | recallTau@0.5 | **recallTau@0.6 (deployed)** |
|---|---|---|---|
| α | skewed (cv 0.85–1.91) | collapsed ~0 (cv 1.16) | **flat, ≥0.20 (cv 0.17–0.44)** |
| # parts (state/time) | 6 / 11 | 48 / 50 | 15 / 21 |
| intra Jaccard (b/s/t) | 0.80/0.45/0.47 | 0.99/0.70/0.76 | **0.43/0.34/0.28** |
| held-rec (b/s/t) | 61/77/82 | 46/70/58 | **65/76/81** |

At 60% coverage per part the reweighting finally has enough uncovered mass to shift the dominant set → genuinely
**more diverse parts** (intra-Jaccard 0.80→0.43) with a healthy flat α — the closest AdaBoost gets to a real
weak-learner ensemble *without* bit-peeling. Cross-class parts stay disjoint (~1%); residual intra-class sharing
concentrates in the generic **F2-adjacent** band (50/35/39) — a band-aware mask target. Blemish: bank FP 18.9%±10
(tiny |A| over-fires at the higher floor); state/time clean.

### 6.1 Under generic `tauSupp` the parts are one bundle, re-expressed (E2E, `E2E=1`)
End-to-end, the parts are **big loose bundles** (|Q| med 330–510 bits under generic `tauSupp`), **few** per class
(discovered ≈6/11/27), and **highly redundant**: pairwise Jaccard **0.80** (bank) / 0.45 / 0.47; bank's 25 parts
draw from a ~600-bit pool (`union/Σ|Q| = 0.06`, ~94% redundant). AdaBoost's *sample* reweighting cannot dislodge
the globally-dominant clique — it re-finds the same core each round.

Three interventions confirm the redundancy is **structural**, not a tuning artifact:
- **Core-trim** (`CORECV`, `coreT`): pruning to high-weight core makes Jaccard *rise* (0.80→0.87) and recall collapse.
- **Recall-based weak extraction** (`RTCV`, `recallTau`): smaller genuinely-weak parts, but *more* redundant, not less:

  | target | generic `tauSupp` (strong) | recall-based (weak, D6) |
  |---|---|---|
  | bank | K25, \|Q\|411, Jacc 0.80, union/⋁A 52%, held 61.4%, fp 9.8% | K2, \|Q\|38, Jacc **0.99**, union/⋁A **4%**, held 46.0%, fp 7.5% |
  | state | K5, \|Q\|512, Jacc 0.45, union/⋁A 19%, held 77.2%, fp 14.9% | K12, \|Q\|97, Jacc 0.70, union/⋁A 4%, held 69.5%, fp 11.4% |
  | time | K10, \|Q\|328, Jacc 0.47, union/⋁A 15%, held 81.7%, fp 9.5% | K29, \|Q\|19, Jacc 0.76, union/⋁A **1%**, held 57.9%, fp **2.7%** |

  Weak parts all grab the *same* top-core (Jaccard ↑, span ↓); tiny FP (time 2.7%) but recall craters. Weakness ≠ diversity.
- **Partition** (`PART`): splitting one dominant set into k disjoint sub-parts + OR-head reconstructs its recall
  **exactly** (Δrec = 0.0pt, per-sub 16–61%, Jaccard 0). So the class is **one bundle**; "parts" are a granularity
  choice, and diverse weak learners are obtainable by *partition* (deterministic), not reweighting. Recall = bundle
  span; precision = co-firing threshold (`m`-of-`n` at bit level ≡ `k`-of-`K` at head level) — orthogonal axes.

**Deployed choice (D6):** recall-based weak extraction — the correct weak-learner input to AdaBoost, small
train/held gap (bank 29→**4.7pt**), low FP; the redundancy is an open structural question (§7).

### 6.2 Class separability — intra vs cross overlap (`OVERLAP=1`)
Signature Jaccard×100 (mean±sd), within-class (intra) vs between-class (cross), for the whole 3F signature and per
positional band (**F2**=adjacent dd1 · **F1**=near dd2–3 · **F0**=far dd≥4). Targets bank/state/time (|ctx| 500/2600/4978):

| band | intra-bank | intra-state | intra-time | **cross** |
|---|---|---|---|---|
| 3F-all | 2.9±7.0 | 2.1±5.1 | 2.8±6.7 | 1.9±4.9 |
| **F1-near** | **6.2±13.9** | 2.7±9.3 | **5.3±14.8** | 3.1±9.6 |
| F2-adj | 5.6±23.0 | 9.6±29.5 | 8.0±27.1 | 6.0±23.7 |
| F0-far | 1.1±4.0 | 1.2±4.4 | 1.0±4.0 | 1.0±4.1 |

- Overlap is **low and sparse** everywhere (short contexts, large vocab); intra only slightly exceeds cross overall
  (2.1–2.9 vs 1.9) — a ~1pt separability margin on the whole signature.
- **The discrimination lives in the near band `F1` (dd 2–3):** intra (bank 6.2, time 5.3) clearly exceeds cross (3.1)
  — the widest margin. State is weaker here (2.7).
- **The adjacent word `F2` does *not* separate** (intra ≈ cross ≈ 6–9, huge ±): the immediately-preceding token is
  shared function words across all classes.
- **The far band `F0` is noise** (intra = cross ≈ 1): distant words carry no class signal.

This explains the loose-bundle parts: the class signal is diffuse and concentrated in one band, with a thin
intra-minus-cross margin — exactly the regime that yields broad low-`m` detectors rather than tight conjunctions.

**Cross-class PART overlap (`XCLASS=1`) is ZERO in every band** (discovered `Q` sets of bank/state/time share no
bits, Jaccard 0.0 for 3F-all and each F-band), while intra-class part overlap is high (state 80 / time 79). So the
redundancy is **purely intra-class** (the stuck dominant set); across classes the parts are cleanly disjoint at this
scale — consistent with §6.4 (3 classes ≪ 250–500-core capacity → room to be disjoint; forced overlap only at scale).

*Note on the mask:* `Mglob` is already computed from **signature bit-frequency** (`bitCount/Ntot > DD` over featurized
contexts), not an external word-frequency list. Given the band asymmetry (F2 generic, F0 noise, F1 discriminative), a
**band-aware** frequency mask (prune F2/F0 harder, preserve F1) is the open variant.

### 6.3 Distribution shapes — head weights skewed, part sizes flat (`DIST=1`)
Pooled over folds (cv = std/mean; flat < 0.3 < moderate < 0.7 < skewed):

| config | α (head wt): mean, cv, range | shape | \|Q\| (size): mean, cv | shape |
|---|---|---|---|---|
| generic | bank 0.17, **1.91**, [0–2.61] · state 0.70, 0.43 · time 0.36, 0.85 | **skewed** | bank 371, 0.26 · state 418, 0.43 · time 323, 0.41 | flat–moderate |
| recallTau | bank 0.01, 1.16 · state 0.09, 1.07 · time 0.01, 1.16 (all [0–0.08]) | skewed-tiny | bank 39, **0.09** · state 122, 0.39 · time 20, **0.24** | flat–moderate |

- **Part size is flat-to-moderate** — ρ sets a characteristic size, so `|Q|` clusters tightly (recallTau bank cv 0.09).
- **Head weight α is NOT flat.** *Generic*: a dominance skew — 1–2 parts carry the vote (bank α up to 2.61, median
  0.11) → effectively a single classifier + low-α tail (redundancy in the weights). *recallTau*: all α **tiny with a
  mass at ~0** — each part sits just above 50% recall so ε≈0.5 ⇒ α≈0; many weak parts contribute ~nothing.
- Neither is a healthy weak-learner ensemble (want α bounded away from 0 and comparable across diverse parts). Argues
  the recall floor should sit **above 0.5** (≈0.55–0.6) so α doesn't collapse.

**Fix — recall floor `tauFloor` above 0.5 (`TFCV=1`; now default 0.6):** raising the floor makes α substantive AND flat,
and recovers the held-recall the α≈0 collapse was costing:

| target | floor 0.50 (α-cv) | floor 0.60 (α-cv) | held-rec 0.50→0.60 | confFP 0.50→0.60 |
|---|---|---|---|---|
| time | 0.01 (1.16) | 0.22 (**0.17**) | **57.9 → 80.8%** | 2.7 → 8.9% |
| state | 0.09 (1.07) | 0.23 (0.23) | 69.5 → 76.0% | 11.4 → 12.6% |
| bank | 0.01 (1.16) | 0.18 (0.65 @0.55) | 46.0 → 64.2% (@0.55) | 7.5 → 16.4% |

At floor 0.6 the parts are genuine weak learners with **flat** substantive α, at ≈ the strong-part recall (time 81%).
**FP confirms the α-flatness link:** in the flat regime (floor ≥0.55) confFP is *stable* (time 8.8→8.9→8.6 across
0.55/0.60/0.65) — FP only moves crossing out of the α-collapsed regime (0.50→0.55). Flat α ⇒ stable operating point.

### 6.4 Union-vs-core, multi-DS vs balanced partition, per-band distribution (`BANDS=1`)

**∪(AdaBoost parts) ≠ the core (1 dominant set).** The union is 4–5× the core and mostly *outside* it, covering
only part of it — so the parts are genuinely different dominant sets, not sub-parts of one core (unlike the old
configs):

| target | ∪ covers of core | ∪ in-core | \|core\|→\|∪\| |
|---|---|---|---|
| bank | 46% | 40% | 200→225 |
| state | 69% | 16% | 164→718 |
| time | 73% | 14% | 162→861 |

**Multiple dominant sets vs the core partitioned into K band-balanced sub-parts** — a recall/cleanliness trade:

| construction | intra-Jaccard | OR-rec (train) | OR-FP (Neg) |
|---|---|---|---|
| Multi-DS (AdaBoost) b/s/t | 0.50 / 0.33 / 0.28 | **95 / 97 / 97%** | 39 / 31 / 35% |
| Balanced partition of core | **0.00** | 83 / 72 / 81% | 35 / **10 / 12%** |

Multiple DS win on recall (span beyond one core); the balanced partition is perfectly **disjoint** and much
**lower-FP** but capped at the single core's recall. ⇒ synthesis: find K distinct cores (AdaBoost) *then*
band-balance-partition each — disjoint low-FP sub-parts *and* 97% coverage. Balanced partition keeps each sub-part
at the same band mix (no L/C segregation).

**Per-band bit distribution [F2-adj / F1-near / F0-far] %:**

| bitset | bank | state | time |
|---|---|---|---|
| ⋁A (OR p⁺) | 15/30/**55** | 16/34/**50** | 9/36/**55** |
| core (1 DS) | 28/26/47 | **37/35**/29 | 25/34/41 |
| ada ∪parts | 23/34/43 | 35/28/38 | 22/40/37 |

`⋁A` is far-band (F0) dominated (50–55%), but discovery **shifts toward the discriminative F2-adjacent + F1-near
bands** (state core F2 16→37%, F0 50→29%) — quantitative support for a **band-aware mask** that prunes F0-far harder.

**Band-aware node mask, tested (`BANDMASK=1`, held-out 5-fold):**

| target | config | \|P⁺ nodes\| | bits-used | held-rec (%) | confFP (%) |
|---|---|---|---|---|---|
| state | keep-all | 3653 | 639 | 76.0 | 12.6 |
| state | **prune-F0(far)** | 1812 (−50%) | 490 (−23%) | 75.0 | 13.3 |
| state | prune-F0+F2 | 1248 | 510 | **55.3** | 20.2 |
| time | keep-all | 4822 | 730 | 80.8 | 8.9 |
| time | **prune-F0(far)** | 2129 (−56%) | 557 (−24%) | 79.8 | **8.2** |
| time | prune-F0+F2 | 1721 | 656 | 70.9 | 11.8 |

**Prune F0-far only = clean win:** ~50% fewer nodes, ~23% fewer bits, held-rec −1 to −2pt (noise), FP ≤ baseline.
**Prune F2-adjacent too = recall collapse** (bank 48.6→4.4%): despite F2's intra≈cross *signature* overlap, the
*specific* adjacent word is load-bearing (cores use 25–37% F2 bits). ⇒ band-aware mask prunes **F0-far only**;
keep F1-near + F2-adjacent. Composes with the tight-core partition (prune F0 → partition the F1/F2 core).

**Frequency-pruning common words ("the") from the nodes also fails** — removing ~66 common bits collapses state
(76→**17.7%**) and time (81→63%). The common-word bits are **load-bearing for coverage** (D3: "the" before the
target fires on most positives = the OR bundle's coverage backbone); FP drops (state 12.6→2.6, they're the loose
hi-recall/hi-FP bits) but recall craters. Frequency-masking can't separate *the*-as-noise from *the*-as-coverage.
The surgical exclusion of "the" is by **x\* weight** (it self-cancels, `u≈0` → low weight → excluded by the tight
~25-bit core), **not** by a frequency node-mask. So: band-aware mask = **F0-far only**; "the"-pruning happens inside
the tight-core weight selection.

**Efficiency — core-partition vs multi-DS on held-out (`COREPART=1`):** one strong core partitioned band-balanced
into K uniform-α sub-parts vs the deployed multi-DS ensemble. bits-used = distinct bits across all parts (the
efficiency metric):

| target | config | K | bits used | held-rec (%) | confFP (%) |
|---|---|---|---|---|---|
| state | multi-DS | 15 | 639 | 76.0 | 12.6 |
| state | core-partition | 15 | **191** (3.3×↓) | 69.6 | 13.9 |
| time | multi-DS | 21 | 730 | 80.8 | 8.9 |
| time | core-partition | 21 | **178** (4.1×↓) | **81.2** | 15.3 |

**The core-partition matches held-out *recall* with 3–4× fewer bits** (time 81.2% vs 80.8%) ⇒ the multi-DS
**over-spends bits** (recall-based extraction is "too generous" reaching the floor). But its extra bits buy
**precision**, not recall — core-partition FP is higher (time 15.3 vs 8.9). Cause: the core here is the *loose*
full ~200-bit dominant set (incl. low-weight tail) split into naive `m=1` chunks → loose firing.

**Tight-core sweep (`TIGHTCORE=1`, weight threshold T on `x*·|Q|`) — how to find the core, and the FP answer:**
tight core = high-weight bits (T≈1→~60–90 bits, T≈2→~20–30). Tightening (↑T) **monotonically lowers FP** (time
15.3→5.1→2.8→1.1%) **but lowers recall in lockstep** (time 81→66→58→34%). ⚠ **There is no "core wins outright":**
it's one recall/FP ROC — no single-core T matches the multi-DS high-recall+low-FP corner (time multi-DS 81%@8.9%;
tight core gives 81%@15% *or* 66%@5%, never 81%@9%). The multi-DS reaches that corner by spanning **multiple** cores
(discriminative structure a single core lacks) — its 4× bits are earned there. **The efficiency win lives at the
loose core (T=0):** matches recall with ~4× fewer bits (higher FP). T is the recall/FP dial; the recall floor (0.6)
is a separate, already-swept knob. (Earlier "tight core fixes the FP gap for free" was wrong.)

**Cross-word (11 canon targets `data/phase1_targets.txt`, |A| 85–4978):** the trade-off is robust across the
frequency spectrum — FP↓ with T monotone (10/11), recall/FP trade universal, loose core = recall/efficiency point
everywhere (~136–284 bits, ρ-set size ~constant across words). But the **~25-bit core (T=2) is too tight for most**:
recall collapses on diffuse words (state 18%, world 30%, music 17%) and only holds for well-separated ones (time
58%, physics 42%, century 56%). `century` is cleanly separable (T=0 88%/5%); `government` is very loose (T=0
85%/**32%** FP). ⇒ **no universal T — per-word T on validation**, each word landing at its own operating point by
core diffuseness.

**Tight-core + AdaBoost — does it escape the core? (`TIGHTBOOST=1`).** Tightening (↑coreT) makes successive cores
*more* identical (inter-core Jaccard state 0.37→0.71, time 0.38→0.72→0.84) — AdaBoost reweighting **re-finds the same
dominant clique**, diversity lives in the loose *tails* not the tight cores. **But tight-core + peel reveals ~2
distinct tight cores**: state/time (coreT=1 +peel) give **K=2, Jaccard 0.00, each recall 53–63% (>50%), α 0.06–0.28,
union 83%** — two disjoint *valid* weak learners, then `no_valid_m` (no 3rd core clears 50%; matches the ~1–3 bundle
count). **# distinct cores is word-specific (1–4), not a fixed "2"** (loose-DS+peel to exhaustion across canon): monolithic
classes → **1** (time/world, one bundle covers 81–83%; rare physics/philosophy/hydrogen, data-poor), state/
government/bank → 2, century/river → 3, **music → 4** (union 98%). It's the number of distinct discriminative
sub-structures — a *semantic* property (polysemy), **not frequency-monotonic** (rare→1, very-common→1, mid-freq
content→3–4). Design: don't fix a count — **peel until `no_valid_m`** auto-discovers the right number per class.
Reconciliation of "tighter is worse": it's the *stopping criterion* — fixed weight-threshold (`coreT`) is
coverage-blind (same clique re-emerges), recall-target (`recallTau`) tracks reweighting → escapes; the escape is
from the coverage-aware stop, not bit count.

**Disjoint ≥50% cores AND union≈one core are mostly incompatible (`COREDECOMP=1`).** Partitioning the dominant core
into K disjoint band-balanced pieces (union=core by construction), the max K where each piece still clears 50%:
**common/data-sufficient words → K=1** (time/state/world/music/river — the core is *indivisible* at 50%; a disjoint
half drops below 50% because the core covers ~80% thinly), century/government/bank → 2, rare physics/hydrogen → 5–6
(an overfit artifact — a huge core over ~90 positives). Decomposability = core redundancy, **inverse to data
density**. So: **multiple disjoint ≥50% cores** (peel) require the union to grow 2–6× beyond one core (adds bits),
while **union = one core** yields a single indivisible ≥50% unit for real words. You get disjoint-multiplicity XOR
union=core, not both — for data-sufficient targets the dominant core is the **atomic unit**.

So **tight-core + peel** yields a clean disjoint ensemble *without* the loose-peel recall collapse — the
diverse, substantive-α heads the parsimony argument wants. Promising; needs tight *and* peel (tightening alone
converges, loose-peel alone collapses).

**Picking tightness without a sweep (`NEFF=1`).** Inverse participation ratio `n_eff = 1/Σx*²` (weight
concentration) as a closed-form core size is **rejected**: it doesn't track recall, so for diffuse words the
top-`n_eff` core falls below 50% (government 46%, music 34%, river 45%). The correct statistic is **recall coverage**
— the smallest top-weight prefix clearing the recall floor (= what `recallTau` computes): a direct per-round
calculation (no sweep) that always clears the floor, is near-precision-optimal (FP 2–8%), and **auto-adapts
tightness to concentration** (century 4 bits, time 11, physics 22 vs music 181, government 104). So adaptive
min-recall tightness = the deployed `recallTau`; `n_eff` is the plausible shortcut ruled out.

**Scale test (8 words spanning |A|=85–3044):** core-efficiency **scales with frequency**. Common (|A|≳1500): core
wins big — world 174b vs 600b (3.5×) at 78.6% vs 80.0%; river 149 vs 310 (2.1×) at 70 vs 74. **Rare (|A|<150)
reverses** — multi-DS is already tight (physics 98b, hydrogen 87b) while the noisy small-sample core is *larger*
(220–257b), ratio 0.4–0.8×; rare recall is weak everywhere (39–47%). FP is **universally higher** for the
core-partition (loose core). Since common words dominate token counts, the core-efficiency win holds for the bulk of
the vocabulary; the fix (tight-core partition) is the same across the range. (Rare-word cmp confounded by adaptive
λ=3 on multi-DS vs λ=1 on the partition θ-tune.)

### 6.5 Capacity / packing limit (noted)
**3F = 99,912** dims (F = 33,304). A generic core `|Q|` ≈ 200–400 bits = **0.20–0.40%** of the space, so only
`⌊3F/|Q|⌋` ≈ **250–500 disjoint cores** fit; beyond that, pigeonhole *forces* overlap. Implications:
- The low measured cross-overlap (1.9%) holds only because there are **3** classes, not 500. At vocabulary scale
  (F = 33k word-classes ≫ 500 slots) disjoint cores are impossible → **cross-class overlap is structural at scale**,
  a distinct source from the intra-class stuck-dominant-set redundancy (which occurs at any scale).
- The binding number is the **discriminative core (~20–30 high-weight bits)**, not the full support: packing 25-bit
  cores raises capacity to ~3,300–5,000. So loose 300-bit bundles **waste ~10× capacity** (0.3% of space for ~0.02%
  of discriminative content). Even so, 4,000 ≪ 33k.
- ⇒ At scale the representation **cannot allocate disjoint cores; it must discriminate combinatorially (m-of-n over
  shared bits)** — which is precisely the gate's role. Argues for **tight cores over loose bundles** (denser packing,
  leaving the shared tail to combinatorics) — pushing the recall-based extraction further toward the ~25-bit core.

## 6.6 Scale evaluation on real WikiText-103 (data-hungry hypothesis CONFIRMED)

The earlier tables are on a 22 MB / 4 M-word slice. Evaluated on a **120 MB / 22 M-word slice of
`data/corpora/wiki.train.txt`** (5.5×), `|A|` grows proportionally and **held-recall climbs while the overfit
gap collapses** (deployed `fitClass`, 5-fold):

| word | \|A\| (4M → 22M) | held-rec (4M → 22M) | train/held gap (4M → 22M) | confFP (22M) |
|---|---|---|---|---|
| bank | 500 → 1890 | ~49–65% → **79.6%** | 29pt → **13.9pt** | 16.4% |
| state | 2600 → 12772 | 76% → **81.0%** | 18pt → **9.8pt** | 12.4% |
| time | 4978 → 27069 | 81% → **87.3%** | 12pt → **3.6pt** | 11.8% |

Bank (the data-poor problem child) reaches **79.6%**; time's gap is essentially gone (**3.6pt**). So the modest
held-recall in §6.0 was a **corpus-size artifact, not a pipeline defect** — the arc's "data-hungry" prediction
holds. Part diversity (intra-Jaccard 0.34–0.44) is stable across scale. The full train (535 MB / 99 M words)
would push `|A|` another ~4× higher.

**FP does not jump at scale — the recall gains come nearly free.** At a *matched* operating point (λ=1):
bank 18.9 → **16.4%** (slight ↓), state 12.6 → **12.4%** (flat), time 8.9 → **11.8%** (slight ↑ — the
recall/FP frontier, +3pt FP for +6pt recall). The bank number above (16.4%) is *not* a scale-driven jump from
its 4M value of 7.7%: that 7.7% used **adaptive λ=3** (bank |A|=500 < 1500 → small-class FP penalty), whereas
at 22M bank has |A|=1890 → **λ=1** (no penalty — correctly, it's no longer a small class). At matched λ, bank
FP *decreased* with more data. So: **more data → large recall gains, roughly flat FP at a fixed λ.**

## 6.7 Performance, storage, and optimization avenues

**Measured (22 M-word slice, `PERF=1`):** featurization (corpus scan + encode) ~10 s **one-time, shared**;
per-word extraction 2 s (bank |A|=1890) → 21 s (state |A|=12772); on-disk parts 4–29 KB/word (delta-varint
bit-ids + m + α). **All-vocabulary estimate:** extraction is embarrassingly parallel (per-word independent) →
~10 CPU-hours for ~1M words on 16 cores.

**Memory — disk saving is real; RAM is corpus-bound (`OPT=1`):**
- **Disk: 6.6–7.5× smaller by storing the union, not Σ|Q|.** Multi-DS stores the *sum* of part sizes, but
  parts overlap heavily so bits are stored many times (state Σ|Q|=4732 vs union=718 → **6.6×**; time 6419 vs
  861 → **7.5×**; single-part bank 1.0×). Two ways: **adaptive-core** (disjoint cores ⇒ Σ|Q|=union for free,
  ~0.6 GB vs ~2.4 GB all-vocab); or **union + per-part membership bitmask** on the current multi-DS —
  **exact, no algorithm change** (~5× for state), available now.
- **RAM: these optimizations do NOT save it.** The negPool index is a ~9 MB **cost** (1.14 M posting
  entries, shared/one-time — negligible vs the ~0.5–1.3 GB footprint). That footprint is **corpus-dominated**
  (all sentences + encode cache + negPool held in memory); the real RAM levers are separate — **stream/chunk
  the corpus** (the big one), a smaller negPool cap, or lossy `|A|` subsampling.

**Where the time goes (`PROF=1`, 4M):** per-word extraction is **81–82 % replicator, 14–16 % affinity,
<5 % everything else** (mfit/support/reweight). Edges/round scale hard with `|A|` (22k→217k). So the
replicator matvec is the only lever that matters for single-word speed.

**Optimization avenues (ranked by measured impact):**
1. **Replicator early-stop — IMPLEMENTED + validated ≈2.6× `[CANDIDATE, quality-neutral]`.** The replicator
   hits its 500-iter cap essentially every round (avg 467–500), but the OUTPUT is `support(x*)` + its
   weight-ordering, not the fully-converged `x` — the support *set* stabilizes far earlier. `replicate` opt-in
   `suppPatience` stops once the support is unchanged for K iters (adaptive, `minIter=20`). `suppPatience=3`
   (`MAXITER=1`): **2.6–2.8× faster replicator (~2.2× total/word), recall identical (99.1–99.3 %), FP
   equal-or-better on the real multi-part words** (state 27.12 vs 27.27, time 33.57 vs 33.67), K preserved.
   A fixed `maxIter=100` gives the same 2.5–2.9× but is non-adaptive. **Default off** (not bit-exact — a few
   support bits shift); recommend enabling in the deployed head after a CV pass. This is the single biggest
   per-word win.
2. **Solver / concavity — TESTED, NOT the bottleneck** `[MEASURED]`. The deployed update is `exp`
   (multiplicative-weights, fixed η=1). Hypothesis: `M` indefinite ⇒ non-concave ⇒ slow. Measured (`SOLVERS=1`):
   the round-0 graph is strongly **repulsion-dominated** (top-magnitude eigenvalue ≈ −144 (state) / −185 (time)),
   so `(M−ρI)` at ρ=80 is comfortably neg-definite / concave — yet **all three solvers still burn ~400–500
   iters** and switching exp→std/hybrid buys only ~8 % at equal quality. Conclusion: the full budget is genuine
   slow approach to the tight `1e-6` L1 tolerance on a flat near-fixed-point landscape, **not** a concavity or
   solver-choice problem. The fix is the tolerance/early-stop (#1), not the solver. (`bank` is the exception —
   dominant eigenvalue ≈ +90 > ρ, mildly indefinite — but it's already sub-150 ms.)
   - **Pelillo/Bomze offsets — IMPLEMENTED, don't unlock speed** (`SHIFT=1`). Constant `γeeᵀ` (`shiftConst`,
     maximizer-preserving, enables monotone std): γ_auto≈91 is large vs the ~1–16 payoff spread → step ≈
     identity → still 500 iters, de-concentrates support (170→4600). The **adaptive** shift already in `std`
     (`s=−min π`, minimal valid) is faster and reached higher `f` than exp. Diagonal `κI` (`shiftDiag`, Bomze)
     is *not* a speed lever but a real **precision/recall knob**: `+κ` concentrates (state FP 27.3→21.7 at rec
     99.3→96.0, and faster), `−κ` spreads.
   - **Temperature / exp-unary — TESTED, COLLAPSES** (`EXPCH=1`). Sharper exponentiation (`eta`≥4, or `expU`
     = odds-ratio domain) converges in 2–20 iters — but to a **degenerate single vertex** (`|supp|=1`,
     winner-take-all, negative `f`) ⇒ boost K=0–1, useless. **The 500-iter crawl is the *price* of a
     distributed clique** (a part is ~150–500 comparable-weight bits); the flat near-fixed-point landscape
     *is* the clique. The signed log-domain `u` + signed edges are load-bearing — any sharpening breaks the
     balance. So the only safe speedup is the support-set early-stop (#1), never re-scaling the dynamics.
3. **Shared negPool inverted index — IMPLEMENTED + verified EXACT** `[DEPLOYED]`. Build `bit→negatives` once
   (107 ms, shared), then `maskedOv = |y∩⋁A| − |y∩D_τ|` via posting lists. **Byte-identical** (Neg+parts, 4M
   and 22M, `OPT=1`). Removes the ~100 ms per-word neg-scan → 15–40 % for small (Zipf-bulk) words, negligible
   for boost-dominated frequent words. But note per #1 the neg-scan is <5 % of a *frequent* word's cost.
4. **Parallelism** — the throughput win for the full vocabulary: featurize once, extract words independently →
   near-linear over cores (~10 CPU-hr → ~40 min on 16). Orthogonal to #1.
5. **Adaptive-core pipeline (§8 of spec, CV pending)** — fewer disjoint cores ⇒ fewer boost rounds + smaller
   affinity + ~3–4× smaller disk. Speed *and* storage.
6. **Active-set matvec — untried, exact-ish, secondary.** Support converges to 9–19 % of nodes; late iters have
   mostly dead edges. Restricting the matvec to live edges would help the tail — but after #1 caps the iters,
   the remaining headroom shrinks. Summation order changes (not bit-exact).
7. **`|A|` subsampling — a quality/speed dial, NOT free** `[MEASURED]`: capping train `|A|` to 4000 costs
   state −2.5pt held (ok) but time −9pt (87.3→78.2) — the extra data is real coverage diversity.
8. **Incremental affinity — MEASURED, REJECTED, REMOVED** `[MEASURED]`. Maintained `{W,mPos,jPos}` with delta
   updates: **lossy** (FP drift shifts PMI edges → state per-part Jaccard 0.245) **and** only 1.02–1.10× (the
   accumulation is <15 % of the round; the matvecs — #1 — are untouched). Code removed after rejection.

**Cross-word core compression — measured, little global headroom at this scale** `[MEASURED, XCLASS=1`, 8 words].
The user's question: can cores be split/shared across the dataset (store a recurring sub-block once, reference
it)? Measured: **within-word** parts overlap heavily (intra-Jaccard 28–55 %) — already captured by union
storage (#the 6.6–7.5× disk win). But **cross-word** cores are ~99 % disjoint (**cross-Jaccard 0.6–1.3 %**):
`bank`/`state`/`time`/… occupy near-disjoint regions of the 3F space, so a global codebook / sub-core
factorization buys almost nothing beyond per-word union storage. Note on the mechanism: an `m-of-n` gate does
NOT split into an α-sum of sub-gates in general (only a *pure conjunction* factors, and into an AND, not a
sum), so "split a core, keep the head weights" is only sound as **exact bit-list dictionary compression**
(reference a shared block), which needs cross-core bit reuse to pay — and there is ~none across unrelated
words. **Caveat (unmeasured):** at full-vocabulary packing (1 M words into 3F≈99,912 bits) cross-word bit reuse
is *forced* by capacity, and morphologically related words (`run`/`running`) should share more — so the
codebook could pay off at scale even though it doesn't for 8 hand-picked distinct nouns.

## 6.8 The resolved efficient algorithm (adaptive-core) — see the spec

The final architecture — extract dominant → **weight-first tight core** at equal recall → if strong-and-clean
ship a single core, else **augment weight-first to 55%, hard-peel disjoint, iterate** — with a
**concentration-derived dynamic FPMAX** (`min(plateau≈0.25, 0.015·r50)`), is specified in full in
`docs/phase1_spec.md` §8 (marked `[CANDIDATE]`, CV pending). Key measured results it rests on (single split):
augment (weight-first) beats trim (min-cover) on FP by building from discriminative bits (hydrogen 1% vs 31%
at 55% recall); the FP-aware strong-stop routes high-FP words to low-FP disjoint ensembles (hydrogen single
36% → ensemble 5%, bank 34% → 17%); min-cover exposes 21–90% wasted bits in a dominant core; the disjoint
core count is word-specific (1–6, a semantic property). Canon result: FP 4–25%, all ensembles disjoint
(overlap 0.00). **CV against the deployed multi-DS is the outstanding validation step.**

## 7. Verdict

- **The formulation works and is now a single deployed rule.** `fitClass` = keep common bits for every
  target + train-pure boosting + validation early-stop + val-tuned θ. CV-stable across bank/state/time; no
  per-target hand-tuning.
- **The mask's real job is neg-selection, not the affinity** — common bits self-cancel in `M`; a *mild*
  mask de-saturates the selection score, and the node universe stays unmasked (`keep`).
- **Recall is unary, precision leans on edges; overfit lives in the low-`m` tail** and is controlled by
  early-stop (which parts) + θ (operating point), not by banning bits.
- **Open, honestly:** held-out recall is modest (state/time ~77–82%, bank ~58–61%); the confFP floor
  (~7–15%) is inherent to boundary confusables and trades against recall via λ. `ρ=80` fixed (per-target
  search not yet needed under keep). Ablations T3/T4/T5 and cross-formulation T11 remain.

## 8. Next

- **T11** cross-formulation agreement: Jaccard of `stqp_boost` parts vs the gain-based `seq`/`parallel`
  parts — the strongest evidence the parts are real.
- Remaining ablations **T3** (precision basis), **T4** (soft m-of-n), **T5** (`a⁻:a⁺`).
- Phase 2: draw `p⁻` from other classes' `{p⁺_i}` and re-evaluate the full gate.
