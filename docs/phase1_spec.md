# Phase-1 — Weak-Classifier Part Discovery via Inhomogeneous StQP + AdaBoost

**Status legend used throughout.** Every non-foundational claim is tagged:
`[DEPLOYED+CV]` shipped in `fitClass` and validated by 5-fold CV · `[CANDIDATE]` the adaptive-core
algorithm (§8), implemented as a benchmark mode and validated on a single structural split — **not yet CV'd
or wired into `fitClass`** · `[MEASURED]` a mechanistic finding from one or more runs · `[OPEN]` untested
lead. Do not read a `[CANDIDATE]`/`[MEASURED]` result as production-validated.

Implementation spec for the JS trainer (`src/core/phase1/*`, driver `benchmark/phase1.js`). Breaks the blunt
positive union `p⁺` into weak parts using Pelillo replicator dynamics on a signed bit-affinity graph with
unary potentials, then boosts them by **AdaBoost sample reweighting** (never peeling the positive set). No
C++, no RNN. Dynamics, the `−ρI` regularizer, and the positivity shift follow **Pavan & Pelillo, "Dominant
Sets and Hierarchical Clustering," ICCV 2003**. The `ρ > s₀−1` size heuristic (§8) is *motivated* by their
bounds but does **not** transfer verbatim to our signed `M` — a starting point confirmed empirically, not a
cited guarantee.

Signatures are the **3F canon** `[L | L1∪L2 | C]` (three positional bands, `benchmark/phase1.js` `feat()`):
band **F2** = adjacent word (dd=1), **F1** = near (dd 2–3), **F0** = far (dd ≥ 4); each band spans the full
`F`-dim part space, so `|3F| = 3·F`. On the reference dict `F = 33,304`, `3F = 99,912`.

Canonical target set for scale-checking: **`data/phase1_targets.txt`** — 11 words spanning `|A|` 85–4978
(common time/world/state/music; medium century/river/government; small bank; rare physics/philosophy/hydrogen).

---

## 0. Why this formulation — the weak-classifier motivation, and where it landed

**The premise.** AdaBoost calls a classifier "weak" if its error beats chance (`1 − 1/N_cls`); at
`N_cls ≈ 1.2M` words the bar is vacuous, so we redefine "weak" against the task: **an m-of-n part with
recall > 0.5 (weighted) and precision ≥ p_min on confusable negatives.** The blunt union `p⁺ = ⋁A` has 100%
recall but no precision (it's the OR of every sense's vocabulary). The move: break `p⁺` into pieces each
individually weak but cumulatively covering, combined by an α-weighted head.

**Where the premise landed (the arc of this investigation).** The design intent was *several distinct
sub-parts per class*. The reality is more nuanced and is the through-line of the whole spec:

1. **`[MEASURED]` With the naïve extraction (the full dominant set each round), AdaBoost does NOT decompose
   the class** — it re-finds one dominant bundle and nudges `m` + a few bits. Part overlap was severe
   (bank ~94% redundant; state/time nested bundles). The pipeline behaved as a **strong single detector per
   class**, not a decomposition (§7.1, §11).
2. **`[DEPLOYED+CV]` The fix that restored a real weak-learner ensemble: recall-based weak extraction with a
   0.6 recall floor.** Extracting each part as the *weight-first prefix grown to ~60% recall* (`recallTau`,
   §5.3) — instead of the full strong dominant set — gives flat, substantive α votes and genuinely more
   diverse parts (intra-Jaccard 0.80 → 0.28–0.43), while recovering held-out recall (§7). This is the
   shipped `fitClass` default.
3. **`[CANDIDATE]` The efficient, structured realization the investigation converged to: the adaptive-core
   algorithm (§8).** Extract dominant → weight-first tight core at equal recall → if strong-and-clean ship a
   single core, else accumulate **disjoint** weak cores via explicit **bit-peeling** until coverage suffices.
   Every knob is data-driven (tightness by recall, count by peel-to-target, single-vs-ensemble by recall
   **and** a concentration-derived FP threshold). It resolves the "no overlap, each core meaningful, small
   union" goals — but is validated only on a single structural split; CV productionization is the open next
   step.

So the class is best described as **one dominant bundle that is efficiently representable as a single
min-covered core, or (for genuinely diffuse/high-FP words) a small set of disjoint weak cores.** The
"orthogonal sub-parts" framing is not supported *from sample-reweighting alone*; disjoint structure requires
explicit bit-peeling, and how many disjoint cores exist is **word-specific (1–6), a semantic property** (§8, §11).

**Two deferrals (not omissions):** `p⁻` (the negative channel — other classes' `{p⁺}`) is **phase 2**;
sense isolation is a **later layer** (phase-1 parts deliberately bundle senses).

### 0.1 Bootstrapping
1. `p⁺ = ⋁A` (union of positive signatures) — 100% recall by construction.
2. **Negative set** = the `|A|` most confusable near-boundary negatives — single sorted pass by
   mild-masked content-overlap score, skip a `δ` band (§2).
3. **Strong bits** — unary log-odds `u` (§1.1).
4. **Two co-occurrence structures** — folded into the PMI-difference `M` (§1.2).
5. Extract parts (§5), accept on recall/precision bounds, reweight, repeat (§7). Deployed extraction is
   recall-based weak (§5.3); the adaptive-core algorithm (§8) is the candidate successor.

---

## 1. Objects

`M_glob(DD) = { bits with corpus-global frequency > DD }` (frequency-based common-word mask, computed
**once over the featurized signatures** — `bitCount/Ntot > DD`, target-independent; note this **is**
signature-bit-frequency, not an external word list `[MEASURED]`); `D_τ = M_glob ∧ (⋁A)`; `M_τ = ¬M_glob`;
`P⁺ = (⋁A) ∧ M_τ`. The mask is **mild** (`DD_sel ≈ 0.05`).

| symbol | definition |
|---|---|
| `A` | positive contexts; `y_i` = 3F canon signature of context `i` |
| `p⁺` | `⋁A`, the blunt union — 100% recall, the object broken up |
| `Neg` | **confusable** negatives (§2) — frozen for the whole phase |
| `w` | **AdaBoost weight** over `A`, init uniform, updated each round (§7). Positive set NEVER shrunk |
| `u` | **unary log-odds** over the node universe (§1.1) — `w`-weighted, rebuilt each round |
| `M` | **PMI-difference** affinity (§1.2) — signed, sparse edge map, defined directly (no `C⁺`/`C⁻`), `w`-weighted, rebuilt each round |
| `x*` | replicator equilibrium over the node universe; `wPart` = per-bit weights (bit importance) |

No `C⁺`/`C⁻` state, no residual `R`. `M` is the affinity directly; positives are reweighted, not peeled.

### 1.1 Unary potential — rate-smoothed log-odds
`u(b) = log( (p⁺(b)+α_r) / (p⁻(b)+α_r) )`, `p⁺(b)` = `w`-weighted positive rate (rebuilt each round),
`p⁻(b)` frozen. Rate-smoothing gives `p⁺=p⁻ ⇒ u=0` **exactly** — an equal-rate (common) bit self-zeroes on
the unary channel, no rule. `α_r` (default `1/|Neg|`) only bounds empty cells. The `−log p⁻` term is a
targeted IDF (down-weights confusable-common bits).

### 1.2 Pairwise potential — PMI difference (each side floored ≥ 0)
`M_ab = max(0, PMI⁺(a,b)) − max(0, PMI⁻(a,b))`, where `PMI⁺` uses `w`-weighted joints/marginals over `A`
(rebuilt each round) and `PMI⁻` is frozen. `M>0` net-attract, `<0` net-repel, `0` no edge. **Marginals
lifted out first**, then sets compared: two common bits that merely co-occur net ≈ 0; a pair bound in
τ-contexts but incidental in confusables nets strongly positive.

**Common-bit self-cancellation `[MEASURED]`.** For a *universal* common bit,
`PMI⁺(a,b) ≈ PMI⁻(a,b) ≈ log(1/f_a)` (the marginal divides out), so `M_ab ≈ 0` for every neighbour — **both
sides cancel.** With `u ≈ 0` and `−ρx`, the payoff `π ≈ −ρx < 0` drives it to zero: the replicator
self-eliminates universal common bits **without any ban** (empirically confirmed: unmasked-node discovery
reached state 99.2%). The channel-decoupling test (`commonCoefU`/`commonCoefM`) confirms recall lives in the
**unary** channel — killing common bits' edges is bit-identical to banning them (state/time), while killing
their unary reproduces the ban — but the edges still *scaffold precision* under early-stop, so keep both.

`u` and `M` are commensurable (log units); weights fixed `a⁺=a⁻=1` (and `|Neg|=|A|`).

---

## 2. The confusable negative set

`Neg` = the `|A|` most confusable near-boundary negatives, single sorted pass:
```
P⁺_sel = (⋁A) ∧ ¬M_glob(DD_sel)          # mild selection mask, DD_sel ≈ 0.05
score(j) = |P⁺_sel ∧ y_j| / |y_j ∧ M_τ|  # content-overlap fraction ∈ [0,1]; 0/0 → 0
sort candidates by score desc; skip while score ≥ 1 − δ  (δ = 0.05, always > 0); take next |A|
```
**Single pass** — no two-pass local re-mask. `|Neg| = |A| ⇒ p_min = 0.5`.

### 2.1 Selection mask — mild, both extremes fail `[MEASURED]`
The mask's load-bearing job is de-saturating this selection score, **not** protecting the affinity.
- **No mask** → score saturates at 1.0 → far-from-boundary negatives admitted → confFP explodes (bank 57.6%).
- **Aggressive mask** (DD small) → `|P⁺_sel|` collapses → held-recall craters (bank 7.2%), state/time
  `no_valid_m`.
- **Mild `DD_sel ≈ 0.05`** (masks ~12 near-universal bits) → de-saturates (`score_cut` 0.83–0.89), keeps
  discovery (98–99% union), keeps confFP 8–12%. **`δ=0.05` band is inert at the sweet spot** (δ=0 vs 0.05
  identical) — retained as a near-duplicate safeguard.
- **Two-pass local re-mask: CLOSED `[MEASURED]`** — re-running the mask mechanism locally over
  {pos ∪ Pass-1 neg} (`θ_loc`) has no useful regime (mild = masks nothing, contexts too diffuse; aggressive
  = collapses like an aggressive global mask). Off by default.

`M_glob` is corpus-global frequency, target-independent, computed once and ranked (a `DD` level is an O(1)
prefix of the sorted bit-list). **Not** a per-target metric (a per-target TF-IDF/unary mask deletes
pairwise-load-bearing bits).

### 2.2 Node mask — a held-recall regularizer; deployed default = keep `[DEPLOYED+CV]`
`maskNodes` (the affinity node universe) is **decoupled** from `maskSelect` (`negSet.js` opts). The node mask
does **not** protect against re-inflation (§1.2 shows common bits self-cancel); it is a regularizer with a
**target-dependent sign**:

| target | node mask | K | union | held-rec | confFP | commonSel |
|---|---|---|---|---|---|---|
| bank | ban / keep | 20/20 | 98.4/100 | 44.0 / **32.4** | 8.8/9.2 | 0 / **6/12** |
| state | ban / keep | 23/4 | 97.5/99.2 | 53.5 / **72.8** | 11.5/14.0 | 0 / 2/12 |
| time | ban / keep | 7/11 | 99.1/99.5 | 77.0 / 78.0 | 10.6/9.2 | 0 / 2/12 |

- **Universal common bit → self-cancels, never selected (state/time 2/12), acts as bundling glue**
  (state coarsens to K=4, held-recall 53.5 → 72.8%). Ban is redundant and mildly harmful.
- **Globally-common but locally-discriminative bit → selected (bank 6/12) → overfits** (train-union 100%,
  held 44 → 32%). Ban is a genuine regularizer here.

**5-fold CV verdict `[DEPLOYED+CV]`:** `ban` is fragile (state fails most folds, 18.1% ± 22.1); `keep` has
best recall but bank FP blows up (28.7% ± 17.9); **only `keep + validation early-stop` is strong and stable
on all three** (bank 58/9.3, state 77.7/15.3, time 81.5/9.3). So the deployed default is
**`maskNodes = ∅` (keep) + val early-stop** — the ban and the `|PMI⁺−PMI⁻|` admission rule (§9.1/D5) remain
untested alternatives. The per-target ban/keep *selector* was **retired** — bank's preference was pure
slice noise (flipped sign across splits).

### 2.3 Band-aware node mask `[MEASURED]`
Since the node universe is decoupled, a band-aware node mask was tested:
- **Prune F0-far only = clean win** — ~50% fewer nodes, ~23% fewer bits-used, held-rec −1 to −2pt
  (noise-level), FP ≤ baseline (the far band carries no class signal; §11 overlap analysis).
- **Prune F2-adjacent too = recall collapse** (bank 48.6 → 4.4%) — despite F2's flat intra≈cross *signature*
  overlap, the *specific* adjacent word is load-bearing (cores use 25–37% F2 bits).
- **Frequency-pruning common words ("the") from the nodes = recall collapse** (state 76 → 18%, time 81 →
  63%) — common-word bits are load-bearing *coverage* (the OR bundle's backbone). Frequency-masking can't
  tell *the*-as-noise from *the*-as-coverage. The correct way to exclude "the" from the discriminative core
  is by **x\* weight** (it self-cancels → low weight → excluded by a tight core), not by a frequency node-mask.

Deployed default: no band mask (keep all nodes). F0-far pruning is an available efficiency option.

---

## 3. The part objective — inhomogeneous StQP (Pavan–Pelillo regularized)
`maximize f(b) = uᵀb + ½ bᵀ(M − ρI) b`, `ρ ≥ 0` the coarseness knob. `uᵀb` sums selected bits' unary
log-odds; `½bᵀMb` the signed PMI affinity; `−ρI` the Pavan–Pelillo regularizer (their `α`, renamed `ρ` to
avoid the AdaBoost vote `α_k`). On the simplex `−ρ‖b‖²` is concave ⇒ **larger `ρ` ⇒ larger, coarser parts**;
small clusters vanish. `f` is maximized on `Δ` (§5), not unconstrained (that returns all of `P⁺`).

## 4. Replicator payoff — direct inhomogeneous gradient
`π(x) = u + Mx − ρx` (the linear `u` enters the payoff directly — no homogenization, no dense `Q`). Standard
multiplicative update needs `π > 0`: add a **scalar** shift `π + s·1` (`s = −min π + ε_s`, constant on `Δ`).
Exponential update needs no shift. `M` is a sparse edge map; `Mx` is one sparse matvec.

## 5. Replicator dynamics + support extraction

### 5.1 Solvers `[DEPLOYED]`
Three solvers on the same payoff: `std` (scalar positivity shift), `exp` (sign-robust, no shift), `hybrid`
(std where `min π > 0`, else exp). **Shipped: `exp`** — equal support (Jaccard 1.000, same `f`), 208 matvecs
vs 500 for std/hybrid (D2). `Mx` = `for (a,b,m) in edges: y[a]+=m·x[b]; y[b]+=m·x[a]`; active-set shrink as
`x` concentrates.

### 5.2 x\* shape — a descending slope, NOT a cliff `[MEASURED]`
The equilibrium `x*` is a **descending slope**: the top ~20–30 bits carry 10–16× uniform weight, then a long
low-weight tail (median 0.5× uniform) to ~0. There is **no natural cliff**; `tauSupp` cuts arbitrarily into
the tail, so the full-support `|Q|` (~200 bits) is **soft**. **Correction:** the previously-logged
`cliffGap = 1.0` was an **artifact** of the active-set prune (excluded bits are hard-zeroed, so the boundary
gap is always 1.0) — it did not measure the natural shape. Consequence: the "tight core" and the "tail" are
two ends of one slope (§5.4).

### 5.3 Support extraction — recall-based weak (deployed) vs generic (retired) `[DEPLOYED+CV]`
- **Generic `tauSupp`** = take the full dominant set (~200-bit support). This is a **strong** classifier
  (66–80% recall) → AdaBoost merely re-finds it → redundant parts. **Retired** as the extraction.
- **Recall-based weak (`recallTau`, deployed)** = grow the support from the top-`x*` bits until *weighted*
  recall clears a floor. This is a genuine weak learner, it **adapts tightness per round to concentration**
  (concentrated word → few bits; diffuse → more), and its weighted-recall stop **tracks the reweighting** so
  successive parts escape the dominant clique (union spans 4–5× the round-1 core). This is the correct
  adaptive-tightness formula; **`n_eff = 1/Σx*²` (inverse participation ratio) was tested and rejected** — it
  measures weight *concentration*, not recall coverage, so it lands below 50% recall for diffuse words.
  - **`tauFloor` = 0.6, not 0.5 `[DEPLOYED+CV]`.** At floor 0.5 each part sits right at the weak-learning
    edge, so `ε ≈ 0.5` and the AdaBoost vote `α = ½log((1−ε)/ε) ≈ 0` — the head collapses to a pile of
    near-zero votes. Raising the floor to **0.6** makes α substantive and **flat** (cv 1.16 → 0.17) and
    recovers held-recall (time 57.9 → 80.8%, state 69.5 → 76%). Confirms the α-flat↔FP link: in the flat-α
    regime confFP is stable; it only moves crossing out of the α-collapse.

### 5.4 Tight core vs tail `[MEASURED]`
Split the dominant core by `x*·|Q| ≥ 2` into **tight** (~8–65 high-weight bits) and **tail** (the rest):
- **Tight core = precision anchor** — FP 0–3% everywhere, but narrow recall (19–23% on diffuse words, high
  only on concentrated ones like century 88%).
- **Tail = coverage engine AND FP source** — adds the bulk of recall for diffuse words (+53 to +64 pts) and
  carries **essentially all the FP** (`full_fp ≈ tail_fp` on every word). Key: **x\* weight measures clique
  *centrality*, not *coverage*** — tail bits are peripheral (low weight) yet high-coverage (46–85% alone),
  because they fire on positives *and* negatives.
- The tail's value is **per-word** (recall:FP ratio 0.6–5.9): cheap recall for state (5.9)/music/world,
  net-negative for century (0.6). This is the mechanistic root of the recall/FP/tightness trade and why the
  right tightness is per-word (§8).

Two distinct "tighten" operations, do not conflate: **min-cover** (greedy fewest bits for a *recall* — keeps
recall + FP, the efficiency measure) vs **weight-first/tight** (high x\* bits — drops FP + recall, the
precision measure). **Min-cover is right for *measuring* waste; weight-first is right for *building* cores**
(§8).

## 6. The m-of-n gate (precision fit)
`m* = argmax_m precision(Q,m) s.t. recall_w(Q,m) ≥ 0.5`, `recall_w` AdaBoost-weighted over `A`.
**Precision is base-rate-correct:** `tp = recall_w·|A|` (count scale, commensurate with the raw confusable
`fp` count) — mixing a rate `tp` with a count `fp` rejects nearly every good part. `recallAt1` (m=1 support
ceiling) < 0.5 ⇒ `ρ`-too-small (`no_valid_m`), not an m-fit failure.

**recall(m) curve `[MEASURED]`:** collapses fast for state/time (m=1 forced → loose OR-bundle), gentler for
bank (m*=2). `m*` sits on the high-recall side, before any cliff. The steep recall(m) *is* the "one loose
bundle" structure numerically.

---

## 7. The AdaBoost reweighting loop
No peeling, no residual (positive set always all of `A`; `w` updated, statistics `w`-weighted and rebuilt).
Accept iff `recall_w ≥ 0.5` **and** `precision ≥ p_min`; `ε = clip(1−recall_w, ε_min, 1−ε_min)`,
`α_k = ½log((1−ε)/ε)`; down-weight covered positives (`w_i ← w_i·exp(−α_k h_k)`), renormalize; stop on true
(unweighted) union recall > 0.99 or the dual bound.

### 7.1 Coverage argument (not the AdaBoost error theorem) `[MEASURED]`
`ε_k = 1 − recall_{w,k} ≤ 0.5` is one-sided (positives only; negatives frozen, gate admissibility only).
Coverage accretes toward 1 by a set-cover argument — **not** the two-class training-error bound (whose
preconditions don't hold: positives-only reweighting, frozen negatives, `ε` = coverage rate). With the
generic extraction this accretion was via **threshold/`m` variation on one bundle** (§0). With the
recall-based weak extraction (§5.3) it is genuinely more diverse.

### 7.2 The deployed head — α-weighted sum + validation-tuned θ `[DEPLOYED+CV]`
Head `S_τ(x) = Σ_k α_k h_k(x)`, `h_k = 1[|Q_k ∧ x| ≥ m_k]`, `fires = S_τ > θ`. `α_k` closed-form (with the
ε-floor — a perfect-recall part gives `α=∞` and a dead head; clip `ε` to `[1e-10, 1−1e-10]`).
- **θ is validation-tuned, not `½Σα`.** `θ* = argmin_θ [λ·FP_val(θ) − recall_val(θ)]` over the val ROC.
  - `λ = 1` is a **strict win** over the ½Σα heuristic — same recall, roughly half the variance
    (bank ±8.9 → ±2.3). It buys stability (free), not precision. Once α is flat (floor ≥ 0.55) the operating
    point is stable; on the *full* discovered set `θ`-tune makes prefix early-stop nearly redundant (θ absorbs
    the extra loose parts) — but keeping the parts + θ-tune is marginally lower-FP and tighter.
  - `λ > 1` is the precision knob (a genuine trade): λ=2 trims ~2pts FP for ~2–3pts recall; λ=3 destabilizes state.
  - **Adaptive `λ` for small classes `[DEPLOYED+CV]`:** `λ = clamp(1500/|A|, 1, 3)` (bank λ=3, state/time λ=1).
    Small classes over-fire (few positives → noisy val θ); the auto penalty fixed and *stabilized* bank FP
    (18.9% ± 10 → 7.7% ± 3.1), at a held-recall cost (65 → 49%) — the inherent small-class trade.
- **Why θ (not the part set) is the FP lever (FPDIAG) `[MEASURED]`:** `m≥2` parts are clean (1–2% FP);
  `m=1` parts drive FP (10–22%); FPs are **redundant** across parts (dropping any single part barely moves
  head FP). So the part set can't be pruned to fix FP — only θ sees the whole ensemble. `m=1` can't be
  forbidden (forced by the recall floor on heterogeneous classes). The FP floor is a genuine recall/FP trade
  (confusables are boundary-hard by construction, `p_min=0.5`) — not eliminable, but honestly val-tunable.

### 7.3 `fitClass` — the deployed pipeline `[DEPLOYED+CV]`
`fitClass(A, negPool, Mglob, opts)` (`src/core/phase1/fit.js`), one call, the operational default:
```
fitClass:
    (A_tr, A_val) = split(A, valFrac=0.25)
    Neg = buildNegSet(A_tr, negPool, Mglob, maskSelect=DD_sel(0.05), maskNodes=∅ /*KEEP*/, δ=0.05)
    G   = boost(A_tr, Neg, ρ=80, solver=exp, recallTau=true, tauFloor=0.6)   # train-pure
    r*  = val prefix maximizing (recall_val − fpWeight·confFP_val)           # early-stop
    θ   = val θ-tune on G[:r*], fpWeight = clamp(1500/|A|,1,3)               # adaptive λ
    return { head: makeHead(G[:r*], θ), G, r*, neg }
```
Defaults: `valFrac 0.25, rho 80, solver exp, delta 0.05, recallTau true, tauFloor 0.6, maskNodes ∅,
fpWeight auto, tuneTheta true, earlyStop true`. Test: `__tests__/core/phase1/fit.test.js` (4 green; the smoke
recall assertion is relaxed to > 0.55 because the deployed config is a weak-learner ensemble at a val-tuned
operating point — moderate recall by design on toy data).

**End-to-end deployed stats (5-fold CV) `[DEPLOYED+CV]`:**

| stat | bank | state | time |
|---|---|---|---|
| parts discovered / kept | 8.8 / ~5 | 15.2 / 14.6 | 21.4 / 20.8 |
| \|Q\| median (bits) | ~100 | 277 | 286 |
| α mean, cv (flat?) | 0.26, ~0.4 | 0.23, **0.23** | 0.22, **0.17** |
| intra-class Jaccard | 0.43 | 0.34 | 0.28 |
| cross-class Jaccard | ~0.01 (disjoint) | ~0.01 | ~0.01 |
| retention \|Q\|/⋁A · union/⋁A | 9% · 14% | 8% · 18% | 6% · 15% |
| train / held-rec · confFP | ~81 / ~49–65 · 8–19% | 94 / 76 · 12.6% | 93 / 81 · 8.9% |

(Bank's held-rec/FP depend on the adaptive-λ operating point: λ=1 gives 65/19; λ=3 gives 49/7.7. State/time
are λ=1.) The parts are **big loose bundles** (|Q| med ~100–290, low `m` mostly 1–4), few per class, and
under `tauFloor=0.6` more diverse than the generic extraction (Jaccard 0.80 → 0.43 on bank) — but the whole
train/held gap and the FP floor persist (§11).

---

## 8. `[CANDIDATE]` The adaptive-core algorithm — efficient, disjoint, data-driven

This is the design the efficiency arc converged to. **Implemented as `ADAPTCORE=1` in the driver and
validated on a single structural split; not yet CV'd or wired into `fitClass`.** It supersedes §7's
multi-DS ensemble *if* CV confirms out-of-sample generalization.

### 8.1 The algorithm
```
each round (peeled bits excluded from the graph; uncovered positives up-weighted):
  1. dominant = replicate(π = u + Mx − ρx)                       # §4–5
  2. tight core = weight-first bits until recall = recall(dominant)   # waste-removed, low-FP (§5.4)
  3. concentration r50 = |weight-first prefix reaching 50% recall|    # cheap, read during growth
     dynFPMAX = min(0.25, 0.015 · r50)                                # predicted ensemble FP
  4. if recall(tight) ≥ STRONG (0.7) AND fp(tight) ≤ dynFPMAX:
        SHIP the single core; stop.
     else:
        weak core = weight-first augment to VIABLE (0.55) WEIGHTED recall   # buffer, low-FP
        accept; HARD-PEEL its bits; iterate.
  stop when union recall ≥ TR (0.85) or no core clears the floor.
```

### 8.2 Why each piece (all `[MEASURED]` unless noted)
- **Weight-first (augment), not min-cover (trim), for construction.** At ~55% recall, weight-first
  (discriminative-first) has **much lower FP** than min-cover (coverage-greedy grabs high-coverage =
  FP-heavy bits): hydrogen **1% vs 31%**, physics 5% vs 16%, bank 9% vs 16% — 8/10 words lower. Min-cover is
  for *measuring* waste (it exposes 21–90% redundant bits in a dominant core), not for *building*.
- **Recall constraint 55% (buffer over 50%).** Below ~0.55 the α vote collapses (§5.3).
- **Explicit HARD bit-peeling enforces disjointness** (`buildAffinity` `exclude`, `excludeCoef=0`: peeled
  bits get `u→−∞`, `M→0`) → inter-core overlap **0.00**. **This is the disjointness mechanism** — AdaBoost
  reweighting **alone does NOT** enforce it (the clique re-emerges, inter-core Jaccard rises to 0.7–0.84 as
  cores tighten); soft peel (scale by `peelCoef`) re-introduces overlap. Reweighting only steers *which*
  uncovered positives the next disjoint core targets.
- **Strong-vs-weak on recall AND FP.** Recall-only routing left high-FP single cores (bank 34%, hydrogen
  36%). The FP-aware stop routes those to disjoint weak ensembles that slash FP — hydrogen **36 → 5%**, bank
  **34 → 17%**, physics **26 → 11%** — while keeping/raising recall.
- **Dynamic FPMAX from concentration** = `min(FPCAP, FPSLOPE·r50)` = `min(0.25, 0.015·r50)`, a **prediction
  of the ensemble FP** (linear-rise-then-plateau). Concentrated words (small `r50`) get a *low* threshold →
  routed to their tiny low-FP ensemble; diffuse words hit the cap → kept single unless genuinely FP-heavy.
  It fixes the century-vs-time misroute a fixed FPMAX can't: both have 12% single-core FP, but century
  (r50=4 → dynFPMAX 6%) → ensemble @4%, time (r50=11 → dynFPMAX 16%) → single @12%. `r50` is the recall-
  coverage prefix size (= the recall-based tightness of §5.3), read free during growth.
  - **The cap models the ensemble-FP *saturation*, and is essential — not arbitrary.** Ensemble FP rises
    with diffuseness then **plateaus (~18–25%)** — a disjoint ensemble can't do better than ~25% FP on a
    diffuse word (government stays 25% even as an ensemble). The linear `FPSLOPE·r50` is only valid in the
    concentrated regime; at high `r50` it over-predicts wildly (r50=104 → 156%), so `FPCAP` clips it to the
    real plateau. **Uncapping breaks routing** `[MEASURED]`: with `FPCAP=1`, the runaway threshold (bank 72%,
    government 100%) lets high-FP single cores pass the strong-stop, reverting bank ensemble-17% → single-34%,
    hydrogen ensemble-**5% → single-35%**. So the `min(plateau, slope·r50)` *shape* is correct; only the two
    constants (0.25, 0.015) are **canon-fit** and pending CV re-estimation. The robust alternative that needs
    neither constant is **compare-both**: build the ensemble and ship whichever (single vs ensemble) has lower
    FP — it *measures* the ensemble FP (so it sees the saturation directly) instead of predicting it.

### 8.3 Canon result (single split, STRONG=0.7, VIABLE=0.55, TR=0.85, dynamic FPMAX) `[CANDIDATE]`
- **Single strong core:** time (81/12), state (72/10), world (83/18).
- **Disjoint ensemble (overlap 0.00, union escapes the dominant core):** century (2 cores, **28 bits,
  93%, 4% FP**), music (K=3, 92%, 18%), river (K=2, 87%, 18%), government (K=3, 94%, 25%), bank (K=3, 92%,
  17%), physics (K=3, 93%, 11%), hydrogen (K=3, 93%, 5%).
- **FP now 4–25% across the canon** (was ≤36% with recall-only single cores). Stubborn: government 25%
  (intrinsically diffuse). All ensembles disjoint; multi-core unions cover more than the dominant core
  (music 54 → 92%, river 70 → 87%).

### 8.4 How many disjoint cores exist — word-specific, semantic `[MEASURED]`
Peel-to-exhaustion (loose dominant set + hard peel, each ≥50%): **1** for monolithic words (time, world;
and data-poor rare physics/philosophy/hydrogen), **2** for state/government/bank, **3–4** for polysemous
mid-frequency content words (river/century 3, music 4). It is the number of distinct discriminative
sub-structures — a **semantic** property, **not** frequency-monotonic. Design: **don't fix a count — peel
until no core clears the floor.**

**Disjoint ≥50% cores AND union ≈ one core are mostly incompatible.** Partitioning the dominant core (union
held = core) into disjoint ≥50% pieces: **1** for data-sufficient words (the core is indivisible at 50% —
splitting a half drops below 50%), 2 for century/government/bank, 5–6 for rare words (overfit artifact).
So multiplicity requires the union to grow beyond one core (peel *adds* bits); for real words the dominant
core is the atomic unit.

### 8.5 Efficiency vs the multi-DS ensemble `[MEASURED]`
The min-covered single dominant core is far more bit-efficient than the multiple cores: after min-cover the
dominant core (29–169 bits) is **3–8× smaller** than the union of the (even minimized) peel cores. Core
efficiency **scales with frequency** — for common words a single core matches multi-DS recall with 3–4×
fewer bits (time 81.2% @ 178 bits vs 80.8% @ 730); it reverses for rare words (their small-sample cores are
noisy/large). The multi-DS occupies the high-recall/low-FP corner only by spending 4× the bits.

### 8.6 Capacity / packing limit (noted) `[MEASURED]`
`3F = 99,912`. A ~200–400-bit core is 0.2–0.4% of the space, so only **~250–500 disjoint cores** fit; beyond
that, overlap is forced by pigeonhole. The measured cross-class part overlap of ~0 holds only at 3 classes;
at vocabulary scale (33k words ≫ 500 slots) **cross-class overlap is structural**. The discriminative core
is ~25 bits (not the full ~300-bit support), so tight cores pack ~10× denser (~4,000) — still ≪ 33k. ⇒ at
scale the representation **must discriminate combinatorially (m-of-n over shared bits)**, which is what the
gate is for. Argues for tight cores.

---

## 9. Parameters

| param | meaning | default |
|---|---|---|
| `DD_sel` | selection mask (§2.1), swept `{0.005,0.01,0.05,0.1}` (D3) | **0.05** |
| `DD_nodes` / `maskNodes` | node mask (§2.2), keep vs ban vs `\|PMI⁺−PMI⁻\|` (D5) | **∅ (keep)** + val early-stop |
| `δ` | near-duplicate band (§2) | **0.05** (fixed, always > 0; inert at the sweet spot) |
| `ρ` | Pavan–Pelillo `−ρI` coarseness (§3, §8-bounds); `ρ≈100→\|Q\|≈219` | **80** |
| `recallTau` / `tauFloor` | recall-based weak extraction (§5.3) | **true / 0.6** |
| `α_r` | rate-smoothing (§1.1) | `1/\|Neg\|` |
| `ε_min` | AdaBoost ε-clip (§7.2) | `1e-10` |
| `λ` / `fpWeight` | val θ-tune FP weight (§7.2) | **adaptive `clamp(1500/\|A\|,1,3)`** |
| `θ` | ensemble threshold | **val-tuned** at `λ` |
| `valFrac` | train/val split (§7.3) | **0.25** |
| solver | std/exp/hybrid (§5.1) | **exp** (D2) |
| `p_min` | precision floor (§8) | `\|A\|/(\|A\|+\|Neg\|)` = **0.5** |
| **`[CANDIDATE]` §8:** `STRONG` / `VIABLE` / `TR` | strong recall / weak recall floor / union target | 0.7 / 0.55 / 0.85 |
| **`[CANDIDATE]`** `FPCAP` / `FPSLOPE` | dynamic FPMAX = `min(FPCAP, FPSLOPE·r50)` — `FPCAP` = ensemble-FP **plateau** (essential, models saturation; do **not** uncap — reverts high-FP words to high-FP singles, §8.2); `FPSLOPE` = concentrated-regime rise. Both canon-fit, CV-pending | 0.25 / 0.015 |
| **`[CANDIDATE]`** `peel` / `peelCoef` | hard bit-peel between cores (§8.2) | on / 0 (hard) |
| `coreT`, `commonCoefU/M`, `scaffoldOnly` | diagnostic knobs (channel/tight ablations) | off |

### 9.1 Decisions checklist — status
| # | decision | resolved value | evidence |
|---|---|---|---|
| D1 | `M` data structure | **sparse edge map** | density 1.3–2.3% post-floor; nnz 25.4k/154k `[MEASURED]` |
| D2 | replicator solver | **exp** | 208 matvecs vs 500; support Jaccard 1.000 `[MEASURED]` |
| D3 | selection mask `DD_sel` | **0.05** | both extremes fail; mild de-saturates + completes `[MEASURED]` |
| D4 | union recall 0.99? | **yes** | reached on bank/state/time; boosting curve `[MEASURED]` |
| D5 | node mask | **keep ∅ + val early-stop** | ban fragile (state 18±22), keep FP blows up (bank 28.7±17.9), keep+es stable; `\|PMI⁺−PMI⁻\|` rule OPEN `[DEPLOYED+CV]` |
| D6 | part diversity | **one bundle → efficient single/few disjoint cores** | generic: one bundle (bank ~94% redundant); recallTau@0.6: Jaccard 0.43/0.34/0.28; disjoint requires hard peel; count 1–6 word-specific `[MEASURED]` |
| D7 | extraction | **recall-based weak, tauFloor 0.6** | generic = strong single detector; recallTau@0.6 = flat α, diverse, recall recovered `[DEPLOYED+CV]` |
| D8 | head operating point | **val θ-tune, adaptive λ** | λ=1 strict win over ½Σα; adaptive λ fixes small-class FP `[DEPLOYED+CV]` |
| D9 | efficient extraction (candidate) | **adaptive-core (§8): weight-first tight core, strong/weak on recall+dynFP, hard-peel disjoint** | single-split canon: FP 4–25%, disjoint, escapes dominant — **CV pending** `[CANDIDATE]` |

---

## 10. Complexity
Reweighting (not peeling) ⇒ every positive-side statistic is `w`-weighted and **rebuilt each round** — no
downdate, no amortization claim (that was a peeling artifact). Per round: build `u`,`M` = `O(|A|·density²)`;
replicator `O(nnz(M)+n)` per step (one sparse matvec, active-set-shrunk). `Neg`, `p⁻`, `P⁻` frozen once.
Total `O(T·|A|·density²)`, `T ≈ tens`, density 1–3%.

## 11. Evaluation hooks + established findings
Driver modes (`benchmark/phase1.js`): `CV` (5-fold ban/keep/keep+es), `E2E` (deployed stats), `DIST`
(α/|Q| shape), `OVERLAP` (signature intra/cross per band), `XCLASS` (cross-class part overlap), `BANDMASK`,
`BANDS`, `COREWASTE`, `COREDECOMP`, `TAIL`, `TIGHTCORE`, `TIGHTBOOST`, `DISJTIGHT`, `COREPART`, `NEFF`,
`COMPARE50`, `ADAPTCORE`, `TFCV`/`FPCV`/`TRIMCV`/`CORECV`/`RTCV`, `SOFTPEEL`, `XSTAR`.

**Class separability (`OVERLAP`) `[MEASURED]`:** signature overlap is low/sparse; discrimination lives in
the **near band F1 (dd 2–3)** (intra bank 6.2/time 5.3 ≫ cross 3.1); **F2-adjacent doesn't separate** (intra
≈ cross ≈ 6–9, shared function words) yet is load-bearing (§2.3); **F0-far is noise** (intra = cross ≈ 1).
Discovery shifts the core toward F2+F1 vs the F0-dominated raw `⋁A`.

**Per-part FP (FPDIAG), part-overlap, tail, min-cover, capacity** — all summarized inline (§5.4, §7.2, §8).
The generic-extraction part-overlap table (bank Jaccard 0.80 / union÷Σ|Q| 0.06 ~94% redundant; state/time
nested) is the historical "one-bundle" evidence; `tauFloor=0.6` reduced it to 0.28–0.43.

**Status of coverage / FP / held-recall.** Coverage is **solved** (train-union 98–100%). Held-out recall is
modest and data-hungry — the train/held gap shrinks with `|A|` (bank ~29pt @500, time ~10pt @5000). The FP
floor is a **frontier**, not a wall: confusables are boundary-hard by construction, the FP is `m=1`-driven
and redundant, and val-tuned θ (with λ) is the honest picker; the adaptive-core algorithm (§8) lowers it
further where the class decomposes.

## 12. Audit / do-nots (delta from the retracted claims)
- **Do NOT read `cliffGap ≈ 1.0` as a real cliff** — it's a prune artifact; x\* is a slope (§5.2).
- **Do NOT ban common bits "to stop re-inflation"** — PMI-difference cancels universal common bits; the node
  ban is a regularizer with a target-dependent sign; deployed default is **keep + val early-stop** (§2.2).
- **Do NOT frequency-prune common words from the nodes** — they're load-bearing coverage; exclude "the" via
  x\* weight (tight core), not frequency (§2.3).
- **Do NOT use min-cover to *build* a shipped core** — it grabs FP-heavy coverage bits; use weight-first
  (augment). Min-cover is only for *measuring* waste (§8.2).
- **Do NOT rely on AdaBoost reweighting for disjointness** — it re-finds the clique; disjointness needs
  **explicit hard bit-peeling** (§8.2). Soft peel re-adds overlap.
- **Do NOT extract the full strong dominant set as the weak learner** — use recall-based weak (`recallTau`),
  and floor at **0.6** not 0.5 (else α collapses to 0) (§5.3).
- **Do NOT use `n_eff` as the tightness formula** — it tracks concentration, not recall (§5.3).
- **Do NOT fix the core count or a single FPMAX** — count = peel-to-target (word-specific 1–6); FPMAX =
  dynamic from concentration (§8).
- **Do NOT uncap the dynamic FPMAX** — the cap `FPCAP` is the ensemble-FP *plateau* (~25%), not a fudge;
  uncapping lets diffuse words' thresholds run away (bank 72%, government 100%) so their high-FP single cores
  pass the strong-stop, reverting hydrogen ensemble-5% → single-35%, bank 17% → 34% (§8.2). If you want to
  drop the constants, use **compare-both** (measure the ensemble FP, ship the lower), not uncapping.
- **Retained do-nots** (still valid): `ε = 1 − weighted recall` one-sided; `α_k` from AdaBoost vote not the
  StQP `f`; ship the α-SUM not the OR; no residual/downdate; `M` direct PMI-diff (not count ratio, no
  `C⁺/C⁻`); rate-smoothing (not count); direct payoff `π=u+Mx−ρx` (no `Q`, no homogenization); scalar
  positivity shift; sparse matvec only; `−ρI` (not `+cI`); mild selection mask (both extremes fail); base-
  rate-correct precision (`tp = recall_w·|A|`); ε-floor; raise `ρ` once if `no_valid_m`; 3F canon (not 2F);
  parts bundle senses (not per-sense).

## 13. Files
- `phase1/fit.js` — **deployed entry** `fitClass` (§7.3): split → keep-Neg → recall-based weak boost →
  val early-stop → adaptive-λ val θ-tune. Opts `recallTau/tauFloor/maskNodes/fpWeight/valFrac/...`.
- `phase1/negSet.js` — `Neg` via mild-masked single sorted pass + δ-band; decoupled `maskSelect`/`maskNodes`;
  frozen `p⁻`,`P⁻`. Two-pass path retained (off) as closed evidence.
- `phase1/affinity.js` — `w`-weighted `u`, `M` (PMI-diff sparse edge map), rebuilt per round; per-channel
  `commonCoefU/M`; `exclude`/`excludeCoef` for hard/soft peel.
- `phase1/replicator.js` — direct payoff `π=u+Mx−ρx`; std/exp/hybrid; support + `cliffGap` (artifact, §5.2).
- `phase1/mfit.js` — m-of-n fit, base-rate-correct precision at recall floor 0.5.
- `phase1/boost.js` — reweight loop; dual-bound; ε-clip; opts `recallTau/tauFloor/peel/peelCoef/coreT/
  commonCoef*/scaffoldOnly`.
- `phase1/head.js` — α-sum head `S_τ=Σα_k h_k > θ`; max-pool readout for A/B.
- `benchmark/phase1.js` — the driver and all evaluation modes (§11); the `[CANDIDATE]` adaptive-core lives
  here as `ADAPTCORE=1`.
- `__tests__/core/phase1/fit.test.js` — 4 tests, green. `data/phase1_targets.txt` — the canon.
- Report: `docs/phase1_report.md` (measured tables, reproducible via the driver flags).

## 14. Run metadata
Corpus wiki_4m, dict english.txt (`F=33,304`, `3F=99,912`), targets = the canon (`data/phase1_targets.txt`),
`|A|` 85–4978. Deployed config: `recallTau tauFloor=0.6, maskNodes=∅ keep, val early-stop, val θ-tune,
adaptive λ, ρ=80, exp solver, DD_sel=0.05, δ=0.05`. Adaptive-core (§8) is the CV-pending candidate successor.
