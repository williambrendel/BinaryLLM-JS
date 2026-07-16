# Normalization stage

Resolves a noisy or out-of-vocabulary word to a canonical known word before it
is structurally encoded, so the peel ([encoding stage](../encoding/README.md))
never has to be typo-robust — robustness lives here, in fuzzy retrieval.

| Function | Purpose |
|---|---|
| `buildInvertedIndex(words, signatures)` | Index the vocabulary's fuzzy signatures F (part id → word ids). |
| `nearestNeighbor(F, index, excludeWi?)` | Jaccard NN over the index (incremental shared-count accumulation). |
| `canonicalize(word, dict, index, {theta})` | Encode F → NN → confidence gate → canonical word (or keep original). |

## Pipeline

```
word ──fuzzyEncode──▶ F ──nearestNeighbor──▶ {wi, j} ──gate(θ)──▶ canonical word
```

## The confidence gate

Jaccard **cannot** distinguish "a typo of X" from "a genuinely new word similar
to X" — the two distributions overlap. So the gate is *not* a typo detector; it
means **"snap to the nearest known word if close enough (`J ≥ θ`), else keep the
original."** The protection for valid rare words is that they are *in the index*
and self-match at `J = 1` (always above θ) — the gate only governs **unseen**
inputs. Measured behaviour (wiki, θ ≈ 0.5):

- **self-match ≈ 99.7 %** — known words return themselves;
- **typo-recovery ≈ 85 %** — typos resolve to the original (misses land on a close sibling);
- junk / far-OOV (e.g. `qwerty`) lands at `J ≈ 0.4` and is kept.

See `scripts/` for the benchmark that produces these numbers.
