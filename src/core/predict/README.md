# `core/predict/` — next-token prediction (pool-discovery predictor)

Turns discovered parts into an actual **language-model readout**: predict the next token from a context
signature, and measure it against unigram/n-gram baselines with held-out log-loss.

## Motivation

Part discovery is only useful if it predicts. This stage grows parts into a **CART tree** — each split is a
coverage-fired part (`|x∩p|/|p| ≥ t`) — so a test context follows a root→leaf path and reads a smoothed
next-token distribution off its leaf. Storing **counts** (never probabilities) and interpolating toward the
unigram floor keeps the readout calibrated and honestly comparable to the baselines it must beat.

## Contents

| file | role |
|---|---|
| `tree.js` | capped, best-first CART tree whose splits are grown parts |
| `growPart.js` | grow one coverage-fired part for a node split (the `best_split` primitive) |
| `poolTree.js` | second-pass readout: one tree over a **fixed pool** of parts |
| `readout.js` | smoothed leaf / unigram distributions + the interpolation floor (counts → probabilities) |
| `pathDecode.js` | the readout over a test signature's root→leaf path |
| `baselines.js` | context-free (unigram) and single-context (n-gram) baselines — the bar to beat |
| `metrics.js` | held-out log-loss / accuracy + the calibration gate |
| `index.js` | barrel export for the prediction stage |
