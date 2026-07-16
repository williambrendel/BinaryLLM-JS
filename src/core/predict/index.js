"use strict";

/**
 * @file index.js
 * @brief Barrel for the prediction / learning stage (pool-discovery predictor).
 *
 * The evaluation guardrail everything is measured against: routed models produce
 * label-count leaves, {@link module:core/predict/readout} smooths them, and
 * {@link module:core/predict/metrics} scores held-out log-loss + the calibration
 * gate. Baselines (unigram / context-gram) are the bars a learned tree must beat.
 * @see spec/pool-discovery.tex §4–§5
 */

export { leafProb, leafLogProb, flooredLogProb, leafArgmax } from "./readout.js";
export { evaluate, calibrationGate } from "./metrics.js";
export { buildUnigram, buildContextGram } from "./baselines.js";
export { buildCartTree } from "./tree.js";
export { buildPoolTree } from "./poolTree.js";
export { growPart } from "./growPart.js";
export { evaluatePath, decodePath, makePathContext } from "./pathDecode.js";
