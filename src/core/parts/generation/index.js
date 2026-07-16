"use strict";

/**
 * @file index.js
 * @brief Barrel for the part-generation stage.
 * @see {@link module:core/parts/generation/generateParts} the orchestrator
 */

export { generateParts } from "./generateParts.js";
export { prefilterWholes } from "./prefilterWholes.js";
export { generativeAffixes } from "./generativeAffixes.js";
export { discriminativeTree } from "./discriminativeTree.js";
