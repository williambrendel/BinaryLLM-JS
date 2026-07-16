"use strict";

/**
 * @file index.js
 * @brief Barrel for the word-encoding stage.
 * @see {@link module:core/parts/encoding/bpeEncode}   structural [L,C,R] peel
 * @see {@link module:core/parts/encoding/fuzzyEncode}  containment signature F
 */

export { bpeEncode } from "./bpeEncode.js";
export { fuzzyEncode } from "./fuzzyEncode.js";
export { addBigramBackstop } from "./bigramBackstop.js";
