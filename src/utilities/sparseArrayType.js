"use strict";

// ============================================================================
// utilities/sparseArrayType.js
//
// Pick the narrowest unsigned TypedArray element type able to hold sparse ids
// for a space of `dim` distinct positions.
//
// Sparse sets store ids in [0, dim), so the largest storable value is dim-1.
// Uint16Array holds 0..65535, so it suffices iff dim <= 65536; beyond that,
// widen to Uint32Array. Used to size signature bands (dim = F) and their flat
// views (dim = 2F or 3F) without over-allocating on small dictionaries.
// ============================================================================

/**
 * @param {number} dim number of distinct ids in the space (largest id is dim-1).
 * @returns {Uint16ArrayConstructor|Uint32ArrayConstructor} Uint16Array if
 *   dim <= 65536, otherwise Uint32Array.
 */
export const sparseArrayType = dim => dim <= 0x10000 && Uint16Array || Uint32Array;

/**
 * @ignore
 */
export default sparseArrayType;
