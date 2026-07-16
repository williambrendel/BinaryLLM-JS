"use strict";

/**
 * @file nearestNeighbor.js
 * @brief Jaccard nearest-neighbour search over an {@link InvertedIndex}.
 *
 * Given a query fuzzy signature F, accumulates the number of shared parts with
 * each candidate word via the inverted postings, then returns the word with the
 * highest Jaccard `shared / (|F| + |F_cand| − shared)`. This is the sparse
 * `jaccard` computed incrementally so only words sharing ≥1 part are scored.
 */

/**
 * @function nearestNeighbor
 * @description Returns the index entry most similar to the query signature `F`.
 *
 * @param {Uint32Array} F - Query fuzzy signature (sorted part ids).
 * @param {import("./invertedIndex.js").InvertedIndex} index - The searched index.
 * @param {number} [excludeWi=-1] - A word id to skip (e.g. to find the nearest
 *   *other* word when measuring OOV separation).
 * @returns {{ wi: number, j: number }} Best word id and its Jaccard similarity,
 *   or `{ wi: -1, j: -1 }` if nothing shares a part. Ties break to the lower id
 *   for determinism.
 *
 * @example
 * nearestNeighbor(fuzzyEncode(dict, "runing"), index);
 * // → { wi: <id of "running">, j: 0.74 }
 */
export const nearestNeighbor = (F, index, excludeWi = -1) => {
  const shared = new Map();
  for (const id of F) {
    const post = index.postings.get(id);
    if (post) for (const wi of post) if (wi !== excludeWi) shared.set(wi, (shared.get(wi) || 0) + 1);
  }
  let best = -1, bestJ = -1;
  for (const [wi, c] of shared) {
    const j = c / (F.length + index.sizes[wi] - c);
    if (j > bestJ || (j === bestJ && wi < best)) { bestJ = j; best = wi; }
  }
  return { wi: best, j: bestJ };
};

export default nearestNeighbor;
