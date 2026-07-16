"use strict";

/**
 * @file fitWorker.js
 * @brief Worker entry for {@link parallelFit}. Reconstructs the shared, read-only negative pool from two
 * SharedArrayBuffers (a flat Int32 bit-stream + per-context offsets) as zero-copy `Int32Array` subarray views,
 * then runs the deployed `fitClass` for each word it is handed and posts back the serializable model
 * `{G, theta}` (the α-sum head is rebuilt in the main thread via `makeHead`).
 */

import { parentPort, workerData } from "node:worker_threads";
import fitClass from "./fit.js";

// reconstruct the shared negPool as subarray views over the shared bit-stream (no copy, no re-featurization)
const flat = new Int32Array(workerData.sabFlat);
const off = new Int32Array(workerData.sabOff);
const nNeg = off.length - 1;
const negPool = new Array(nNeg);
for (let i = 0; i < nNeg; i++) negPool[i] = flat.subarray(off[i], off[i + 1]);
const Mglob = new Set(workerData.mglob);
const opts = workerData.opts || {};

parentPort.on("message", (msg) => {
  const { word, A } = msg;
  try {
    const fit = fitClass(A, negPool, Mglob, opts);
    parentPort.postMessage({ word, ok: true, G: fit.G, theta: fit.theta, rStar: fit.rStar });
  } catch (e) {
    parentPort.postMessage({ word, ok: false, error: String((e && e.message) || e) });
  }
});
