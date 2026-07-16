"use strict";

/**
 * @file parallelFit.js
 * @brief Parallel per-word part extraction. Featurization is done ONCE by the caller (shared corpus pass →
 * per-word contexts + one confusable negative pool); this module fans the per-word `fitClass` out across a
 * worker pool. Word extraction is embarrassingly parallel (each word's model is independent), so throughput
 * scales ~linearly with cores.
 *
 * The large read-only negative pool (~10² k contexts) is shared **zero-copy** via `SharedArrayBuffer` — a flat
 * Int32 bit-stream + per-context offsets — so it is neither re-featurized nor re-serialized per worker. A
 * dynamic queue hands the next word to whichever worker frees up (extraction time is very uneven across
 * words). `fitClass` is deterministic, so the result is identical to a sequential run.
 */

import { Worker } from "node:worker_threads";
import os from "node:os";
import { fileURLToPath } from "node:url";
import makeHead from "./head.js";

const WORKER_URL = new URL("./fitWorker.js", import.meta.url);

/** Flatten Array<sorted int array> into a shared {flat, off} Int32 pair. */
const shareNegPool = (negPool) => {
  let total = 0; for (const y of negPool) total += y.length;
  const sabFlat = new SharedArrayBuffer(total * 4), sabOff = new SharedArrayBuffer((negPool.length + 1) * 4);
  const flat = new Int32Array(sabFlat), off = new Int32Array(sabOff);
  let o = 0;
  for (let i = 0; i < negPool.length; i++) { off[i] = o; const y = negPool[i]; for (let k = 0; k < y.length; k++) flat[o++] = y[k]; }
  off[negPool.length] = o;
  return { sabFlat, sabOff };
};

/**
 * @function parallelFit
 * @param {Map<string, Array<number[]>>} classA - per-word positive contexts (featurized once).
 * @param {Array<number[]>} negPool - the shared confusable candidate pool (read-only).
 * @param {Set<number>} Mglob - common-bit mask.
 * @param {object} [opts] - forwarded to `fitClass` (rho, delta, solver, …).
 * @param {object} [poolOpts] @param {number} [poolOpts.workers] - worker count (default cores−1).
 * @returns {Promise<Map<string, {G:Array, theta:number, head:object}>>} per-word model (head rebuilt here).
 */
export const parallelFit = (classA, negPool, Mglob, opts = {}, poolOpts = {}) => {
  const words = [...classA.keys()];
  const N = Math.max(1, Math.min(poolOpts.workers || os.cpus().length - 1, words.length));
  const { sabFlat, sabOff } = shareNegPool(negPool);
  const workerData = { sabFlat, sabOff, mglob: [...Mglob], opts };
  const results = new Map();
  const queue = words.slice();

  return new Promise((resolve, reject) => {
    let done = 0; const workers = [];
    const assign = (w) => { if (queue.length) w.postMessage({ word: queue[0], A: classA.get(queue.shift()) }); };
    const finishOne = (w, msg) => {
      results.set(msg.word, msg.ok ? { G: msg.G, theta: msg.theta, head: makeHead(msg.G, msg.theta) } : { error: msg.error });
      if (++done === words.length) { for (const x of workers) x.terminate(); resolve(results); return; }
      assign(w);
    };
    for (let i = 0; i < N; i++) {
      const w = new Worker(WORKER_URL, { workerData });
      workers.push(w);
      w.on("message", (msg) => finishOne(w, msg));
      w.on("error", (e) => { for (const x of workers) x.terminate(); reject(e); });
      assign(w);   // seed each worker with one word
    }
  });
};

export default parallelFit;

// Support running fitWorker via a URL when imported from a different cwd (worker_threads needs a resolvable path).
export const _WORKER_PATH = fileURLToPath(WORKER_URL);
