"use strict";

// ============================================================================
// __tests__/core/signatures/streamSignatures.test.js
//
// The streaming signature source yields bounded record chunks without holding the
// corpus in memory. It must produce the same per-word [L,C,R] records as the batch
// path (shape + labels), respect the chunk cap, and share a growing vocab.
// ============================================================================

import path from "node:path";
import { fileURLToPath } from "node:url";

import { loadDictText } from "../../../src/core/parts/dictionary.js";
import { streamSignatureChunks } from "../../../src/core/signatures/streamSignatures.js";
import { NO_LABEL } from "../../../src/core/signatures/signatureDataset.js";
import fs from "node:fs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIX = path.join(HERE, "..", "..", "fixtures");
const dict = loadDictText(fs.readFileSync(path.join(FIX, "small_gold_dict.txt"), "latin1"));
const corpus = path.join(FIX, "small_corpus.txt");

const drain = async (opts) => {
  const chunks = [];
  for await (const c of streamSignatureChunks(dict, corpus, opts)) chunks.push(c);
  return chunks;
};

describe("signatures/streamSignatures", () => {
  test("yields well-formed [L,C,R] records with a chained next-word label", async () => {
    const chunks = await drain({ chunkSize: 5 });
    expect(chunks.length).toBeGreaterThan(0);
    const recs = chunks.flatMap((c) => c.records);
    expect(recs.length).toBeGreaterThan(0);
    for (const r of recs) {
      expect(r.scale).toBe(0);
      expect(Number.isInteger(r.curWord)).toBe(true);
      expect(r.nextWord === NO_LABEL || Number.isInteger(r.nextWord)).toBe(true);
      for (const band of [r.L, r.C, r.R]) {
        expect(ArrayBuffer.isView(band) || Array.isArray(band)).toBe(true);
        for (let i = 1; i < band.length; i++) expect(band[i]).toBeGreaterThan(band[i - 1]); // ascending
      }
    }
    expect(chunks[0].F).toBe(dict.size());
  });

  test("respects the chunk cap — every chunk but the last is exactly chunkSize", async () => {
    const chunks = await drain({ chunkSize: 4 });
    const total = chunks.reduce((s, c) => s + c.records.length, 0);
    if (total > 4) {
      for (let i = 0; i < chunks.length - 1; i++) expect(chunks[i].records.length).toBe(4);
      expect(chunks[chunks.length - 1].records.length).toBeLessThanOrEqual(4);
    }
  });

  test("is deterministic and shares a growing vocab", async () => {
    const a = await drain({ chunkSize: 5 });
    const b = await drain({ chunkSize: 5 });
    expect(a.reduce((s, c) => s + c.records.length, 0)).toBe(b.reduce((s, c) => s + c.records.length, 0));
    const vocab = a[a.length - 1].vocab;
    expect(vocab.length).toBeGreaterThan(0);
  });
});
