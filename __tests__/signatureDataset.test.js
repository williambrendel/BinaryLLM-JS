"use strict";

// ============================================================================
// __tests__/signatureDataset.test.js
//
// Round-trip tests for the SIG1 signature-dataset (de)serializer, plus a small
// end-to-end build over the fixture corpus using the real segmentation +
// encoder path.
// ============================================================================

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { tokenizeStream, StreamTokenType } from "../src/core/parts/tokenize.js";
import { loadDictText } from "../src/core/parts/dictionary.js";
import { encode } from "../src/core/signatures/encoder.js";
import { writeDataset, readDataset, NO_LABEL } from "../src/core/signatures/signatureDataset.js";

const FIX = path.join(path.dirname(fileURLToPath(import.meta.url)), "fixtures");
const dict = loadDictText(fs.readFileSync(path.join(FIX, "small_gold_dict.txt"), "latin1"));
const F = dict.size(); // 509

describe("signatureDataset write/read round-trip", () => {
  const vocab = ["the", "peppers", "were", "pickled"];
  const records = [
    {
      scale: 0,
      curWord: 1,
      nextWord: 2,
      L: Uint16Array.from([0]),
      C: Uint16Array.from([1]),
      R: Uint16Array.from([7, 9]),
    },
    {
      scale: 1,
      curWord: 3,
      nextWord: NO_LABEL,
      L: Uint16Array.from([]),
      C: Uint16Array.from([7]),
      R: Uint16Array.from([]),
    },
  ];

  test("preserves F, vocab, and every record field", () => {
    const back = readDataset(writeDataset({ F, vocab, records }));
    expect(back.F).toBe(F);
    expect(back.vocab).toEqual(vocab);
    expect(back.records).toHaveLength(2);
    const r0 = back.records[0];
    expect(r0.scale).toBe(0);
    expect(r0.curWord).toBe(1);
    expect(r0.nextWord).toBe(2);
    expect(Array.from(r0.L)).toEqual([0]);
    expect(Array.from(r0.C)).toEqual([1]);
    expect(Array.from(r0.R)).toEqual([7, 9]);
    const r1 = back.records[1];
    expect(r1.scale).toBe(1);
    expect(r1.nextWord).toBe(NO_LABEL);
    expect(Array.from(r1.L)).toEqual([]);
    expect(Array.from(r1.R)).toEqual([]);
  });

  test("bands come back as F-typed arrays (Uint16 for F=509)", () => {
    const back = readDataset(writeDataset({ F, vocab, records }));
    expect(back.records[0].C).toBeInstanceOf(Uint16Array);
  });

  test("writeDataset emits the SIG2 format", () => {
    const bytes = writeDataset({ F, vocab, records });
    expect(String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3])).toBe("SIG2");
  });

  test("rejects bad magic", () => {
    expect(() => readDataset(Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]))).toThrow(/magic/);
  });

  test("handles a wide dictionary (Uint32 bands, large ids)", () => {
    const wide = {
      F: 100000, // > 65536 -> Uint32 bands
      vocab: ["a"],
      records: [
        { scale: 0, curWord: 0, nextWord: NO_LABEL, L: Uint32Array.from([1, 99999]), C: Uint32Array.from([0]), R: Uint32Array.from([]) },
      ],
    };
    const back = readDataset(writeDataset(wide));
    expect(back.records[0].L).toBeInstanceOf(Uint32Array);
    expect(Array.from(back.records[0].L)).toEqual([1, 99999]); // delta-varint reconstructs large gaps
  });

  test("reads legacy SIG1 files (back-compat)", () => {
    // Hand-build a minimal SIG1 buffer (row-major, fixed-width, idWidth=2).
    const w = [];
    const u8 = (v) => w.push(v & 0xff);
    const u16 = (v) => w.push(v & 0xff, (v >> 8) & 0xff);
    const u32 = (v) => {
      v >>>= 0;
      w.push(v & 0xff, (v >>> 8) & 0xff, (v >>> 16) & 0xff, (v >>> 24) & 0xff);
    };
    "SIG1".split("").forEach((c) => u8(c.charCodeAt(0)));
    u8(1); // version
    u8(2); // idWidth
    u16(0); // reserved
    u32(509); // F
    u32(2); // V
    u32(1); // N
    for (const word of ["the", "cat"]) {
      u16(word.length);
      for (const ch of word) u8(ch.charCodeAt(0));
    }
    // one record: scale 1, cur 0, next 1, L=[3,4], C=[7], R=[]
    u8(1);
    u32(0);
    u32(1);
    u32(2); u16(3); u16(4); // L
    u32(1); u16(7); // C
    u32(0); // R
    const back = readDataset(Uint8Array.from(w));
    expect(back.F).toBe(509);
    expect(back.vocab).toEqual(["the", "cat"]);
    expect(back.records[0]).toMatchObject({ scale: 1, curWord: 0, nextWord: 1 });
    expect(Array.from(back.records[0].L)).toEqual([3, 4]);
    expect(Array.from(back.records[0].C)).toEqual([7]);
    expect(back.records[0].R.length).toBe(0);
  });
});

describe("end-to-end: encode a sentence into records", () => {
  test("labels line up: curWord/nextWord follow the word order", () => {
    const sentence = "the peppers were pickled";
    const toks = tokenizeStream(sentence);
    const words = toks.filter((t) => t.type === StreamTokenType.Word).map((t) => t.value);
    const sigs = encode(dict, toks);

    const vocabIndex = new Map();
    const vocab = [];
    const wordId = (v) => {
      let id = vocabIndex.get(v);
      if (id === undefined) vocabIndex.set(v, (id = vocab.push(v) - 1));
      return id;
    };
    const records = sigs.map(([L, C, R], i) => ({
      scale: 0,
      curWord: wordId(words[i]),
      nextWord: i + 1 < words.length ? wordId(words[i + 1]) : NO_LABEL,
      L,
      C,
      R,
    }));

    const back = readDataset(writeDataset({ F, vocab, records }));
    // "peppers" is record 1: cur = peppers, next = were.
    expect(back.vocab[back.records[1].curWord]).toBe("peppers");
    expect(back.vocab[back.records[1].nextWord]).toBe("were");
    // last word has no next.
    expect(back.records[3].nextWord).toBe(NO_LABEL);
    // its C band is the current word's Option B bag (peppers -> id 1).
    expect(Array.from(back.records[1].C)).toEqual([1]);
  });
});
