"use strict";

// ============================================================================
// core/signatures/signatureDataset.js
//
// Binary (de)serialization of a signature dataset: one record per Word token,
// each holding the three F-wide bands [L, C, R] (local ids in [0,F)) plus two
// word-level labels — the current word C and the next word — and a scale tag
// (0 = sentence scope, 1 = paragraph scope). This is the JS analog of the C++
// build_candidates 3F stream, but with training labels attached.
//
// Format "SIG1" (all integers little-endian):
//   HEADER (20 bytes):
//     magic[4]   = "SIG1"
//     version u8 = 1
//     idWidth u8 = band-id byte width (2 if F<=65536 else 4)
//     reserved u16
//     F   u32    = dictionary size (band width)
//     V   u32    = vocabulary size (label space)
//     N   u32    = record count
//   VOCAB (V entries): u16 len, len raw bytes (the word, byte-string)
//   RECORDS (N entries):
//     scale u8         (0 sentence, 1 paragraph)
//     curWord  u32     (vocab id of C; 0xFFFFFFFF if absent)
//     nextWord u32     (vocab id of the next word; 0xFFFFFFFF at a scope end)
//     for band in [L, C, R]:
//       pop u32, pop * idWidth-byte ids
// ============================================================================

import { sparseArrayType } from "../../utilities/sparseArrayType.js";

const MAGIC = [0x53, 0x49, 0x47, 0x31]; // "SIG1"
const VERSION = 1;
export const NO_LABEL = 0xffffffff;

/**
 * @typedef {Object} SignatureRecord
 * @property {number} scale 0 = sentence scope, 1 = paragraph scope.
 * @property {number} curWord vocab id of the current word (or NO_LABEL).
 * @property {number} nextWord vocab id of the next word (or NO_LABEL).
 * @property {Uint16Array|Uint32Array} L before band (local ids in [0,F)).
 * @property {Uint16Array|Uint32Array} C current band.
 * @property {Uint16Array|Uint32Array} R after band.
 */

/**
 * Serialize a signature dataset to the SIG1 binary format.
 * @param {{F: number, vocab: string[], records: SignatureRecord[]}} data
 *   `vocab` holds words as byte-strings (char codes in [0,255]).
 * @returns {Uint8Array}
 */
export const writeDataset = ({ F, vocab, records }) => {
  const idWidth = F <= 0x10000 ? 2 : 4;

  let size = 20;
  for (const w of vocab) size += 2 + w.length;
  for (const r of records) {
    size += 1 + 4 + 4;
    size += 4 + r.L.length * idWidth + 4 + r.C.length * idWidth + 4 + r.R.length * idWidth;
  }

  const buf = new Uint8Array(size);
  const dv = new DataView(buf.buffer);
  let o = 0;
  const u8 = (v) => {
    dv.setUint8(o, v);
    o += 1;
  };
  const u16 = (v) => {
    dv.setUint16(o, v, true);
    o += 2;
  };
  const u32 = (v) => {
    dv.setUint32(o, v >>> 0, true);
    o += 4;
  };
  const id = idWidth === 2 ? u16 : u32;
  const band = (b) => {
    u32(b.length);
    for (let i = 0; i < b.length; i++) id(b[i]);
  };

  for (const b of MAGIC) u8(b);
  u8(VERSION);
  u8(idWidth);
  u16(0); // reserved
  u32(F);
  u32(vocab.length);
  u32(records.length);
  for (const w of vocab) {
    u16(w.length);
    for (let i = 0; i < w.length; i++) u8(w.charCodeAt(i) & 0xff);
  }
  for (const r of records) {
    u8(r.scale);
    u32(r.curWord);
    u32(r.nextWord);
    band(r.L);
    band(r.C);
    band(r.R);
  }
  return buf;
};

/**
 * Parse the SIG1 binary format.
 * @param {Uint8Array} bytes
 * @returns {{F: number, vocab: string[], records: SignatureRecord[]}}
 *   `vocab` words are byte-strings (use fromByteString to display).
 */
export const readDataset = (bytes) => {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let o = 0;
  const u8 = () => dv.getUint8(o++);
  const u16 = () => {
    const v = dv.getUint16(o, true);
    o += 2;
    return v;
  };
  const u32 = () => {
    const v = dv.getUint32(o, true);
    o += 4;
    return v;
  };

  for (const m of MAGIC) if (u8() !== m) throw new Error("readDataset: bad magic (expected SIG1)");
  const version = u8();
  if (version !== VERSION) throw new Error("readDataset: unsupported version " + version);
  const idWidth = u8();
  u16(); // reserved
  const F = u32();
  const V = u32();
  const N = u32();
  const readId = idWidth === 2 ? u16 : u32;
  const Arr = sparseArrayType(F);

  const vocab = new Array(V);
  for (let i = 0; i < V; i++) {
    const len = u16();
    let s = "";
    for (let k = 0; k < len; k++) s += String.fromCharCode(u8());
    vocab[i] = s;
  }

  const band = () => {
    const pop = u32();
    const b = new Arr(pop);
    for (let i = 0; i < pop; i++) b[i] = readId();
    return b;
  };
  const records = new Array(N);
  for (let i = 0; i < N; i++) {
    const scale = u8();
    const curWord = u32();
    const nextWord = u32();
    const L = band();
    const C = band();
    const R = band();
    records[i] = { scale, curWord, nextWord, L, C, R };
  }
  return { F, vocab, records };
};

export default writeDataset;
