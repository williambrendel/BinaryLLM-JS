"use strict";

// ============================================================================
// core/signatures/signatureDataset.js
//
// (De)serialization of a signature dataset: one record per Word token, each
// holding the three F-wide bands [L, C, R] (local ids in [0,F)) plus two
// word-level labels — the current word C and the next word — and a scale tag
// (0 = sentence scope, 1 = paragraph scope).
//
// writeDataset emits the current format, SIG2. readDataset auto-detects and
// reads both SIG2 and the older SIG1 (see below), so existing files keep
// loading.
//
// ---- SIG2 (current): columnar + delta-varint --------------------------------
// A struct-of-arrays layout: fields are grouped into columns (all scales, then
// all current-word labels, …) so homogeneous values sit together (far better
// gzip), and every count/label/id is an unsigned LEB128 varint. Band ids are
// ascending, so each band stores gaps (delta) — small numbers, ~1 byte each.
//
//   HEADER: magic "SIG2", version u8=2, flags u8, reserved u16,
//           F u32, V u32, N u32
//   VOCAB (V): u16 len, len raw bytes (word, byte-string)
//   COLUMNS:
//     scale[N]           : u8 each
//     curWord[N]         : varint (0 = none, else id+1)
//     nextWord[N]        : varint (0 = none, else id+1)
//     for band in [L,C,R]:
//       pop[N]           : varint (band size)
//       ids              : per record, `pop` delta-varints (ascending gaps)
//
// ---- SIG1 (legacy, read-only): row-major, fixed-width -----------------------
//   HEADER: magic "SIG1", version u8=1, idWidth u8, reserved u16, F,V,N (u32)
//   VOCAB (V): u16 len + bytes
//   RECORDS (N): scale u8, curWord u32, nextWord u32,
//                per band: pop u32 + pop*idWidth ids
// ============================================================================

import { sparseArrayType } from "../../utilities/sparseArrayType.js";

const MAGIC2 = "SIG2";
const MAGIC1 = "SIG1";
const VERSION2 = 2;
const VERSION1 = 1;
export const NO_LABEL = 0xffffffff;

const BANDS = ["L", "C", "R"];

/**
 * @typedef {Object} SignatureRecord
 * @property {number} scale 0 = sentence scope, 1 = paragraph scope.
 * @property {number} curWord vocab id of the current word (or NO_LABEL).
 * @property {number} nextWord vocab id of the next word (or NO_LABEL).
 * @property {Uint16Array|Uint32Array} L before band (ascending local ids in [0,F)).
 * @property {Uint16Array|Uint32Array} C current band (ascending).
 * @property {Uint16Array|Uint32Array} R after band (ascending).
 */

// ---- byte writer / reader (varint = unsigned LEB128) ---------------------

class ByteWriter {
  constructor() {
    this.b = [];
  }
  u8(v) {
    this.b.push(v & 0xff);
  }
  u16(v) {
    this.b.push(v & 0xff, (v >> 8) & 0xff);
  }
  u32(v) {
    v >>>= 0;
    this.b.push(v & 0xff, (v >>> 8) & 0xff, (v >>> 16) & 0xff, (v >>> 24) & 0xff);
  }
  varint(v) {
    v >>>= 0;
    while (v > 0x7f) {
      this.b.push((v & 0x7f) | 0x80);
      v >>>= 7;
    }
    this.b.push(v);
  }
  chars(str) {
    for (let i = 0; i < str.length; i++) this.b.push(str.charCodeAt(i) & 0xff);
  }
  done() {
    return Uint8Array.from(this.b);
  }
}

class ByteReader {
  constructor(bytes, offset = 0) {
    this.a = bytes;
    this.o = offset;
  }
  u8() {
    return this.a[this.o++];
  }
  u16() {
    const v = this.a[this.o] | (this.a[this.o + 1] << 8);
    this.o += 2;
    return v;
  }
  u32() {
    const v =
      (this.a[this.o] | (this.a[this.o + 1] << 8) | (this.a[this.o + 2] << 16) | (this.a[this.o + 3] << 24)) >>> 0;
    this.o += 4;
    return v;
  }
  varint() {
    let shift = 0;
    let res = 0;
    let byte;
    do {
      byte = this.a[this.o++];
      res |= (byte & 0x7f) << shift;
      shift += 7;
    } while (byte & 0x80);
    return res >>> 0;
  }
  chars(len) {
    let s = "";
    for (let i = 0; i < len; i++) s += String.fromCharCode(this.a[this.o++]);
    return s;
  }
}

// ---- write (SIG2) --------------------------------------------------------

/**
 * Serialize a signature dataset to the SIG2 binary format.
 * @param {{F: number, vocab: string[], records: SignatureRecord[]}} data
 *   `vocab` holds words as byte-strings; bands must be ascending.
 * @returns {Uint8Array}
 */
export const writeDataset = ({ F, vocab, records }) => {
  const w = new ByteWriter();
  const N = records.length;

  w.chars(MAGIC2);
  w.u8(VERSION2);
  w.u8(0); // flags
  w.u16(0); // reserved
  w.u32(F);
  w.u32(vocab.length);
  w.u32(N);
  for (const word of vocab) {
    w.u16(word.length);
    w.chars(word);
  }

  for (const r of records) w.u8(r.scale);
  for (const r of records) w.varint(r.curWord === NO_LABEL ? 0 : r.curWord + 1);
  for (const r of records) w.varint(r.nextWord === NO_LABEL ? 0 : r.nextWord + 1);
  for (const key of BANDS) {
    for (const r of records) w.varint(r[key].length);
    for (const r of records) {
      const band = r[key];
      let prev = 0;
      for (let i = 0; i < band.length; i++) {
        w.varint(band[i] - prev); // ascending -> non-negative gap
        prev = band[i];
      }
    }
  }
  return w.done();
};

// ---- read (auto-detect SIG2 / SIG1) --------------------------------------

const readSig2 = (bytes) => {
  const r = new ByteReader(bytes, 4); // past magic
  const version = r.u8();
  if (version !== VERSION2) throw new Error("readDataset: unsupported SIG2 version " + version);
  r.u8(); // flags
  r.u16(); // reserved
  const F = r.u32();
  const V = r.u32();
  const N = r.u32();
  const Arr = sparseArrayType(F);

  const vocab = new Array(V);
  for (let i = 0; i < V; i++) vocab[i] = r.chars(r.u16());

  const scale = new Array(N);
  for (let i = 0; i < N; i++) scale[i] = r.u8();
  const cur = new Array(N);
  for (let i = 0; i < N; i++) {
    const v = r.varint();
    cur[i] = v === 0 ? NO_LABEL : v - 1;
  }
  const next = new Array(N);
  for (let i = 0; i < N; i++) {
    const v = r.varint();
    next[i] = v === 0 ? NO_LABEL : v - 1;
  }

  const bands = { L: null, C: null, R: null };
  for (const key of BANDS) {
    const pops = new Array(N);
    for (let i = 0; i < N; i++) pops[i] = r.varint();
    const col = new Array(N);
    for (let i = 0; i < N; i++) {
      const pop = pops[i];
      const b = new Arr(pop);
      let prev = 0;
      for (let j = 0; j < pop; j++) {
        prev += r.varint();
        b[j] = prev;
      }
      col[i] = b;
    }
    bands[key] = col;
  }

  const records = new Array(N);
  for (let i = 0; i < N; i++) {
    records[i] = { scale: scale[i], curWord: cur[i], nextWord: next[i], L: bands.L[i], C: bands.C[i], R: bands.R[i] };
  }
  return { F, vocab, records };
};

const readSig1 = (bytes) => {
  const r = new ByteReader(bytes, 4); // past magic
  const version = r.u8();
  if (version !== VERSION1) throw new Error("readDataset: unsupported SIG1 version " + version);
  const idWidth = r.u8();
  r.u16(); // reserved
  const F = r.u32();
  const V = r.u32();
  const N = r.u32();
  const readId = idWidth === 2 ? () => r.u16() : () => r.u32();
  const Arr = sparseArrayType(F);

  const vocab = new Array(V);
  for (let i = 0; i < V; i++) vocab[i] = r.chars(r.u16());

  const band = () => {
    const pop = r.u32();
    const b = new Arr(pop);
    for (let i = 0; i < pop; i++) b[i] = readId();
    return b;
  };
  const records = new Array(N);
  for (let i = 0; i < N; i++) {
    const scale = r.u8();
    const curWord = r.u32();
    const nextWord = r.u32();
    const L = band();
    const C = band();
    const R = band();
    records[i] = { scale, curWord, nextWord, L, C, R };
  }
  return { F, vocab, records };
};

/**
 * Parse a signature dataset (SIG2 or legacy SIG1 — auto-detected by magic).
 * @param {Uint8Array} bytes
 * @returns {{F: number, vocab: string[], records: SignatureRecord[]}}
 *   `vocab` words are byte-strings (use fromByteString to display).
 */
export const readDataset = (bytes) => {
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic === MAGIC2) return readSig2(bytes);
  if (magic === MAGIC1) return readSig1(bytes);
  throw new Error("readDataset: bad magic (expected SIG1 or SIG2)");
};

export default writeDataset;
