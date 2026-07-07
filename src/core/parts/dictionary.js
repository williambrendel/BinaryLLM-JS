"use strict";

// ============================================================================
// core/parts/dictionary.js
//
// Port of core/parts/dictionary.cpp + dict_io.cpp.
//
// PartDictionary stores the trained part vocabulary: built once by the
// extractor, then queried (read-only) by the decomposer. Identity is
// (kind, value); insertion order is the part's ID.
//
// Serialization uses the "trained-only" dict_io format: singletons (Letter
// atoms) and specials (single-char Whole atoms, connector Delimiter atoms)
// are NOT stored — augmentWithAtoms() regenerates them deterministically at
// load. Text: one value per line with the "##" position convention and a
// trailing [delim] section. Binary: "BDI2" header + (u8 kind, u16 len, bytes)
// records. All values are byte-strings (see byteString.js).
// ============================================================================

import { Kind, kInvalidPartId } from "./kind.js";

// Connector characters seeded by augmentWithAtoms / excluded from files.
const CONNECTORS = ["-", "'", "&", ",", ".", "$"];
const CONNECTOR_CODES = new Set(CONNECTORS.map((c) => c.charCodeAt(0)));

const keyOf = (kind, value) => String.fromCharCode(kind) + value;

/**
 * The trained part vocabulary. Keys are (kind, value) pairs; the insertion
 * index is the part ID.
 */
export class PartDictionary {
  constructor() {
    /** @type {Array<{kind: number, value: string}>} */
    this._idToKey = [];
    /** @type {Map<string, number>} */
    this._keyToId = new Map();
  }

  /**
   * Add a (kind, value) part, or return the existing ID if already present.
   * @param {number} kind
   * @param {string} value byte-string.
   * @returns {number} the part ID.
   */
  add(kind, value) {
    const k = keyOf(kind, value);
    const existing = this._keyToId.get(k);
    if (existing !== undefined) return existing;
    const id = this._idToKey.length;
    this._idToKey.push({ kind, value });
    this._keyToId.set(k, id);
    return id;
  }

  /**
   * @param {number} kind
   * @param {string} value byte-string.
   * @returns {number} the part ID, or kInvalidPartId.
   */
  lookup(kind, value) {
    const id = this._keyToId.get(keyOf(kind, value));
    return id === undefined ? kInvalidPartId : id;
  }

  /**
   * @param {number} id
   * @returns {{kind: number, value: string}}
   */
  at(id) {
    return this._idToKey[id];
  }

  /** @returns {number} */
  size() {
    return this._idToKey.length;
  }

  /**
   * @param {number} kind
   * @returns {number}
   */
  countOfKind(kind) {
    let n = 0;
    for (const k of this._idToKey) if (k.kind === kind) n++;
    return n;
  }

  /** @returns {Array<{kind: number, value: string}>} parts in ID order. */
  allParts() {
    return this._idToKey;
  }

  hasStart(value) {
    return this.lookup(Kind.Start, value) !== kInvalidPartId;
  }
  hasEnd(value) {
    return this.lookup(Kind.End, value) !== kInvalidPartId;
  }
  hasMid(value) {
    return this.lookup(Kind.Mid, value) !== kInvalidPartId;
  }
  hasWhole(value) {
    return this.lookup(Kind.Whole, value) !== kInvalidPartId;
  }
  hasDelimiter(value) {
    return this.lookup(Kind.Delimiter, value) !== kInvalidPartId;
  }
}

// ---- Augment set membership ----------------------------------------------

const isConnectorValue = (value) => value.length === 1 && CONNECTOR_CODES.has(value.charCodeAt(0));

// True if `augmentWithAtoms` would regenerate this part (so IO omits it).
const isAugmented = ({ kind, value }) => {
  if (kind === Kind.Letter) return true;
  if (kind === Kind.Whole && value.length === 1) {
    const c = value.charCodeAt(0);
    if (c >= 97 && c <= 122) return true; // a-z
    if (c >= 48 && c <= 57) return true; // 0-9
    if (isConnectorValue(value)) return true;
  }
  if (kind === Kind.Delimiter && isConnectorValue(value)) return true;
  return false;
};

/**
 * Regenerate the deterministic singleton/special atoms omitted from files:
 * single-char Whole atoms, positional Letter atoms, and connector Delimiter
 * atoms. Idempotent (add() dedups).
 * @param {PartDictionary} dict
 */
export const augmentWithAtoms = (dict) => {
  const addLetterPositions = (s) => {
    dict.add(Kind.Letter, s + "##");
    dict.add(Kind.Letter, "##" + s + "##");
    dict.add(Kind.Letter, "##" + s);
  };
  for (let c = 97; c <= 122; c++) dict.add(Kind.Whole, String.fromCharCode(c));
  for (let c = 48; c <= 57; c++) dict.add(Kind.Whole, String.fromCharCode(c));
  for (const c of CONNECTORS) dict.add(Kind.Whole, c);

  for (let c = 97; c <= 122; c++) addLetterPositions(String.fromCharCode(c));
  for (let c = 48; c <= 57; c++) addLetterPositions(String.fromCharCode(c));
  for (const c of CONNECTORS) addLetterPositions(c);

  for (const c of CONNECTORS) dict.add(Kind.Delimiter, c);
};

// ---- Text escape / unescape (for delimiter values) -----------------------

const escapeValue = (s) => {
  let out = "";
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c === 10) out += "\\n";
    else if (c === 9) out += "\\t";
    else if (c === 13) out += "\\r";
    else if (c === 92) out += "\\\\";
    else if (c < 0x20 || c === 0x7f) {
      out += "\\x" + c.toString(16).toUpperCase().padStart(2, "0");
    } else out += s[i];
  }
  return out;
};

const hexDigit = (c) => {
  if (c >= 48 && c <= 57) return c - 48;
  if (c >= 97 && c <= 102) return c - 97 + 10;
  if (c >= 65 && c <= 70) return c - 65 + 10;
  return -1;
};

const unescapeValue = (s) => {
  let out = "";
  for (let i = 0; i < s.length; ) {
    if (s.charCodeAt(i) !== 92) {
      out += s[i++];
      continue;
    }
    if (i + 1 >= s.length) {
      out += "\\";
      i++;
      continue;
    }
    const esc = s.charCodeAt(i + 1);
    if (esc === 110) {
      out += "\n";
      i += 2;
    } else if (esc === 116) {
      out += "\t";
      i += 2;
    } else if (esc === 114) {
      out += "\r";
      i += 2;
    } else if (esc === 92) {
      out += "\\";
      i += 2;
    } else if (esc === 120) {
      // \xHH
      if (i + 3 < s.length) {
        const h = hexDigit(s.charCodeAt(i + 2));
        const l = hexDigit(s.charCodeAt(i + 3));
        if (h >= 0 && l >= 0) {
          out += String.fromCharCode((h << 4) | l);
          i += 4;
          continue;
        }
      }
      out += "\\";
      i++;
    } else {
      out += "\\";
      i++;
    }
  }
  return out;
};

const startsWithHashHash = (s) => s.length >= 2 && s.charCodeAt(0) === 35 && s.charCodeAt(1) === 35;
const endsWithHashHash = (s) =>
  s.length >= 2 && s.charCodeAt(s.length - 2) === 35 && s.charCodeAt(s.length - 1) === 35;

// ---- Text IO -------------------------------------------------------------

/**
 * Serialize to the trained-only text format. Returns a byte-string; write it
 * to disk as latin1 to reproduce the C++ bytes exactly.
 * @param {PartDictionary} dict
 * @returns {string} byte-string.
 */
export const saveDictText = (dict) => {
  let out = "";
  // Pass 1: Whole / Start / Mid / End in ID order.
  for (const k of dict.allParts()) {
    if (isAugmented(k)) continue;
    switch (k.kind) {
      case Kind.Whole:
        out += k.value + "\n";
        break;
      case Kind.Start:
        out += k.value + "##\n";
        break;
      case Kind.Mid:
        out += "##" + k.value + "##\n";
        break;
      case Kind.End:
        out += "##" + k.value + "\n";
        break;
      default:
        break; // Letter filtered above, Delimiter handled below
    }
  }
  // Pass 2: Delimiter section.
  out += "[delim]\n";
  for (const k of dict.allParts()) {
    if (k.kind !== Kind.Delimiter) continue;
    if (isAugmented(k)) continue;
    out += escapeValue(k.value) + "\n";
  }
  return out;
};

/**
 * Parse the trained-only text format and augment with atoms.
 * @param {string} text byte-string (read the file as latin1).
 * @returns {PartDictionary}
 */
export const loadDictText = (text) => {
  const dict = new PartDictionary();
  let inDelim = false;
  for (let line of text.split("\n")) {
    while (line.length > 0 && line.charCodeAt(line.length - 1) === 13) {
      line = line.slice(0, -1);
    }
    if (line.length === 0) continue;
    if (line === "[delim]") {
      inDelim = true;
      continue;
    }
    if (inDelim) {
      dict.add(Kind.Delimiter, unescapeValue(line));
      continue;
    }
    const s = startsWithHashHash(line);
    const e = endsWithHashHash(line);
    if (s && e) {
      if (line.length >= 4) dict.add(Kind.Mid, line.slice(2, line.length - 2));
    } else if (e) {
      dict.add(Kind.Start, line.slice(0, line.length - 2));
    } else if (s) {
      dict.add(Kind.End, line.slice(2));
    } else {
      dict.add(Kind.Whole, line);
    }
  }
  augmentWithAtoms(dict);
  return dict;
};

// ---- Binary IO (BDI2) ----------------------------------------------------

const MAGIC_BDI2 = [0x42, 0x44, 0x49, 0x32]; // "BDI2"
const BINARY_VERSION = 1;

/**
 * Serialize to the trained-only binary (BDI2) format.
 * @param {PartDictionary} dict
 * @returns {Uint8Array}
 */
export const saveDictBinary = (dict) => {
  const bytes = [...MAGIC_BDI2, BINARY_VERSION, 0, 0, 0]; // header: magic + ver + reserved u8 + reserved u16
  for (const k of dict.allParts()) {
    if (isAugmented(k)) continue;
    const len = k.value.length;
    if (len > 0xffff) throw new Error("part value exceeds uint16 length");
    bytes.push(k.kind); // kind byte == Kind numeric value (Start0..Delimiter5)
    bytes.push(len & 0xff, (len >> 8) & 0xff); // u16 LE
    for (let i = 0; i < len; i++) bytes.push(k.value.charCodeAt(i) & 0xff);
  }
  return Uint8Array.from(bytes);
};

/**
 * Parse the trained-only binary (BDI2) format and augment with atoms.
 * @param {Uint8Array} bytes
 * @returns {PartDictionary}
 */
export const loadDictBinary = (bytes) => {
  if (
    bytes.length < 8 ||
    bytes[0] !== MAGIC_BDI2[0] ||
    bytes[1] !== MAGIC_BDI2[1] ||
    bytes[2] !== MAGIC_BDI2[2] ||
    bytes[3] !== MAGIC_BDI2[3]
  ) {
    throw new Error("loadDictBinary: bad magic (expected BDI2)");
  }
  const version = bytes[4];
  if (version !== BINARY_VERSION) {
    throw new Error("loadDictBinary: unsupported version " + version);
  }
  // bytes[5] reserved u8, bytes[6..7] reserved u16.

  const dict = new PartDictionary();
  let p = 8;
  while (p < bytes.length) {
    const kind = bytes[p++]; // clean EOF if p was already == length
    if (kind > 5) throw new Error("loadDictBinary: bad kind byte");
    if (p + 2 > bytes.length) throw new Error("loadDictBinary: truncated length");
    const len = bytes[p] | (bytes[p + 1] << 8);
    p += 2;
    if (p + len > bytes.length) throw new Error("loadDictBinary: truncated value");
    let value = "";
    for (let i = 0; i < len; i++) value += String.fromCharCode(bytes[p + i]);
    p += len;
    dict.add(kind, value);
  }
  augmentWithAtoms(dict);
  return dict;
};

/**
 * @ignore
 */
export default PartDictionary;