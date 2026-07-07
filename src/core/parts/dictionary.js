"use strict";

/**
 * @file dictionary.js
 * @brief Trained part vocabulary ({@link PartDictionary}) plus its text and
 * binary serialization.
 *
 * Port of `core/parts/dictionary.cpp` + `dict_io.cpp`.
 *
 * A {@link PartDictionary} stores the trained part vocabulary: it is built
 * once by the extractor, then queried read-only by the decomposer. A part's
 * identity is the pair `(kind, value)` (see {@link Kind}); insertion order is
 * the part's ID.
 *
 * Serialization uses the "trained-only" `dict_io` format: singletons (Letter
 * atoms) and specials (single-char Whole atoms, connector Delimiter atoms) are
 * **not** stored — {@link augmentWithAtoms} regenerates them deterministically
 * at load. The text form ({@link saveDictText} / {@link loadDictText}) writes
 * one value per line using the `"##"` position convention with a trailing
 * `[delim]` section. The binary form ({@link saveDictBinary} /
 * {@link loadDictBinary}) writes a `"BDI2"` header followed by
 * `(u8 kind, u16 len, bytes)` records. All values are byte-strings (see
 * `byteString.js`).
 *
 * **Exported surface:**
 * - `PartDictionary` — the vocabulary container.
 * - `augmentWithAtoms(dict)` — seed the deterministic singleton/special atoms.
 * - `saveDictText(dict)` / `loadDictText(text)` — trained-only text IO.
 * - `saveDictBinary(dict)` / `loadDictBinary(bytes)` — trained-only binary IO.
 *
 * @see {@link Kind}
 */

import { Kind, kInvalidPartId } from "./kind.js";

// Connector characters seeded by augmentWithAtoms / excluded from files.
const CONNECTORS = ["-", "'", "&", ",", ".", "$"];
const CONNECTOR_CODES = new Set(CONNECTORS.map((c) => c.charCodeAt(0)));

const keyOf = (kind, value) => String.fromCharCode(kind) + value;

/**
 * @class PartDictionary
 * @description The trained part vocabulary. Each entry is keyed by its
 * `(kind, value)` pair, and its insertion index is the stable part ID. Adding
 * the same `(kind, value)` twice is a no-op that returns the original ID, so
 * IDs never shift once assigned. Values are byte-strings (see `byteString.js`).
 *
 * A dictionary is built once by the extractor (via repeated {@link PartDictionary#add}
 * calls), then queried read-only by the decomposer through {@link PartDictionary#lookup}
 * and the `has*` predicates. Note that identity is on the pair: the same value
 * under a different {@link Kind} is a distinct part with its own ID.
 *
 * @example
 * const dict = new PartDictionary();
 * const the = dict.add(Kind.Whole, "the");   // → 0  (first part, ID 0)
 * dict.add(Kind.Whole, "the");               // → 0  (dedups: same ID)
 * dict.add(Kind.Start, "the");               // → 1  (distinct kind, new ID)
 * dict.size();                               // → 2
 * dict.lookup(Kind.Whole, "the");            // → 0
 * dict.lookup(Kind.Whole, "missing");        // → kInvalidPartId
 * dict.hasWhole("the");                      // → true
 * dict.hasStart("the");                      // → true
 * dict.hasStart("missing");                  // → false
 */
export class PartDictionary {
  constructor() {
    /** @type {Array<{kind: number, value: string}>} */
    this._idToKey = [];
    /** @type {Map<string, number>} */
    this._keyToId = new Map();
  }

  /**
   * @method add
   * @description Adds a `(kind, value)` part, or returns the existing ID if the
   * pair is already present. IDs are assigned in insertion order and never
   * change, so this is safe to call repeatedly while building the vocabulary.
   * @param {number} kind - A {@link Kind} value.
   * @param {string} value - The part value as a byte-string.
   * @returns {number} The part ID (newly assigned, or the existing one).
   *
   * @example
   * const dict = new PartDictionary();
   * dict.add(Kind.Whole, "the");   // → 0
   * dict.add(Kind.Whole, "the");   // → 0  (already present, deduped)
   * dict.add(Kind.End, "ing");     // → 1
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
   * @method lookup
   * @description Returns the ID of the `(kind, value)` part, or
   * {@link kInvalidPartId} if it is not in the dictionary.
   * @param {number} kind - A {@link Kind} value.
   * @param {string} value - The part value as a byte-string.
   * @returns {number} The part ID, or `kInvalidPartId` when absent.
   *
   * @example
   * const dict = new PartDictionary();
   * dict.add(Kind.Whole, "the");
   * dict.lookup(Kind.Whole, "the");     // → 0
   * dict.lookup(Kind.Whole, "missing"); // → kInvalidPartId
   * dict.lookup(Kind.Start, "the");     // → kInvalidPartId  (kind differs)
   */
  lookup(kind, value) {
    const id = this._keyToId.get(keyOf(kind, value));
    return id === undefined ? kInvalidPartId : id;
  }

  /**
   * @method at
   * @description Returns the `(kind, value)` entry stored at the given part ID,
   * or `undefined` if the ID is out of range.
   * @param {number} id - A part ID (as returned by {@link PartDictionary#add}).
   * @returns {{ kind: number, value: string } | undefined} The entry, or
   *   `undefined` when the ID is not present.
   *
   * @example
   * const dict = new PartDictionary();
   * dict.add(Kind.Whole, "the");
   * dict.at(0);   // → { kind: Kind.Whole, value: "the" }
   * dict.at(99);  // → undefined
   */
  at(id) {
    return this._idToKey[id];
  }

  /**
   * @method size
   * @description Returns the number of parts in the dictionary.
   * @returns {number} The part count (also the ID that the next {@link PartDictionary#add}
   *   would assign).
   *
   * @example
   * const dict = new PartDictionary();
   * dict.size();                 // → 0
   * dict.add(Kind.Whole, "the");
   * dict.size();                 // → 1
   */
  size() {
    return this._idToKey.length;
  }

  /**
   * @method countOfKind
   * @description Counts how many parts in the dictionary have the given
   * {@link Kind}.
   * @param {number} kind - A {@link Kind} value.
   * @returns {number} The number of parts of that kind.
   *
   * @example
   * const dict = new PartDictionary();
   * augmentWithAtoms(dict);
   * dict.countOfKind(Kind.Whole);      // → 42
   * dict.countOfKind(Kind.Letter);     // → 126
   * dict.countOfKind(Kind.Delimiter);  // → 6
   */
  countOfKind(kind) {
    let n = 0;
    for (const k of this._idToKey) if (k.kind === kind) n++;
    return n;
  }

  /**
   * @method allParts
   * @description Returns the backing array of `(kind, value)` entries in ID
   * order. The array is returned by reference (not a copy) — treat it as
   * read-only. Used by the serializers to iterate the vocabulary.
   * @returns {Array<{ kind: number, value: string }>} Parts in ID order.
   *
   * @example
   * const dict = new PartDictionary();
   * dict.add(Kind.Whole, "the");
   * dict.add(Kind.End, "ing");
   * dict.allParts();
   * // → [{ kind: Kind.Whole, value: "the" }, { kind: Kind.End, value: "ing" }]
   */
  allParts() {
    return this._idToKey;
  }

  /**
   * @method hasStart
   * @description Returns `true` if a {@link Kind.Start} part with the given
   * value is in the dictionary.
   * @param {string} value - The part value as a byte-string.
   * @returns {boolean}
   *
   * @example
   * const dict = new PartDictionary();
   * dict.add(Kind.Start, "un");
   * dict.hasStart("un");   // → true
   * dict.hasStart("re");   // → false
   */
  hasStart(value) {
    return this.lookup(Kind.Start, value) !== kInvalidPartId;
  }

  /**
   * @method hasEnd
   * @description Returns `true` if a {@link Kind.End} part with the given value
   * is in the dictionary.
   * @param {string} value - The part value as a byte-string.
   * @returns {boolean}
   *
   * @example
   * const dict = new PartDictionary();
   * dict.add(Kind.End, "ing");
   * dict.hasEnd("ing");   // → true
   * dict.hasEnd("ed");    // → false
   */
  hasEnd(value) {
    return this.lookup(Kind.End, value) !== kInvalidPartId;
  }

  /**
   * @method hasMid
   * @description Returns `true` if a {@link Kind.Mid} part with the given value
   * is in the dictionary.
   * @param {string} value - The part value as a byte-string.
   * @returns {boolean}
   *
   * @example
   * const dict = new PartDictionary();
   * dict.add(Kind.Mid, "port");
   * dict.hasMid("port");    // → true
   * dict.hasMid("graph");   // → false
   */
  hasMid(value) {
    return this.lookup(Kind.Mid, value) !== kInvalidPartId;
  }

  /**
   * @method hasWhole
   * @description Returns `true` if a {@link Kind.Whole} part with the given
   * value is in the dictionary.
   * @param {string} value - The part value as a byte-string.
   * @returns {boolean}
   *
   * @example
   * const dict = new PartDictionary();
   * dict.add(Kind.Whole, "the");
   * dict.hasWhole("the");   // → true
   * dict.hasWhole("am");    // → false
   */
  hasWhole(value) {
    return this.lookup(Kind.Whole, value) !== kInvalidPartId;
  }

  /**
   * @method hasDelimiter
   * @description Returns `true` if a {@link Kind.Delimiter} part with the given
   * value is in the dictionary.
   * @param {string} value - The part value as a byte-string.
   * @returns {boolean}
   *
   * @example
   * const dict = new PartDictionary();
   * augmentWithAtoms(dict);
   * dict.hasDelimiter("-");   // → true
   * dict.hasDelimiter("?");   // → false
   */
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
 * @function augmentWithAtoms
 * @description Regenerates the deterministic singleton/special atoms that the
 * serializers omit from files, adding them into `dict` in place:
 * - single-char {@link Kind.Whole} atoms for `a`–`z`, `0`–`9`, and the six
 *   connectors (`- ' & , . $`);
 * - the three positional {@link Kind.Letter} forms (`x##`, `##x##`, `##x`) for
 *   each of those same 42 characters;
 * - one {@link Kind.Delimiter} atom per connector.
 *
 * Because {@link PartDictionary#add} dedups, this is idempotent and safe to run
 * on an already-populated dictionary — it only fills in any missing atoms. This
 * is invoked automatically at the end of {@link loadDictText} and
 * {@link loadDictBinary}.
 * @param {PartDictionary} dict - The dictionary to seed, mutated in place.
 * @returns {void}
 *
 * @example
 * const dict = new PartDictionary();
 * augmentWithAtoms(dict);
 * dict.countOfKind(Kind.Whole);       // → 42   (26 letters + 10 digits + 6 connectors)
 * dict.countOfKind(Kind.Letter);      // → 126  (42 chars × 3 positions)
 * dict.countOfKind(Kind.Delimiter);   // → 6    (one per connector)
 * dict.lookup(Kind.Letter, "##a##");  // not kInvalidPartId
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
 * @function saveDictText
 * @description Serializes a dictionary to the trained-only text format. Atoms
 * that {@link augmentWithAtoms} would regenerate are omitted. Whole/Start/Mid/End
 * parts are written first in ID order (using the `"##"` position convention),
 * followed by a `[delim]` section listing the trained Delimiter values (escaped).
 * Returns a byte-string; write it to disk as latin1 to reproduce the C++ bytes
 * exactly.
 * @param {PartDictionary} dict - The dictionary to serialize.
 * @returns {string} The serialized text as a byte-string.
 *
 * @example
 * const dict = new PartDictionary();
 * dict.add(Kind.Whole, "the");
 * dict.add(Kind.Start, "un");
 * dict.add(Kind.End, "ing");
 * dict.add(Kind.Mid, "port");
 * saveDictText(dict);
 * // → "the\nun##\n##ing\n##port##\n[delim]\n"
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
 * @function loadDictText
 * @description Parses the trained-only text format produced by
 * {@link saveDictText} into a new {@link PartDictionary}, then calls
 * {@link augmentWithAtoms} to restore the omitted singleton/special atoms.
 * Blank lines are skipped and trailing `\r` (CRLF) is tolerated; the `[delim]`
 * marker switches parsing into the delimiter section. Read the file as latin1
 * so the bytes match the C++ output exactly.
 * @param {string} text - The serialized text as a byte-string.
 * @returns {PartDictionary} A dictionary containing the parsed parts plus the
 *   regenerated atoms.
 *
 * @example
 * const dict = loadDictText("the\nun##\n##ing\n[delim]\n");
 * dict.hasWhole("the");   // → true
 * dict.hasStart("un");    // → true
 * dict.hasEnd("ing");     // → true
 * dict.hasWhole("a");     // → true  (restored by augmentWithAtoms)
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
 * @function saveDictBinary
 * @description Serializes a dictionary to the trained-only binary `BDI2` format:
 * a `"BDI2"` magic, a version byte, three reserved bytes, then one
 * `(u8 kind, u16 length LE, value bytes)` record per trained part in ID order.
 * Atoms that {@link augmentWithAtoms} would regenerate are omitted. Throws if a
 * part value is longer than `0xffff` bytes.
 * @param {PartDictionary} dict - The dictionary to serialize.
 * @returns {Uint8Array} The encoded byte buffer.
 *
 * @example
 * const dict = new PartDictionary();
 * dict.add(Kind.Whole, "hi");
 * const bytes = saveDictBinary(dict);
 * // First four bytes are the "BDI2" magic:
 * Array.from(bytes.slice(0, 4));   // → [0x42, 0x44, 0x49, 0x32]
 * // Round-trips back to the same trained parts:
 * loadDictBinary(bytes).hasWhole("hi");   // → true
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
 * @function loadDictBinary
 * @description Parses the trained-only binary `BDI2` format produced by
 * {@link saveDictBinary} into a new {@link PartDictionary}, then calls
 * {@link augmentWithAtoms} to restore the omitted singleton/special atoms.
 * Throws on a bad magic, an unsupported version, an out-of-range kind byte, or
 * a truncated record.
 * @param {Uint8Array} bytes - The encoded byte buffer.
 * @returns {PartDictionary} A dictionary containing the parsed parts plus the
 *   regenerated atoms.
 * @throws {Error} If the magic is not `"BDI2"`, the version is unsupported, a
 *   kind byte exceeds `5`, or a record is truncated.
 *
 * @example
 * const dict = new PartDictionary();
 * dict.add(Kind.Whole, "hi");
 * const reloaded = loadDictBinary(saveDictBinary(dict));
 * reloaded.hasWhole("hi");   // → true
 * reloaded.hasWhole("a");    // → true  (restored by augmentWithAtoms)
 *
 * @example
 * loadDictBinary(Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]));
 * // → throws Error: loadDictBinary: bad magic (expected BDI2)
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