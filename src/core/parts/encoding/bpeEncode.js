"use strict";

/**
 * @file bpeEncode.js
 * @brief The structural (BPE-style) word encoder → `[L, C, R]`.
 *
 * A strict, discriminative-before-generative peel producing a *non-redundant*
 * segmentation grouped into positional components: `start` (the one Start), `end`
 * (the one End), and `mid` (the interior Mids). It proceeds in three passes:
 *
 *   1. **Whole** — if the word is a registered Whole, emit it and stop.
 *   2. **Boundary pair** — pick the (Start, End) pair (across lengths) that keeps
 *      BOTH boundaries anchored, preferring discriminative over generative, then
 *      coverage, then balance. Choosing the pair together avoids a long End
 *      swallowing the Start slot (`dream → dre·am`, not `d·ream`), which is what
 *      keeps the lone-letter rate down.
 *   3. **Interior Mids by level** — for `L = longest … 2`, take the
 *      discriminative Mids of that length, then generative, then untagged; then
 *      `L-1`. Any position still uncovered falls to a length-1 Start/Mid/End atom
 *      by position.
 *
 * Strict positions (Start = proper prefix, End = proper suffix with `L < n`, Mid
 * strict interior) keep a part from ever covering a whole word, and since a Start
 * always sits at offset 0 and an End at the suffix, the peel yields at most one of
 * each — so `start`/`end` are well-defined and a leftover boundary char becomes
 * the length-1 `start`/`end` itself. Runs on *canonical* words
 * (post-normalization), so it need not be typo-robust.
 */

import { Kind, kInvalidPartId, kMaxPartLength, positionalKind } from "../kind.js";

const INV = kInvalidPartId;

/**
 * @typedef {Object} EncodedPart
 * @property {number} kind - The part {@link Kind}.
 * @property {string} value - The part value (byte-string).
 * @property {number} id - The dictionary part id.
 * @property {number} pos - Inclusive start offset within the word.
 * @property {number} [L] - Span length.
 */

/**
 * @typedef {Object} Peel
 * @property {EncodedPart|null} whole - Set (others null/empty) when the word is a Whole atom.
 * @property {EncodedPart|null} start - The Start part (prefix), or `null`.
 * @property {EncodedPart[]} mid - Interior Mids (incl. length-1 Mid fills), in position order.
 * @property {EncodedPart|null} end - The End part (suffix), or `null`.
 * @property {EncodedPart[]} parts - The full decomposition in position order.
 *
 * NOTE: these are `Start`/`Mid`/`End` — the word's own segmentation. They are
 * NOT the chunk-contextual `[L, C, R]` (past/current/future F-pools), which are a
 * different object built later from the sequence of F's (see the signature stage).
 */

/**
 * @function bpeEncode
 * @description Peels `word` into its `[L, C, R]` structural signature.
 *
 * @param {import("../dictionary.js").PartDictionary} dict - Dictionary of parts
 *   (learned + bigram backstop + single-char atoms).
 * @param {string} word - The (canonical, lowercased byte-string) word to encode.
 * @param {Object<string, string>} [tracks={}] - Optional map from `"kind|value"`
 *   to a track tag (`"d"` discriminative, `"g"` generative). Drives the
 *   discriminative-before-generative tiebreak; anything untagged (e.g. the
 *   bigram backstop) sorts last.
 * @returns {Peel} The `start`/`mid`/`end` segmentation.
 *
 * @example
 * bpeEncode(dict, "running", tracks);
 * // → { whole:null, start:{value:"ru",..}, mid:[], end:{value:"nning",..}, parts:[..] }
 * bpeEncode(dict, "the", tracks);
 * // → { whole:{value:"the",..}, start:null, mid:[], end:null, parts:[{value:"the",..}] }
 */
export const bpeEncode = (dict, word, tracks = dict.tracks || {}) => {
  const trackPri = (kind, value) => { const t = tracks[kind + "|" + value]; return t === "d" ? 0 : t === "g" ? 1 : 2; };
  const n = word.length;
  if (n === 0) return { whole: null, start: null, mid: [], end: null, parts: [] };

  // Pass 0: the Whole part.
  if (dict.hasWhole(word)) {
    const p = { kind: Kind.Whole, value: word, id: dict.lookup(Kind.Whole, word), pos: 0, L: n };
    return { whole: p, start: null, mid: [], end: null, parts: [p] };
  }

  const claimed = new Array(n).fill(false);
  const chosen = [];
  const free = (pos, L) => { for (let i = pos; i < pos + L; i++) if (claimed[i]) return false; return true; };
  const take = (pos, L, kind, value, id) => { for (let i = pos; i < pos + L; i++) claimed[i] = true; chosen.push({ pos, L, kind, value, id }); };

  // Pass 1: boundary PAIR. Enumerate every registered Start prefix and End suffix
  // (plus the "none" option) and pick the non-overlapping (Start, End) pair by:
  // both boundaries anchored → discriminative over generative → most coverage →
  // balanced → longer Start. Choosing the pair TOGETHER (across lengths, not one
  // greedy longest affix) is what keeps both ends anchored — a long End can't
  // swallow the Start slot and strand a lone leading letter (dream → dre·am, not
  // d·ream). The "none" side carries a track penalty (3) so a real both-anchored
  // pair outranks a one-sided affix.
  const starts = [{ a: 0, tp: 3 }], ends = [{ b: 0, tp: 3 }];
  for (let L = 2; L <= kMaxPartLength && L < n; L++) {
    const sv = word.slice(0, L); let id = dict.lookup(Kind.Start, sv); if (id !== INV) starts.push({ a: L, id, value: sv, tp: trackPri(Kind.Start, sv) });
    const ev = word.slice(n - L, n); id = dict.lookup(Kind.End, ev); if (id !== INV) ends.push({ b: L, id, value: ev, tp: trackPri(Kind.End, ev) });
  }
  let best = { both: -1, tp: 99, cover: -1, bal: -99, sa: -1, s: starts[0], e: ends[0] };
  for (const s of starts) for (const e of ends) {
    if (s.a + e.b > n) continue;
    const both = s.a > 0 && e.b > 0 ? 1 : 0, tp = s.tp + e.tp, cover = s.a + e.b, bal = -Math.abs(s.a - e.b);
    if (both > best.both ||
      (both === best.both && (tp < best.tp ||
        (tp === best.tp && (cover > best.cover ||
          (cover === best.cover && (bal > best.bal ||
            (bal === best.bal && s.a > best.sa)))))))) best = { both, tp, cover, bal, sa: s.a, s, e };
  }
  if (best.s.a > 0) take(0, best.s.a, Kind.Start, best.s.value, best.s.id);
  if (best.e.b > 0) take(n - best.e.b, best.e.b, Kind.End, best.e.value, best.e.id);

  // Pass 2: interior Mids by level — length L longest → shortest, and within a
  // level discriminative (0) before generative (1) before untagged backstop (2),
  // left-to-right. Each is taken only if its span is still free.
  for (let L = Math.min(kMaxPartLength, n - 1); L >= 2; L--) {
    for (let pri = 0; pri <= 2; pri++) {
      for (let p = 1; p + L <= n - 1; p++) { if (!free(p, L)) continue; const v = word.slice(p, p + L), id = dict.lookup(Kind.Mid, v); if (id !== INV && trackPri(Kind.Mid, v) === pri) take(p, L, Kind.Mid, v, id); }
    }
  }

  // Level 1: positional single-char atoms fill every remaining position. A Start
  // only ever sits at offset 0 and an End at n-1, so there is at most one of each.
  let hs = chosen.some((c) => c.kind === Kind.Start);
  let he = chosen.some((c) => c.kind === Kind.End);
  for (let p = 0; p < n; p++) if (!claimed[p]) {
    const k = positionalKind(p === 0 && !hs, p === n - 1 && !he);
    if (k === Kind.Start) hs = true; else if (k === Kind.End) he = true;
    chosen.push({ pos: p, L: 1, kind: k, value: word[p], id: dict.lookup(k, word[p]) });
  }
  chosen.sort((a, b) => a.pos - b.pos);

  return {
    whole: null,
    start: chosen.find((c) => c.kind === Kind.Start) || null,
    mid: chosen.filter((c) => c.kind === Kind.Mid),
    end: chosen.find((c) => c.kind === Kind.End) || null,
    parts: chosen,
  };
};

export default bpeEncode;
