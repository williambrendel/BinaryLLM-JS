"use strict";

// ============================================================================
// __tests__/core/parts/generation/generateParts.test.js
//
// Mirrors src/core/parts/generation/generateParts.js — the four-track
// orchestrator. Verifies the end-to-end shape on a tiny inline corpus: the
// {dict, tracks, stats} result, the stats fields, that every tracked learned
// part has a consistent w/d/g tag, that common words land as wholes, and that
// the augment atoms (letters) are present.
// ============================================================================

import { generateParts } from "../../../../src/core/parts/generation/generateParts.js";
import { wordFreq } from "../../../../src/utilities/corpus/wordFreq.js";
import { Kind } from "../../../../src/core/parts/kind.js";

// A small deterministic corpus with a dominant function word ("the") and a
// family of morphologically related words (run/jump/sing + -ing/-s/-ed). The
// extra length-3 distractors (cat/dog/fox/pen/cup) give the length-3 whole
// bucket ≥3 members so the elbow criterion can isolate "the" as a clean head.
const CORPUS =
  "the the the the the the the the running runner runs jumping jumped jumps " +
  "singing sings reading reads walking walked cat dog fox pen cup";

const buildResult = () => {
  const buf = new TextEncoder().encode(CORPUS);
  return generateParts(wordFreq([buf]));
};

describe("generateParts — result shape", () => {
  test("returns {dict, tracks, stats} with the documented stats fields", () => {
    const { dict, tracks, stats } = buildResult();

    expect(dict).toBeDefined();
    expect(tracks).toBeDefined();
    expect(stats).toBeDefined();

    expect(typeof stats.words).toBe("number");
    expect(typeof stats.wholes).toBe("number");
    expect(typeof stats.generative).toBe("number");
    expect(typeof stats.discriminative).toBe("number");

    expect(stats.words).toBeGreaterThan(0);
    // words count equals the number of distinct word tokens.
    const distinct = wordFreq([new TextEncoder().encode(CORPUS)]).size;
    expect(stats.words).toBe(distinct);
  });
});

describe("generateParts — tracks tagging", () => {
  test("every tracked learned part has a valid w/d/g tag (no '?')", () => {
    const { tracks } = buildResult();

    const keys = Object.keys(tracks);
    expect(keys.length).toBeGreaterThan(0);
    for (const key of keys) {
      const tag = tracks[key];
      expect(["w", "d", "g"]).toContain(tag);
    }
  });

  test("track keys are `${kind}|value` for TRACKED kinds with value length ≥ 2", () => {
    const { tracks, dict } = buildResult();

    for (const key of Object.keys(tracks)) {
      const kind = key.charCodeAt(0) - 48;
      const value = key.slice(key.indexOf("|") + 1);
      expect([Kind.Whole, Kind.Start, Kind.Mid, Kind.End]).toContain(kind);
      expect(value.length).toBeGreaterThanOrEqual(2);
      // The key must name a part that actually lives in the dict.
      expect(dict.lookup(kind, value)).not.toBe(0xffffffff);
    }
  });

  test("Whole parts are tagged 'w'", () => {
    const { tracks } = buildResult();
    for (const key of Object.keys(tracks)) {
      const kind = key.charCodeAt(0) - 48;
      if (kind === Kind.Whole) expect(tracks[key]).toBe("w");
    }
  });
});

describe("generateParts — dictionary content", () => {
  test('the dominant function word "the" is captured as a whole', () => {
    const { dict } = buildResult();
    expect(dict.hasWhole("the")).toBe(true);
  });

  test("augment atoms are present (length-1 fragment singletons seeded)", () => {
    const { dict } = buildResult();
    // The three positional length-1 forms for a common char exist post-augment
    // (bare value; position is the kind).
    expect(dict.lookup(Kind.Start, "a")).not.toBe(0xffffffff);
    expect(dict.lookup(Kind.Mid, "a")).not.toBe(0xffffffff);
    expect(dict.lookup(Kind.End, "a")).not.toBe(0xffffffff);
  });

  test("stats.wholes/generative/discriminative are non-negative and consistent", () => {
    const { dict, stats } = buildResult();

    expect(stats.wholes).toBeGreaterThanOrEqual(0);
    expect(stats.generative).toBeGreaterThanOrEqual(0);
    expect(stats.discriminative).toBeGreaterThanOrEqual(0);
    // At least the "the" whole and some morpheme structure were learned.
    expect(stats.wholes).toBeGreaterThan(0);
    // The dict holds strictly more than just the learned parts (atoms added).
    expect(dict.size()).toBeGreaterThan(stats.wholes + stats.generative);
  });
});
