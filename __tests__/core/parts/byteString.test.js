"use strict";

// ============================================================================
// __tests__/core/parts/byteString.test.js
//
// Mirrors src/core/parts/byteString.js — UTF-8 <-> latin1 "byte string"
// conversion used so multi-byte characters can be handled byte-wise by the
// tokenizer and dictionary.
// ============================================================================

import { toByteString, fromByteString } from "../../../src/core/parts/byteString.js";
import { tokenizeStream, StreamTokenType } from "../../../src/core/parts/tokenize.js";

describe("byteString", () => {
  test("round-trips UTF-8", () => {
    expect(fromByteString(toByteString("héllo — señor"))).toBe("héllo — señor");
  });

  test("multi-byte chars expand to their UTF-8 bytes", () => {
    expect(toByteString("é").length).toBe(2); // C3 A9
    expect(toByteString("abc").length).toBe(3);
  });

  test("non-ASCII punctuation tokenizes byte-wise", () => {
    // A curly em-dash "—" is 3 UTF-8 bytes, none of them word chars, so it
    // forms a delimiter run kept literally (as its bytes).
    const toks = tokenizeStream("a—b");
    expect(toks.map((t) => t.type)).toEqual([
      StreamTokenType.Word,
      StreamTokenType.Delimiter,
      StreamTokenType.Word,
    ]);
    expect(fromByteString(toks[1].value)).toBe("—");
  });
});
