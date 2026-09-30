#!/usr/bin/env python3
"""
Build a RAG knowledge base for the braille tutor.

Output: training/ueb_knowledge.json
Contains every UEB symbol, contraction, and rule with accurate dot patterns,
Unicode codepoints, and explanations. Used at inference time to inject
accurate facts into the LLM's context window.
"""

import json

def dots(*ds):
    """Build a braille character from dot numbers (1-8)."""
    bits = 0
    for d in ds:
        bits |= 1 << (d - 1)
    return chr(0x2800 + bits)

def dot_str(cell):
    """Return human-readable dot numbers for a braille cell, e.g. 'dots 1-2-4'."""
    if len(cell) != 1 or ord(cell) < 0x2800 or ord(cell) > 0x28FF:
        return None
    bits = ord(cell) - 0x2800
    ds = [i + 1 for i in range(8) if bits & (1 << i)]
    if not ds:
        return "empty cell"
    return "dots " + "-".join(str(d) for d in ds)

def multi_dot_str(braille):
    """Return dot descriptions for multi-cell braille."""
    parts = []
    for ch in braille:
        d = dot_str(ch)
        if d:
            parts.append(d)
    return " / ".join(parts)

kb = []

# ── Letters ──────────────────────────────────────────────────────────────
LETTERS = {
    "a": (1,), "b": (1,2), "c": (1,4), "d": (1,4,5), "e": (1,5),
    "f": (1,2,4), "g": (1,2,4,5), "h": (1,2,5), "i": (2,4), "j": (2,4,5),
    "k": (1,3), "l": (1,2,3), "m": (1,3,4), "n": (1,3,4,5), "o": (1,3,5),
    "p": (1,2,3,4), "q": (1,2,3,4,5), "r": (1,2,3,5), "s": (2,3,4),
    "t": (2,3,4,5), "u": (1,3,6), "v": (1,2,3,6), "w": (2,4,5,6),
    "x": (1,3,4,6), "y": (1,3,4,5,6), "z": (1,3,5,6),
}

# Pattern explanations
ROW_INFO = {
    "a": "row 1 (dots 1-2-4-5 only)", "b": "row 1", "c": "row 1",
    "d": "row 1", "e": "row 1",
    "f": "row 1 pattern shifted (same as a-e but with dot 4 added instead of dot 5)",
    "g": "row 2", "h": "row 2", "i": "row 2", "j": "row 2",
    "k": "row 3 (add dot 3 to a-e)", "l": "row 3", "m": "row 3",
    "n": "row 3", "o": "row 3",
    "p": "row 4 (add dots 3+4 to a-e)", "q": "row 4", "r": "row 4",
    "s": "row 4", "t": "row 4",
    "u": "row 5 (add dot 6 to k-o)", "v": "row 5", "x": "row 5",
    "y": "row 5", "z": "row 5",
    "w": "exception -- w was not in the original French alphabet Louis Braille used",
}

for letter, d in LETTERS.items():
    cell = dots(*d)
    dot_nums = "dots " + "-".join(str(x) for x in d)
    # Also make dash-separated dot pattern for reverse lookup: "1-2-5", "dots 1-2-5"
    dot_key = "-".join(str(x) for x in d)
    row = ROW_INFO.get(letter, "")
    kb.append({
        "type": "letter",
        "id": f"letter_{letter}",
        "keywords": [letter, f"letter {letter}", "letter", "alphabet", "letters", dot_key, dot_nums, cell],
        "fact": f"The letter '{letter}' in UEB braille is {cell} ({dot_nums}). {row}".strip(),
    })

# ── Digits ───────────────────────────────────────────────────────────────
DIGIT_LETTERS = "abcdefghij"
for digit in range(10):
    idx = (digit - 1) % 10  # 1->a, 2->b, ..., 0->j
    letter = DIGIT_LETTERS[idx]
    d = LETTERS[letter]
    cell = dots(*d)
    num_ind = dots(3,4,5,6)
    dot_nums = "dots " + "-".join(str(x) for x in d)
    kb.append({
        "type": "number",
        "id": f"number_{digit}",
        "keywords": [str(digit), f"number {digit}", "number", "numbers", "digit", "digits", "numeric"],
        "fact": f"The digit {digit} in UEB is {num_ind}{cell} (numeric indicator {dot_str(num_ind)}, then {dot_nums}). Numbers use the same cell as letter '{letter}' but preceded by the numeric indicator. The numeric indicator is {num_ind} ({dot_str(num_ind)}). Once a numeric indicator appears, all following a-j cells are read as digits until a space, letter sign, or non-digit appears.",
    })

# Number system overview
kb.append({
    "type": "concept",
    "id": "number_system",
    "keywords": ["number", "numbers", "digit", "numeric", "number system", "how numbers work", "counting"],
    "fact": f"UEB numbers: digits 1-9 and 0 use the same cells as letters a-j. They are preceded by the numeric indicator {dots(3,4,5,6)} ({dot_str(dots(3,4,5,6))}). For example, 42 is {dots(3,4,5,6)}{dots(1,4,5)}{dots(1,2)} (numeric indicator + d + b). Multi-digit numbers don't need a new indicator for each digit. The letter sign {dots(5,6)} ({dot_str(dots(5,6))}) is used after a number if the next character is a-j to prevent it from being read as a digit.",
})

# ── Punctuation ──────────────────────────────────────────────────────────
PUNCTUATION = {
    "period": (".", (2,5,6)),
    "full stop": (".", (2,5,6)),
    "comma": (",", (2,)),
    "semicolon": (";", (2,3)),
    "colon": (":", (2,5)),
    "exclamation mark": ("!", (2,3,5)),
    "question mark": ("?", (2,3,6)),
    "apostrophe": ("'", (3,)),
    "hyphen": ("-", (3,6)),
}
for name, (char, d) in PUNCTUATION.items():
    cell = dots(*d)
    dot_nums = "dots " + "-".join(str(x) for x in d)
    kb.append({
        "type": "punctuation",
        "id": f"punct_{name.replace(' ', '_')}",
        "keywords": [name, char, "punctuation", "symbol"],
        "fact": f"The {name} ({char}) in UEB is {cell} ({dot_nums}).",
    })

# ── Indicators ───────────────────────────────────────────────────────────
kb.append({
    "type": "indicator",
    "id": "capital_letter",
    "keywords": ["capital", "uppercase", "capitalization", "capital letter", "indicator"],
    "fact": f"Capital letter indicator: {dots(6)} ({dot_str(dots(6))}). Placed before a single uppercase letter. For two consecutive capitals, use capital word indicator: {dots(6)}{dots(6)} (two cells of {dot_str(dots(6))}). For three or more consecutive capitals, use capital passage indicator: {dots(6)}{dots(6)}{dots(6)} (three cells), and end with capital terminator {dots(6)}{dots(3)} ({dot_str(dots(6))} + {dot_str(dots(3))}).",
})

kb.append({
    "type": "indicator",
    "id": "numeric_indicator",
    "keywords": ["numeric indicator", "number sign", "number indicator"],
    "fact": f"Numeric indicator: {dots(3,4,5,6)} ({dot_str(dots(3,4,5,6))}). Placed before digits. It signals that the following a-j cells should be read as 1-9,0 instead of letters.",
})

kb.append({
    "type": "indicator",
    "id": "grade1_indicator",
    "keywords": ["grade 1 indicator", "letter sign", "grade 1 symbol"],
    "fact": f"Grade 1 indicator (letter sign): {dots(5,6)} ({dot_str(dots(5,6))}). Used to indicate that the next cell is a letter, not a number or contraction. For example, after a number, to show that the next cell is a letter a-j rather than a digit.",
})

# ── Alphabetic Wordsigns (Grade 2) ───────────────────────────────────────
ALPHA_WORDSIGNS = {
    "but": "b", "can": "c", "do": "d", "every": "e", "from": "f",
    "go": "g", "have": "h", "just": "j", "knowledge": "k", "like": "l",
    "more": "m", "not": "n", "people": "p", "quite": "q", "rather": "r",
    "so": "s", "that": "t", "us": "u", "very": "v", "will": "w",
    "it": "x", "you": "y", "as": "z",
}
for word, letter in ALPHA_WORDSIGNS.items():
    d = LETTERS[letter]
    cell = dots(*d)
    dot_nums = "dots " + "-".join(str(x) for x in d)
    kb.append({
        "type": "alphabetic_wordsign",
        "id": f"aws_{word}",
        "keywords": [word, "wordsign", "grade 2", "contraction", "alphabetic wordsign"],
        "fact": f"Alphabetic wordsign: '{word}' = {cell} (letter {letter}, {dot_nums}). This is the same cell as the letter '{letter}'. It can only be used when it stands alone (surrounded by spaces or punctuation). In Grade 2 (contracted) braille, the single letter {letter} represents the whole word '{word}'.",
    })

# ── Strong Wordsigns ─────────────────────────────────────────────────────
STRONG_WORDSIGNS = {
    "child": (1,6), "shall": (1,4,6), "this": (1,4,5,6),
    "which": (1,5,6), "out": (1,2,5,6), "still": (3,4),
}
for word, d in STRONG_WORDSIGNS.items():
    cell = dots(*d)
    dot_nums = "dots " + "-".join(str(x) for x in d)
    kb.append({
        "type": "strong_wordsign",
        "id": f"sws_{word}",
        "keywords": [word, "wordsign", "strong wordsign", "grade 2", "contraction"],
        "fact": f"Strong wordsign: '{word}' = {cell} ({dot_nums}). Standing alone only. These signs use dots in both upper and lower halves of the cell.",
    })

# ── Strong Contractions ──────────────────────────────────────────────────
STRONG_CONTRACTIONS = {
    "and": (1,2,3,4,6), "for": (1,2,3,4,5,6), "of": (1,2,3,5,6),
    "the": (2,3,4,6), "with": (2,3,4,5,6),
}
for word, d in STRONG_CONTRACTIONS.items():
    cell = dots(*d)
    dot_nums = "dots " + "-".join(str(x) for x in d)
    kb.append({
        "type": "strong_contraction",
        "id": f"sc_{word}",
        "keywords": [word, "contraction", "strong contraction", "grade 2"],
        "fact": f"Strong contraction: '{word}' = {cell} ({dot_nums}). Can be used as a whole word (standing alone) or as part of a word (groupsign). For example, 'band' = b + {cell} (b + and), 'fortune' = {dots(1,2,3,4,5,6)} + une (for + une).",
    })

# ── Strong Groupsigns ────────────────────────────────────────────────────
STRONG_GROUPSIGNS = {
    "ch": (1,6), "gh": (1,2,6), "sh": (1,4,6), "th": (1,4,5,6),
    "wh": (1,5,6), "ed": (1,2,4,6), "er": (1,2,4,5,6), "ou": (1,2,5,6),
    "ow": (2,4,6), "st": (3,4), "ar": (3,4,5), "ing": (3,4,6),
}
for combo, d in STRONG_GROUPSIGNS.items():
    cell = dots(*d)
    dot_nums = "dots " + "-".join(str(x) for x in d)
    kb.append({
        "type": "strong_groupsign",
        "id": f"sg_{combo}",
        "keywords": [combo, "groupsign", "strong groupsign", "grade 2", "contraction"],
        "fact": f"Strong groupsign: '{combo}' = {cell} ({dot_nums}). Used within words to shorten common letter combinations. Cannot bridge syllable boundaries after aspirated h (e.g., 'mishap' does not use the 'sh' groupsign because sh bridges mis-hap).",
    })

# ── Lower Wordsigns ──────────────────────────────────────────────────────
LOWER_WORDSIGNS = {
    "be": (2,3), "enough": (2,6), "were": (2,3,5,6),
    "his": (2,3,6), "was": (3,5,6), "in": (3,5),
}
for word, d in LOWER_WORDSIGNS.items():
    cell = dots(*d)
    dot_nums = "dots " + "-".join(str(x) for x in d)
    kb.append({
        "type": "lower_wordsign",
        "id": f"lws_{word}",
        "keywords": [word, "wordsign", "lower wordsign", "grade 2", "contraction"],
        "fact": f"Lower wordsign: '{word}' = {cell} ({dot_nums}). Standing alone only. Uses only lower dots (dots 2-3-5-6). Subject to the lower sign rule: cannot be adjacent to punctuation that also uses only lower dots.",
    })

# ── Lower Groupsigns ────────────────────────────────────────────────────
LOWER_GROUPSIGNS = {
    "ea": (2,), "bb": (2,3), "cc": (2,5), "ff": (2,3,5), "gg": (2,3,5,6),
    "en": (2,6), "in": (3,5),
}
# Prefixes
LOWER_PREFIXES = {
    "be": (2,3), "con": (2,5), "dis": (2,5,6),
}
for combo, d in LOWER_GROUPSIGNS.items():
    cell = dots(*d)
    dot_nums = "dots " + "-".join(str(x) for x in d)
    kb.append({
        "type": "lower_groupsign",
        "id": f"lg_{combo}",
        "keywords": [combo, "groupsign", "lower groupsign", "grade 2"],
        "fact": f"Lower groupsign: '{combo}' = {cell} ({dot_nums}). Used within words. Subject to bridging restrictions.",
    })
for prefix, d in LOWER_PREFIXES.items():
    cell = dots(*d)
    dot_nums = "dots " + "-".join(str(x) for x in d)
    kb.append({
        "type": "lower_groupsign",
        "id": f"lg_prefix_{prefix}",
        "keywords": [prefix, "prefix", "groupsign", "lower groupsign", "grade 2"],
        "fact": f"Lower groupsign prefix: '{prefix}' = {cell} ({dot_nums}). Used at the beginning of words as a contraction.",
    })

# ── Initial-Letter Contractions ──────────────────────────────────────────
INITIAL_DOT5 = {
    "day": (5, "d"), "ever": (5, "e"), "father": (5, "f"),
    "here": (5, "h"), "know": (5, "k"), "lord": (5, "l"),
    "mother": (5, "m"), "name": (5, "n"), "one": (5, "o"),
    "part": (5, "p"), "question": (5, "q"), "right": (5, "r"),
    "some": (5, "s"), "time": (5, "t"), "under": (5, "u"),
    "young": (5, "y"), "there": (5, "the"), "character": (5, "ch"),
    "through": (5, "th"), "where": (5, "wh"), "ought": (5, "ou"),
    "work": (5, "w"),
}
INITIAL_DOTS45 = {
    "upon": (45, "u"), "word": (45, "w"), "these": (45, "the"),
    "those": (45, "th"), "whose": (45, "wh"),
}
INITIAL_DOTS456 = {
    "cannot": (456, "c"), "had": (456, "h"), "many": (456, "m"),
    "spirit": (456, "s"), "world": (456, "w"), "their": (456, "the"),
}

def make_initial(prefix_dots, second_key):
    """Build the braille for an initial-letter contraction."""
    if prefix_dots == 5:
        p = dots(5)
        prefix_str = "dot 5"
    elif prefix_dots == 45:
        p = dots(4, 5)
        prefix_str = "dots 4-5"
    else:
        p = dots(4, 5, 6)
        prefix_str = "dots 4-5-6"
    # Second part: if it's a groupsign name, use that; otherwise it's a letter
    GS = {"the": (2,3,4,6), "th": (1,4,5,6), "ch": (1,6), "wh": (1,5,6), "ou": (1,2,5,6)}
    if second_key in GS:
        s = dots(*GS[second_key])
        second_str = f"'{second_key}' groupsign"
    else:
        s = dots(*LETTERS[second_key])
        second_str = f"letter '{second_key}'"
    return p + s, f"{prefix_str} + {second_str}"

for word, (prefix, second) in INITIAL_DOT5.items():
    braille, desc = make_initial(prefix, second)
    kb.append({
        "type": "initial_letter_contraction",
        "id": f"ilc_{word}",
        "keywords": [word, "initial-letter contraction", "contraction", "grade 2"],
        "fact": f"Initial-letter contraction: '{word}' = {braille} ({desc}). Two-cell sign. Can represent the whole word or appear as part of a longer word.",
    })
for word, (prefix, second) in INITIAL_DOTS45.items():
    braille, desc = make_initial(prefix, second)
    kb.append({
        "type": "initial_letter_contraction",
        "id": f"ilc_{word}",
        "keywords": [word, "initial-letter contraction", "contraction", "grade 2"],
        "fact": f"Initial-letter contraction: '{word}' = {braille} ({desc}). Two-cell sign.",
    })
for word, (prefix, second) in INITIAL_DOTS456.items():
    braille, desc = make_initial(prefix, second)
    kb.append({
        "type": "initial_letter_contraction",
        "id": f"ilc_{word}",
        "keywords": [word, "initial-letter contraction", "contraction", "grade 2"],
        "fact": f"Initial-letter contraction: '{word}' = {braille} ({desc}). Two-cell sign.",
    })

# ── Final-Letter Groupsigns ──────────────────────────────────────────────
FINAL_46 = {
    "ound": (4,6, 1,4,5), "ance": (4,6, 1,5), "sion": (4,6, 1,3,4,5),
    "less": (4,6, 2,3,4), "ount": (4,6, 2,3,4,5),
}
FINAL_56 = {
    "ence": (5,6, 1,5), "ong": (5,6, 1,2,4,5), "ful": (5,6, 1,2,3),
    "tion": (5,6, 1,3,4,5), "ness": (5,6, 2,3,4), "ment": (5,6, 2,3,4,5),
    "ity": (5,6, 1,3,4,5,6),
}

for suffix, d in FINAL_46.items():
    prefix_d = d[:2]
    second_d = d[2:]
    braille = dots(*prefix_d) + dots(*second_d)
    kb.append({
        "type": "final_letter_groupsign",
        "id": f"flg_{suffix}",
        "keywords": [suffix, "final-letter groupsign", "groupsign", "grade 2", "suffix"],
        "fact": f"Final-letter groupsign: '-{suffix}' = {braille} (dots {prefix_d[0]}-{prefix_d[1]} + {dot_str(dots(*second_d))}). Used at the end of words.",
    })
for suffix, d in FINAL_56.items():
    prefix_d = d[:2]
    second_d = d[2:]
    braille = dots(*prefix_d) + dots(*second_d)
    kb.append({
        "type": "final_letter_groupsign",
        "id": f"flg_{suffix}",
        "keywords": [suffix, "final-letter groupsign", "groupsign", "grade 2", "suffix"],
        "fact": f"Final-letter groupsign: '-{suffix}' = {braille} (dots {prefix_d[0]}-{prefix_d[1]} + {dot_str(dots(*second_d))}). Used at the end of words.",
    })

# ── Shortforms ───────────────────────────────────────────────────────────
SHORTFORMS = [
    ("about", "ab"), ("above", "abv"), ("according", "ac"), ("across", "acr"),
    ("after", "af"), ("afternoon", "afn"), ("afterward", "afw"), ("again", "ag"),
    ("almost", "alm"), ("already", "alr"), ("also", "al"), ("altogether", "alt"),
    ("always", "alw"), ("blind", "bl"), ("braille", "brl"), ("children", "chn"),
    ("could", "cd"), ("deceive", "dcv"), ("declare", "dcl"), ("either", "ei"),
    ("first", "fst"), ("friend", "fr"), ("good", "gd"), ("great", "grt"),
    ("him", "hm"), ("himself", "hmf"), ("immediate", "imm"), ("its", "xs"),
    ("itself", "xf"), ("letter", "lr"), ("little", "ll"), ("much", "mch"),
    ("must", "mst"), ("myself", "myf"), ("necessary", "nec"), ("neither", "nei"),
    ("paid", "pd"), ("perhaps", "prh"), ("quick", "qk"), ("receive", "rcv"),
    ("rejoice", "rjc"), ("said", "sd"), ("such", "sch"), ("today", "td"),
    ("together", "tgr"), ("tomorrow", "tm"), ("tonight", "tn"), ("would", "wd"),
    ("your", "yr"), ("yourself", "yrf"), ("yourselves", "yrvs"),
]
for word, abbrev in SHORTFORMS:
    kb.append({
        "type": "shortform",
        "id": f"sf_{word}",
        "keywords": [word, abbrev, "shortform", "grade 2", "abbreviation"],
        "fact": f"Shortform: '{word}' is written as '{abbrev}' in Grade 2 braille. Shortforms are abbreviated spellings that save space. They can be used as whole words and within compound words.",
    })

# ── Concepts / Rules ────────────────────────────────────────────────────
concepts = [
    {
        "id": "braille_cell",
        "keywords": ["cell", "braille cell", "dots", "how many dots", "6 dots", "8 dots", "dot positions"],
        "fact": "A standard braille cell has 6 dots arranged in a 2x3 grid. Dots are numbered: dot 1 (top-left), dot 2 (middle-left), dot 3 (bottom-left), dot 4 (top-right), dot 5 (middle-right), dot 6 (bottom-right). Computer braille (8-dot) adds dot 7 (below dot 3) and dot 8 (below dot 6), giving 256 possible patterns. Standard 6-dot braille has 64 possible patterns (including the blank cell).",
    },
    {
        "id": "grade_1_vs_2",
        "keywords": ["grade 1", "grade 2", "uncontracted", "contracted", "difference", "grades"],
        "fact": "Grade 1 (uncontracted) braille spells every word letter by letter. Grade 2 (contracted) braille uses ~200 contractions and shortforms to reduce the number of cells. Grade 2 is standard for published English braille. Grade 1 is used for beginners, technical notation, and when contractions would be ambiguous.",
    },
    {
        "id": "louis_braille",
        "keywords": ["Louis Braille", "inventor", "history", "invented", "who invented", "origin"],
        "fact": "Louis Braille invented the braille system in 1824 at age 15 while a student at the Royal Institute for Blind Youth in Paris. He adapted Charles Barbier's 12-dot 'night writing' military code into a 6-dot system. Braille lost his sight at age 3 from an accident in his father's leather workshop. The system was not officially adopted in France until 1854, two years after his death.",
    },
    {
        "id": "ueb",
        "keywords": ["UEB", "Unified English Braille", "standard", "what is UEB"],
        "fact": "Unified English Braille (UEB) is the standard braille code for English, adopted in 2004-2013 across English-speaking countries. It replaced older national codes (BAUK, BANA) with a single unified system. UEB handles literary text, math, and technical notation within one code. The definitive reference is 'The Rules of Unified English Braille' published by ICEB (International Council on English Braille).",
    },
    {
        "id": "standing_alone",
        "keywords": ["standing alone", "wordsign rule", "when to use wordsign"],
        "fact": "In UEB, a wordsign can only be used when the word 'stands alone' -- meaning it is surrounded by spaces, punctuation, or the start/end of text. An apostrophe followed by common suffixes ('d, 'll, 're, 's, 't, 've) is also allowed. For example, the wordsign for 'but' (letter b) can be used in 'but' and 'but's' but NOT in 'butter' (where b-u-t-t-er would be spelled out or use groupsigns).",
    },
    {
        "id": "lower_sign_rule",
        "keywords": ["lower sign rule", "lower dots", "lower wordsign", "lower groupsign"],
        "fact": "The lower sign rule: lower wordsigns (be, enough, were, his, was, in) use only dots in the lower half of the cell (dots 2,3,5,6). They must NOT be placed adjacent to punctuation that also uses only lower dots (like comma, semicolon), because the reader couldn't tell where the punctuation ends and the wordsign begins.",
    },
    {
        "id": "contraction_types",
        "keywords": ["types of contractions", "contraction categories", "how many contractions"],
        "fact": "UEB Grade 2 has these contraction categories: (1) 23 alphabetic wordsigns -- single letters represent whole words. (2) 6 strong wordsigns -- unique cells for child, shall, this, which, out, still. (3) 5 strong contractions -- and, for, of, the, with -- usable anywhere. (4) 12 strong groupsigns -- ch, sh, th, wh, ed, er, ou, ow, st, ar, ing, gh. (5) 6 lower wordsigns. (6) 10 lower groupsigns. (7) 33 initial-letter contractions. (8) 12 final-letter groupsigns. (9) 75 shortforms. Total: ~200 contractions.",
    },
    {
        "id": "aspirated_h",
        "keywords": ["aspirated h", "bridging", "mishap", "sh rule", "th rule"],
        "fact": "Bridging restriction: groupsigns sh, th, ch, wh, gh must not bridge across a syllable boundary when the h starts the second syllable (aspirated h). For example: 'mishap' = m-i-s-h-a-p (NOT mi-sh-ap), 'pothole' = p-o-t-h-o-l-e (NOT po-th-ole). This is because the h belongs to the second syllable.",
    },
    {
        "id": "braille_display",
        "keywords": ["braille display", "refreshable", "braille reader", "hardware"],
        "fact": "A refreshable braille display is a device that renders braille characters using pins that raise and lower. Displays range from 14 to 80 cells. They connect to computers and phones via USB or Bluetooth. Contractions (Grade 2) reduce cells by 20-50%, which is significant when a display only shows 20-40 cells at a time.",
    },
    {
        "id": "reading_direction",
        "keywords": ["reading direction", "left to right", "how to read"],
        "fact": "Braille is read left to right, like print English. Fingers move across the line. Each cell is felt as a pattern of raised dots. Experienced braille readers can read 100-200 words per minute. Reading uses mainly the index fingers, with other fingers used for tracking the line.",
    },
]
for c in concepts:
    kb.append({
        "type": "concept",
        "id": c["id"],
        "keywords": c["keywords"],
        "fact": c["fact"],
    })

# ── Letter comparisons (common confusion pairs) ─────────────────────────
CONFUSIONS = [
    ("d", "f", "d is dots 1-4-5, f is dots 1-2-4. Both share dots 1 and 4, but d has dot 5 (middle-right) while f has dot 2 (middle-left)."),
    ("e", "i", "e is dots 1-5 (top-left + middle-right), i is dots 2-4 (middle-left + top-right). They are mirror images of each other."),
    ("h", "j", "h is dots 1-2-5, j is dots 2-4-5. h has dot 1 (top-left), j has dot 4 (top-right)."),
    ("d", "n", "d is dots 1-4-5, n is dots 1-3-4-5. n has everything d has plus dot 3."),
    ("m", "n", "m is dots 1-3-4, n is dots 1-3-4-5. n has everything m has plus dot 5."),
    ("s", "t", "s is dots 2-3-4, t is dots 2-3-4-5. t has everything s has plus dot 5."),
]
for a, b, explanation in CONFUSIONS:
    da = LETTERS[a]
    db = LETTERS[b]
    cell_a = dots(*da)
    cell_b = dots(*db)
    kb.append({
        "type": "comparison",
        "id": f"compare_{a}_{b}",
        "keywords": [a, b, f"difference between {a} and {b}", f"{a} vs {b}", "difference", "compare", "confusion"],
        "fact": f"Comparing '{a}' ({cell_a}, dots {'-'.join(str(x) for x in da)}) and '{b}' ({cell_b}, dots {'-'.join(str(x) for x in db)}): {explanation}",
    })

# Write out
output_path = "ueb_knowledge.json"
with open(output_path, "w") as f:
    json.dump(kb, f, indent=2, ensure_ascii=False)

print(f"Built {len(kb)} knowledge base entries -> {output_path}")
# Show category breakdown
from collections import Counter
types = Counter(e["type"] for e in kb)
for t, count in types.most_common():
    print(f"  {t}: {count}")
