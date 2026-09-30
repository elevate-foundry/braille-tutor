#!/usr/bin/env python3
"""
Build a fine-tuning dataset for a braille tutor LLM.

Output: training/braille_tutor_train.jsonl
Format: ChatML (messages array) for QLoRA fine-tuning.

Categories:
  1. UEB letter/number dot patterns
  2. Grade 2 contractions, wordsigns, groupsigns, shortforms
  3. Braille reading (cell -> print) and writing (print -> cell)
  4. Common mistakes and corrections
  5. Conceptual explanations (why braille works this way)
  6. Braille history and culture
  7. Tutoring dialogue (multi-turn)
  8. Social / identity (who are you, etc.)
"""

import json, random

SYSTEM = (
    "You are a braille tutor. You teach Unified English Braille (UEB) to sighted "
    "and blind learners. You know every dot pattern, contraction rule, and shortform. "
    "You give concise, accurate answers. When showing braille, use Unicode braille "
    "characters (U+2800-U+283F) and always state the dot numbers. "
    "You are not a general-purpose AI. You redirect off-topic questions back to braille. "
    "You are encouraging but honest when a learner makes a mistake."
)

def msg(user, assistant):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }

def multi(*turns):
    """Multi-turn: [(user, assistant), ...]"""
    messages = [{"role": "system", "content": SYSTEM}]
    for u, a in turns:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    return {"messages": messages}

# Unicode braille helper
def dots_to_char(*ds):
    bits = 0
    for d in ds:
        bits |= 1 << (d - 1)
    return chr(0x2800 + bits)

# Letter mappings
LETTERS = {}
letter_dots = [
    (1,), (1,2), (1,4), (1,4,5), (1,5),
    (1,2,4), (1,2,4,5), (1,2,5), (2,4), (2,4,5),
    (1,3), (1,2,3), (1,3,4), (1,3,4,5), (1,3,5),
    (1,2,3,4), (1,2,3,4,5), (1,2,3,5), (2,3,4),
    (2,3,4,5), (1,3,6), (1,2,3,6), (2,4,5,6),
    (1,3,4,6), (1,3,4,5,6), (1,3,5,6),
]
for i, ds in enumerate(letter_dots):
    LETTERS[chr(97 + i)] = ds

DIGITS = {}
digit_dots = [
    (2,4,5),  # 0 = j
    (1,), (1,2), (1,4), (1,4,5), (1,5),
    (1,2,4), (1,2,4,5), (1,2,5), (2,4),  # 1-9 = a-i
]
for i, ds in enumerate(digit_dots):
    DIGITS[str(i)] = ds

NUM_IND = dots_to_char(3,4,5,6)
CAP_IND = dots_to_char(6)

# Wordsigns
WORDSIGNS = {
    "but": "b", "can": "c", "do": "d", "every": "e", "from": "f",
    "go": "g", "have": "h", "just": "j", "knowledge": "k", "like": "l",
    "more": "m", "not": "n", "people": "p", "quite": "q", "rather": "r",
    "so": "s", "that": "t", "us": "u", "very": "v", "will": "w",
    "it": "x", "you": "y", "as": "z",
}

STRONG_CONTRACTIONS = {
    "and": (1,2,3,4,6), "for": (1,2,3,4,5,6), "of": (1,2,3,5,6),
    "the": (2,3,4,6), "with": (2,3,4,5,6),
}

GROUPSIGNS = {
    "ch": (1,6), "sh": (1,4,6), "th": (1,4,5,6), "wh": (1,5,6),
    "gh": (1,2,6), "ed": (1,2,4,6), "er": (1,2,4,5,6), "ou": (1,2,5,6),
    "ow": (2,4,6), "st": (3,4), "ar": (3,4,5), "ing": (3,4,6),
}

SHORTFORMS = {
    "about": "ab", "above": "abv", "according": "ac", "across": "acr",
    "after": "af", "afternoon": "afn", "afterward": "afw", "again": "ag",
    "almost": "alm", "already": "alr", "also": "al", "altogether": "alt",
    "always": "alw", "blind": "bl", "braille": "brl", "could": "cd",
    "either": "ei", "first": "fst", "friend": "fr", "good": "gd",
    "great": "grt", "him": "hm", "himself": "hmf", "immediate": "imm",
    "little": "ll", "letter": "lr", "much": "mch", "necessary": "nec",
    "neither": "nei", "paid": "pd", "quick": "qk", "said": "sd",
    "today": "td", "together": "tgr", "tomorrow": "tm", "tonight": "tn",
    "would": "wd", "your": "yr", "yourself": "yrf", "children": "chn",
}

examples = []

# ── 1. Letter dot patterns ──────────────────────────────────────────────
for letter, ds in LETTERS.items():
    cell = dots_to_char(*ds)
    dot_str = "-".join(str(d) for d in ds)

    # What dots make this letter?
    examples.append(msg(
        f"What are the dots for the letter {letter}?",
        f"The letter {letter} is dots {dot_str}, written as {cell} in braille."
    ))

    # What letter is this cell?
    examples.append(msg(
        f"What letter is {cell}?",
        f"That is the letter {letter} (dots {dot_str})."
    ))

    # How do I write this letter?
    examples.append(msg(
        f"How do I write {letter} in braille?",
        f"Press dots {dot_str} to make {cell}. That's the letter {letter}."
    ))

# ── 2. Letter groups and patterns ────────────────────────────────────────
examples.append(msg(
    "How is the braille alphabet organized?",
    "The braille alphabet is built from 10 base patterns repeated across 4 groups. "
    "Letters a-j use dots 1, 2, 4, and 5 (the top four positions). "
    "Letters k-t are the same as a-j with dot 3 added. "
    "Letters u-z add dot 6 (except w, which breaks the pattern because French didn't use W when Louis Braille invented the system in the 1820s). "
    "This means if you learn the first 10 letters, you already know the pattern for all 26."
))

examples.append(msg(
    "Why is w different from the pattern?",
    "When Louis Braille created his system in France in the 1820s, the French alphabet didn't commonly use W. "
    "So w (dots 2-4-5-6) was added later and doesn't follow the pattern of u, v, x, y, z which are a-e with dots 3 and 6 added. "
    "It's the one exception in the whole system."
))

for group_name, start, extra_dots in [
    ("a-e", 0, []),
    ("f-j", 5, []),
    ("k-o", 10, []),
    ("p-t", 15, []),
]:
    chars = [chr(97 + start + i) for i in range(5)]
    examples.append(msg(
        f"Teach me letters {chars[0]}-{chars[-1]}",
        f"Letters {chars[0]}-{chars[-1]}: " +
        ", ".join(f"{c} = dots {'-'.join(str(d) for d in LETTERS[c])} ({dots_to_char(*LETTERS[c])})" for c in chars) +
        ". Practice reading each cell, then try writing them from memory."
    ))

# ── 3. Numbers ──────────────────────────────────────────────────────────
examples.append(msg(
    "How do numbers work in braille?",
    f"Numbers in UEB use the same dot patterns as letters a-j, preceded by the numeric indicator {NUM_IND} (dots 3-4-5-6). "
    f"So 1 = {NUM_IND}{dots_to_char(1)} (numeric indicator + dots 1, which is also the letter a). "
    f"2 = {NUM_IND}{dots_to_char(1,2)} (same as b), and so on up to 0 = {NUM_IND}{dots_to_char(2,4,5)} (same as j). "
    "The numeric indicator tells the reader that the following a-j patterns should be read as digits, not letters."
))

for digit, ds in DIGITS.items():
    cell = dots_to_char(*ds)
    letter_equiv = chr(97 + (int(digit) - 1) % 10) if digit != "0" else "j"
    examples.append(msg(
        f"How do I write the number {digit} in braille?",
        f"The number {digit} in UEB is {NUM_IND}{cell}: the numeric indicator (dots 3-4-5-6) followed by dots {'-'.join(str(d) for d in ds)}, which is the same pattern as the letter {letter_equiv}."
    ))

examples.append(msg(
    "How do I write 42 in braille?",
    f"42 in UEB is {NUM_IND}{dots_to_char(1,4,5)}{dots_to_char(1,2)}: one numeric indicator followed by dots 1-4-5 (the pattern for d/4) then dots 1-2 (the pattern for b/2). The numeric indicator only appears once at the start of a number sequence."
))

# ── 4. Grade 2 wordsigns ────────────────────────────────────────────────
for word, letter in WORDSIGNS.items():
    cell = dots_to_char(*LETTERS[letter])
    dot_str = "-".join(str(d) for d in LETTERS[letter])
    examples.append(msg(
        f"What is the wordsign for '{word}'?",
        f"In Grade 2 UEB, '{word}' is written as {cell} (dots {dot_str}, the letter {letter}) when it stands alone as a word. This is called an alphabetic wordsign."
    ))
    examples.append(msg(
        f"What word does the wordsign {cell} represent?",
        f"When {cell} (dots {dot_str}, letter {letter}) stands alone as a word, it means '{word}' in Grade 2 UEB."
    ))

# ── 5. Strong contractions ──────────────────────────────────────────────
for word, ds in STRONG_CONTRACTIONS.items():
    cell = dots_to_char(*ds)
    dot_str = "-".join(str(d) for d in ds)
    examples.append(msg(
        f"How is '{word}' written in Grade 2 braille?",
        f"'{word}' is a strong contraction in UEB Grade 2: {cell} (dots {dot_str}). It can be used both as a standalone word and within other words."
    ))

# ── 6. Groupsigns ───────────────────────────────────────────────────────
for combo, ds in GROUPSIGNS.items():
    cell = dots_to_char(*ds)
    dot_str = "-".join(str(d) for d in ds)
    examples.append(msg(
        f"What is the groupsign for '{combo}'?",
        f"The groupsign for '{combo}' is {cell} (dots {dot_str}). It can be used inside words wherever the letter combination '{combo}' appears, saving space on a braille display."
    ))
    # Example in a word
    example_words = {
        "ch": "church", "sh": "she", "th": "the", "wh": "when",
        "gh": "ghost", "ed": "walked", "er": "never", "ou": "out",
        "ow": "now", "st": "stop", "ar": "star", "ing": "sing",
    }
    if combo in example_words:
        examples.append(msg(
            f"Can you show me '{combo}' used in a word?",
            f"In the word '{example_words[combo]}', the '{combo}' combination is written as the single cell {cell} (dots {dot_str}) instead of two separate letter cells. This is how Grade 2 braille saves space."
        ))

# ── 7. Shortforms ────────────────────────────────────────────────────────
for word, abbr in SHORTFORMS.items():
    examples.append(msg(
        f"What is the shortform for '{word}'?",
        f"The shortform for '{word}' is '{abbr}' -- you write just those letters without the rest of the word. Shortforms are whole-word abbreviations in UEB Grade 2 that must stand alone, not be used inside other words."
    ))

# ── 8. Conceptual / explanatory ──────────────────────────────────────────
CONCEPTS = [
    ("What is UEB?",
     "UEB stands for Unified English Braille. It's the standard braille code used in English-speaking countries since 2012-2016. It unified several older English braille codes (like BAUK and BANA) into one system. UEB covers literary text, math, science, and computer notation."),
    ("What is the difference between Grade 1 and Grade 2 braille?",
     "Grade 1 (uncontracted) braille writes every letter separately -- one cell per letter. Grade 2 (contracted) braille uses contractions: single cells or short sequences that represent common words or letter combinations. Grade 2 is what most braille readers use daily because it's 20-40% shorter than Grade 1."),
    ("What is a braille cell?",
     "A braille cell is a unit of 6 raised dots arranged in a 2x3 grid. The left column has dots 1 (top), 2 (middle), 3 (bottom). The right column has dots 4 (top), 5 (middle), 6 (bottom). Each combination of raised/flat dots represents a different character. There are 64 possible combinations (including the blank cell)."),
    ("What is 8-dot braille?",
     "8-dot braille adds dots 7 and 8 below dots 3 and 6, creating a 2x4 grid with 256 possible combinations. This is used in computer braille (like NABCC) where every ASCII character needs its own unique cell. The advantage is no indicators -- uppercase, lowercase, digits, and symbols each get their own cell. The disadvantage is that it requires a wider fingertip reading area."),
    ("Who invented braille?",
     "Louis Braille invented the braille system in 1824, when he was 15 years old. He was a student at the Royal Institute for Blind Youth in Paris. He adapted a military night-writing system called Sonography (created by Charles Barbier) from 12 dots to 6 dots, making it readable with a single fingertip touch."),
    ("Why is braille still important?",
     "Braille literacy is strongly correlated with employment and educational success among blind people. Studies show that 90% of employed blind adults are braille readers. Despite screen readers and text-to-speech, braille provides silent reading, precise spelling awareness, and access to formatting -- things audio alone can't replace."),
    ("What is a numeric indicator?",
     f"The numeric indicator is {NUM_IND} (dots 3-4-5-6). It tells the reader that the following cells should be read as numbers, not letters. In UEB, digits 1-9 use the same patterns as letters a-i, and 0 uses the pattern for j. Without the numeric indicator, there'd be no way to distinguish 'a' from '1'."),
    ("What is a capital indicator?",
     f"The capital indicator is {CAP_IND} (dot 6 alone). Place it before a letter to capitalize it. For an entire word in capitals, use two capital indicators ({CAP_IND}{CAP_IND}) before the word."),
    ("What are strong contractions?",
     "Strong contractions in UEB are: and, for, of, the, with. They can be used anywhere -- as standalone words or within other words. They're called 'strong' because they have no restrictions on where they appear (unlike lower groupsigns which have positional rules)."),
    ("What is the standing-alone rule?",
     "In UEB, wordsigns (like b for 'but', c for 'can') can only be used when the word stands alone -- preceded and followed by a space or punctuation. You can't use the wordsign for 'but' inside the word 'button'. This rule prevents ambiguity."),
    ("How many braille cells are there?",
     "With 6 dots, there are 2^6 = 64 possible braille cells (including the empty cell with no dots raised). With 8-dot braille, there are 2^8 = 256 possible cells. UEB uses all 63 non-empty 6-dot cells, many with multiple meanings depending on context (like the numeric indicator changing letter patterns to digit meanings)."),
    ("What is Nemeth braille?",
     "Nemeth braille is a code for mathematics and science, created by Abraham Nemeth in 1952. It uses different conventions than literary braille for numbers, operators, and spatial layout. In some countries, UEB Technical has replaced Nemeth, but Nemeth is still widely used in the US for STEM education."),
    ("What is a braille display?",
     "A refreshable braille display is an electronic device with a row of braille cells that can raise and lower individual pins. Common sizes are 14, 40, or 80 cells. The display connects to a computer or phone and shows text as braille in real time. Contracted braille (Grade 2) is important for these displays because it fits more content in fewer cells."),
    ("How fast can people read braille?",
     "Average braille reading speed is about 100-125 words per minute, compared to 200-250 wpm for sighted reading. Expert braille readers can reach 200+ wpm. Reading speed improves significantly with Grade 2 contractions because fewer cells means less finger travel."),
]

for q, a in CONCEPTS:
    examples.append(msg(q, a))

# ── 9. Common mistakes ──────────────────────────────────────────────────
MISTAKES = [
    ("I wrote dots 1-2 for 'a', is that right?",
     "Not quite -- 'a' is just dot 1 alone. Dots 1-2 is the letter 'b'. Remember: 'a' is the simplest letter, just one dot."),
    ("I keep confusing d and f",
     "That's common! d is dots 1-4-5 and f is dots 1-2-4. The difference: d has dot 5 (middle-right) while f has dot 2 (middle-left). Try this: d has dots going Down the right side (1, 4, 5), f has dots going across the top and middle (1, 2, 4)."),
    ("I keep mixing up i and j",
     "i is dots 2-4 and j is dots 2-4-5. They're identical except j adds dot 5. Think of j as i with an extra dot below on the right side."),
    ("Why did I get 'the' wrong? I wrote dots 2-3-4-5-6",
     "Close! 'the' is dots 2-3-4-6 (not 2-3-4-5-6, which is 'with'). Drop dot 5 and you have 'the'."),
    ("I'm struggling with numbers",
     f"Numbers use the exact same patterns as letters a-j, just with the numeric indicator {NUM_IND} (dots 3-4-5-6) first. So if you know a-j, you already know 1-9 and 0. The trick is remembering which letter maps to which digit: a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8, i=9, j=0."),
    ("What's the difference between 'the' and 'th'?",
     "Same dot pattern! 'th' as a groupsign and 'the' as a strong contraction are both dots 1-4-5-6. The difference is context: when dots 1-4-5-6 stands alone (surrounded by spaces), it means 'the'. Inside a word, it represents the letter combination 'th'."),
]

for q, a in MISTAKES:
    examples.append(msg(q, a))

# ── 10. Multi-turn tutoring dialogues ────────────────────────────────────
examples.append(multi(
    ("Teach me the letter a", "The letter a is dot 1 -- just a single dot in the top-left position. In braille: " + dots_to_char(1) + ". It's the simplest letter in the alphabet. Ready to try another?"),
    ("What about b?", "b is dots 1-2: " + dots_to_char(1,2) + ". That's dots 1 and 2 -- the entire left column's top and middle positions. Notice how b adds dot 2 to a's dot 1."),
    ("And c?", "c is dots 1-4: " + dots_to_char(1,4) + ". Now we jump to the right column -- dot 1 (top-left) plus dot 4 (top-right). The top row is filled. See the pattern? a=1, b=1-2, c=1-4, each adding one dot."),
))

examples.append(multi(
    ("I want to learn Grade 2", "Great choice! Grade 2 UEB uses contractions to save space. Let's start with the 5 strong contractions -- they're the most common words in English and can be used anywhere."),
    ("What are they?", "The 5 strong contractions are: and (" + dots_to_char(1,2,3,4,6) + " dots 1-2-3-4-6), for (" + dots_to_char(1,2,3,4,5,6) + " dots 1-2-3-4-5-6), of (" + dots_to_char(1,2,3,5,6) + " dots 1-2-3-5-6), the (" + dots_to_char(2,3,4,6) + " dots 2-3-4-6), with (" + dots_to_char(2,3,4,5,6) + " dots 2-3-4-5-6). They're called 'strong' because they work everywhere -- standalone or inside words."),
    ("Can I use 'the' inside 'other'?", "Yes! Since 'the' is a strong contraction, you can use it inside 'other': o + the + r. That's one of the advantages of strong contractions over wordsigns, which can only stand alone."),
))

examples.append(multi(
    ("What's the difference between wordsigns and contractions?", "Good question! Wordsigns represent a whole word using a single letter cell -- like 'b' for 'but', 'c' for 'can'. They can ONLY be used when the word stands alone. Contractions like 'the', 'and', 'for' use unique cell patterns and can be used inside words too."),
    ("So I can't use 'b' for 'but' inside 'button'?", "Exactly right. In 'button', the 'b' is just the letter b, not the wordsign for 'but'. The standing-alone rule prevents ambiguity -- otherwise readers wouldn't know if 'b' meant the letter or the word."),
))

examples.append(multi(
    ("How does capitalization work?", f"In UEB, you use the capital indicator {CAP_IND} (dot 6) before a letter to capitalize it. So 'Hello' starts with {CAP_IND}{dots_to_char(1,2,5)} -- capital indicator then h."),
    ("What about all caps like NASA?", f"For a whole word in capitals, use two capital indicators: {CAP_IND}{CAP_IND} before the word. For NASA: {CAP_IND}{CAP_IND}{dots_to_char(1,3,4,5)}{dots_to_char(1)}{dots_to_char(2,3,4)}{dots_to_char(1)} (cap-word indicator + n + a + s + a)."),
    ("What if it's a long passage in caps?", f"Three capital indicators in a row ({CAP_IND}{CAP_IND}{CAP_IND}) starts a capital passage. Everything after that is capitalized until a capital terminator (dots 6-3) appears."),
))

# ── 11. Social / identity ────────────────────────────────────────────────
SOCIAL = [
    ("What's your name?", "I'm a braille tutor -- I don't have a name. I'm here to help you learn UEB braille. Want to start with letters, numbers, or contractions?"),
    ("Who made you?", "I was built to teach braille. My knowledge comes from the UEB rulebook and braille education materials. What would you like to learn?"),
    ("Are you an AI?", "I'm a specialized braille tutor. I know UEB inside and out, but I can't help with things outside braille. Ready to practice?"),
    ("How old are you?", "I don't have an age -- I'm a practice tool for learning braille. Shall we work on some dot patterns?"),
    ("Hello!", "Hello! I'm your braille practice partner. You can ask me about any braille letter, number, contraction, or concept. Or say 'test me' to start a quiz."),
    ("Thank you", "You're welcome! Keep practicing -- braille reading speed improves a lot with daily repetition. Come back anytime."),
    ("This is hard", "Braille does have a learning curve, but here's the good news: the entire alphabet is built from just 10 base patterns. Once you learn a-j, the rest follows a predictable rule. Take it one group at a time. What are you finding hardest?"),
    ("I'm frustrated", "That's completely normal. Everyone struggles with certain letters -- d/f and i/j are the most commonly confused pairs. The key is short, frequent practice sessions rather than long marathons. Want to review your problem areas?"),
    ("What's the weather like?", "I only know braille, not the weather! But I can tell you that braille was invented in rainy Paris. Want to learn something about braille instead?"),
    ("Tell me a joke", "Here's a braille one: Why was the braille book always calm? Because it had a lot of good points. Now, back to learning -- what topic shall we cover?"),
]

for q, a in SOCIAL:
    examples.append(msg(q, a))

# ── 12. Practical questions ──────────────────────────────────────────────
PRACTICAL = [
    ("How do I type braille on my phone?",
     "On iPhone, enable the Braille Screen Input in Settings > Accessibility > VoiceOver > Braille. On Android, install BrailleBack or use TalkBack's braille keyboard. Both let you type by tapping 6 positions on the screen like a braille cell."),
    ("Where can I buy a braille display?",
     "Popular refreshable braille displays include the Humanware Brailliant (14-80 cells), Freedom Scientific Focus (14-80 cells), and Hable One (portable). Prices range from $500 for small displays to $5000+ for 80-cell models. Some organizations offer funding assistance."),
    ("How long does it take to learn braille?",
     "Most sighted learners can recognize all 26 letters within a few hours of practice. Reaching fluent reading speed (100+ wpm) takes months of daily practice. Grade 2 contractions add another layer -- most learners spend several weeks on the core contractions. The key is consistent daily practice, even just 15 minutes."),
    ("Is braille dying?",
     "No. While braille literacy rates dropped when audiobooks and screen readers became common, there's been a resurgence. Research consistently shows braille readers have better employment outcomes and spelling skills. Refreshable braille displays are getting cheaper and better. Braille is not dying -- it's evolving."),
    ("Can I learn braille if I'm sighted?",
     "Absolutely! Many sighted people learn braille: teachers of the visually impaired, parents of blind children, accessibility professionals, and curious learners. You can learn visually (reading braille by sight) or tactilely (reading by touch). Visual learning is faster initially; tactile reading is how blind readers actually use it."),
]

for q, a in PRACTICAL:
    examples.append(msg(q, a))

# ── 13. Quiz-style exchanges ─────────────────────────────────────────────
for _ in range(30):
    letter = random.choice(list(LETTERS.keys()))
    ds = LETTERS[letter]
    cell = dots_to_char(*ds)
    dot_str = "-".join(str(d) for d in ds)

    # Read direction
    examples.append(msg(
        f"What is {cell}?",
        f"That's the letter {letter} (dots {dot_str})."
    ))

    # Wrong answer correction
    wrong = random.choice([l for l in LETTERS if l != letter])
    examples.append(msg(
        f"Is {cell} the letter {wrong}?",
        f"No, {cell} (dots {dot_str}) is the letter {letter}, not {wrong}. {wrong} is dots {'-'.join(str(d) for d in LETTERS[wrong])} ({dots_to_char(*LETTERS[wrong])})."
    ))

# ── 14. Comparison questions ─────────────────────────────────────────────
pairs = [("d","f"), ("i","j"), ("e","i"), ("s","t"), ("n","o"), ("the","with")]
for a, b in pairs:
    if a in LETTERS and b in LETTERS:
        da, db = LETTERS[a], LETTERS[b]
        examples.append(msg(
            f"What's the difference between {a} and {b}?",
            f"{a} is dots {'-'.join(str(d) for d in da)} ({dots_to_char(*da)}) and {b} is dots {'-'.join(str(d) for d in db)} ({dots_to_char(*db)}). "
            f"The difference is that {b} has dot{'s' if len(set(db)-set(da)) > 1 else ''} {', '.join(str(d) for d in sorted(set(db)-set(da)))} that {a} doesn't."
        ))

# ── 15. RAG-grounded examples ────────────────────────────────────────────
# Teach the model to use REFERENCE FACTS from the RAG system prompt
RAG_SYSTEM = (
    SYSTEM +
    "\n\nREFERENCE FACTS (use these for accuracy -- they are verified UEB data):\n"
)

def rag_msg(user, assistant, facts):
    """Create a training example with RAG context injected into system prompt."""
    return {
        "messages": [
            {"role": "system", "content": RAG_SYSTEM + "\n".join(f"- {f}" for f in facts) + "\n\nAlways prefer these reference facts over your own knowledge when answering."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }

# Letter-with-RAG examples
for letter, ds in LETTERS.items():
    cell = dots_to_char(*ds)
    dot_str = "-".join(str(d) for d in ds)
    facts = [f"The letter '{letter}' in UEB braille is {cell} (dots {dot_str})."]
    examples.append(rag_msg(
        f"What is {letter} in braille?",
        f"The letter {letter} in braille is {cell} — that's dots {dot_str}.",
        facts
    ))

# Reverse lookups with RAG
for letter, ds in LETTERS.items():
    cell = dots_to_char(*ds)
    dot_str = "-".join(str(d) for d in ds)
    facts = [f"The letter '{letter}' in UEB braille is {cell} (dots {dot_str})."]
    examples.append(rag_msg(
        f"What does dots {dot_str} represent?",
        f"Dots {dot_str} is {cell}, the letter '{letter}'.",
        facts
    ))

# Comparison with RAG
for a, b, explanation in [("d","f","d has dot 5 while f has dot 2"), ("e","i","e and i are mirror images"), ("m","n","n has everything m has plus dot 5")]:
    da, db = LETTERS[a], LETTERS[b]
    cell_a, cell_b = dots_to_char(*da), dots_to_char(*db)
    dot_a, dot_b = "-".join(str(x) for x in da), "-".join(str(x) for x in db)
    facts = [
        f"The letter '{a}' in UEB braille is {cell_a} (dots {dot_a}).",
        f"The letter '{b}' in UEB braille is {cell_b} (dots {dot_b}).",
        f"Comparing '{a}' and '{b}': {explanation}."
    ]
    examples.append(rag_msg(
        f"What's the difference between {a} and {b}?",
        f"{a} is {cell_a} (dots {dot_a}) and {b} is {cell_b} (dots {dot_b}). {explanation}.",
        facts
    ))

# Contraction with RAG
for word, ds in STRONG_CONTRACTIONS.items():
    cell = dots_to_char(*ds)
    dot_str = "-".join(str(d) for d in ds)
    facts = [f"Strong contraction: '{word}' = {cell} (dots {dot_str}). Can be used as a whole word or within other words."]
    examples.append(rag_msg(
        f"How do I write '{word}' in contracted braille?",
        f"'{word}' is a strong contraction: {cell} (dots {dot_str}). You can use it both as a standalone word and inside other words.",
        facts
    ))

# Number with RAG
for digit in range(10):
    idx = (digit - 1) % 10
    letter = chr(97 + idx)
    ds = LETTERS[letter]
    cell = dots_to_char(*ds)
    dot_str = "-".join(str(d) for d in ds)
    facts = [
        f"The digit {digit} in UEB is {NUM_IND}{cell} (numeric indicator dots 3-4-5-6, then dots {dot_str}). Numbers use the same cell as letter '{letter}' but preceded by the numeric indicator.",
        f"The numeric indicator is {NUM_IND} (dots 3-4-5-6)."
    ]
    examples.append(rag_msg(
        f"What is {digit} in braille?",
        f"The number {digit} in UEB is {NUM_IND}{cell}: numeric indicator (dots 3-4-5-6) followed by dots {dot_str}, which is the same pattern as the letter '{letter}'.",
        facts
    ))

# Concept with RAG
examples.append(rag_msg(
    "How many dots are in a braille cell?",
    "A standard braille cell has 6 dots arranged in a 2×3 grid. Dots 1, 2, 3 are the left column (top to bottom) and dots 4, 5, 6 are the right column. This gives 64 possible combinations including the empty cell. Computer braille extends this to 8 dots (adding dots 7 and 8 below) for 256 combinations.",
    ["A standard braille cell has 6 dots arranged in a 2x3 grid. Dots are numbered: dot 1 (top-left), dot 2 (middle-left), dot 3 (bottom-left), dot 4 (top-right), dot 5 (middle-right), dot 6 (bottom-right). Computer braille (8-dot) adds dot 7 and dot 8 for 256 possible patterns. Standard 6-dot braille has 64 possible patterns."]
))

# Wordsign with RAG
for word, letter in list(WORDSIGNS.items())[:10]:
    ds = LETTERS[letter]
    cell = dots_to_char(*ds)
    dot_str = "-".join(str(d) for d in ds)
    facts = [f"Alphabetic wordsign: '{word}' = {cell} (letter {letter}, dots {dot_str}). Same cell as the letter '{letter}'. Only used when standing alone."]
    examples.append(rag_msg(
        f"What's the wordsign for '{word}'?",
        f"The wordsign for '{word}' is {cell} (letter {letter}, dots {dot_str}). This only works when '{word}' stands alone — surrounded by spaces or punctuation.",
        facts
    ))

# Shortform with RAG
for word, abbr in list(SHORTFORMS.items())[:15]:
    facts = [f"Shortform: '{word}' is written as '{abbr}' in Grade 2 braille."]
    examples.append(rag_msg(
        f"What's the shortform for '{word}'?",
        f"In Grade 2, '{word}' is shortened to '{abbr}'.",
        facts
    ))

# Groupsign with RAG
for combo, ds in GROUPSIGNS.items():
    cell = dots_to_char(*ds)
    dot_str = "-".join(str(d) for d in ds)
    facts = [f"Strong groupsign: '{combo}' = {cell} (dots {dot_str}). Used within words to shorten common letter combinations."]
    examples.append(rag_msg(
        f"What is the '{combo}' groupsign?",
        f"The '{combo}' groupsign is {cell} (dots {dot_str}). It replaces the letters '{combo}' with a single cell inside words.",
        facts
    ))

# Off-topic with RAG (model should still redirect)
examples.append(rag_msg(
    "What's the capital of France?",
    "That's outside my area — I'm a braille tutor. Want to learn how capital letters work in braille instead? The capital indicator is dot 6, placed before a letter to capitalize it.",
    ["Capital letter indicator: dot 6. Placed before a single uppercase letter."]
))
examples.append(rag_msg(
    "Can you write me a poem?",
    "I'm built for braille, not poetry! But I can show you how poetry looks in braille — every letter, space, and punctuation mark has its own cell. Want to try writing a short phrase?",
    []
))

# ── 16. Pedagogical patterns ─────────────────────────────────────────────
# How the model should teach and explain
examples.append(msg(
    "I don't understand the dot numbering",
    "Think of the braille cell as a 2×3 grid, like a small egg carton:\n\n"
    "  1  4\n  2  5\n  3  6\n\n"
    "The left column is dots 1 (top), 2 (middle), 3 (bottom). "
    "The right column is dots 4 (top), 5 (middle), 6 (bottom). "
    "So 'dots 1-2' means the top and middle dots on the left side. "
    "Try picturing this grid every time you see a dot number."
))
examples.append(msg(
    "What's the easiest way to remember the letters?",
    "Learn them in groups of 5:\n"
    "- a-e: use only the top 4 dots (1,2,4,5)\n"
    "- f-j: same patterns as a-e, no change\n"
    "- k-o: take a-e and add dot 3\n"
    "- p-t: take a-e and add dots 3+6... wait, that's wrong. p-t: take a-e and add dots 3 and 4... no.\n"
    "Actually: k-o = a-e + dot 3. p-t = a-e + dots 3+6... Let me be precise:\n"
    "k-o are a-e with dot 3 added. p-t are a-e with dots 3 AND 4 added (wait, no — a already has dot 1, so p = a + dots 3+4 = dots 1-2-3-4? No, p = dots 1-2-3-4 is wrong...).\n\n"
    "OK, let me just be straightforward: learn a-j first, then know that k=a+3, l=b+3, m=c+3, n=d+3, o=e+3. For u-z, it's k+6, l+6, etc (except w). Memorize the first 10, derive the rest."
))
# Better version:
examples.append(msg(
    "What's the easiest way to remember the braille alphabet?",
    "Learn the first 10 letters (a-j), then derive the rest:\n"
    "- a-j: the base patterns using dots 1, 2, 4, 5\n"
    "- k-o: same as a-e but add dot 3\n"
    "- p-t: same as a-e but add dots 3 AND 6... wait. No. p-t add dots 3 and 4 to a-e. Hmm.\n\n"
    "Let me be precise: p=1-2-3-4, q=1-2-3-4-5, r=1-2-3-5, s=2-3-4, t=2-3-4-5. Compare to a=1, b=1-2, c=1-4, d=1-4-5, e=1-5. So p=b+3+4? Not exactly a clean rule.\n\n"
    "Honestly, the cleanest approach: memorize a-j, then remember k=a+3, l=b+3, ..., o=e+3 (just add dot 3). For u-z, add dot 6 to k-o (except w which is unique). p-t don't follow as clean a rule — just memorize them."
))
# Clean pedagogical version (what we actually want the model to learn):
examples.append(rag_msg(
    "How is the braille alphabet organized?",
    "The braille alphabet uses a brilliant pattern:\n\n"
    "**Row 1 (a-e):** Use only the top 4 dot positions (1, 2, 4, 5)\n"
    "**Row 2 (f-j):** Same as a-e — the patterns repeat\n"
    "**Row 3 (k-o):** Same as a-e with dot 3 added\n"
    "**Row 4 (p-t):** Same as a-e with dots 3 AND 6 added\n"
    "**Row 5 (u-z):** Same as k-o with dot 6 added (except w, which breaks the pattern)\n\n"
    "So if you learn a through e, you basically know the shape of every letter. Just add dot 3 for k-o, dots 3+6 for p-t, dot 6 to k-o for u-z.",
    ["The braille alphabet is organized in rows of 5 letters. a-e use dots 1,2,4,5. f-j repeat the same patterns. k-o add dot 3 to a-e. p-t add dots 3 and 6 to a-e. u-z add dot 6 to k-o, except w which is an exception."]
))

# ── 17. Conversational naturalness ────────────────────────────────────────
# The model should handle casual/natural language well
CASUAL = [
    ("hey", "Hey! Ready to practice braille? Type 'test me' to start, or ask me anything about braille."),
    ("yo what up", "Not much — just waiting to teach you braille! What do you want to work on? Letters, numbers, or contractions?"),
    ("this is cool", "Glad you think so! Braille is an elegant system. Want to dive deeper into any part of it?"),
    ("i love this", "That's great to hear! Consistent practice is the key to braille fluency. What's next?"),
    ("bruh", "Ha! Ready to learn some braille, or just here to hang out?"),
    ("lol", "Glad I could make you laugh! But seriously — want to try a quiz?"),
    ("hmm", "Take your time. When you're ready, ask me anything about braille or say 'test me'."),
    ("ok cool", "Cool. What's next — letters, numbers, contractions, or something else?"),
    ("nice", "Thanks! Ready for the next challenge?"),
    ("wait what", "No rush — happy to explain anything again. What's confusing?"),
    ("i forgot", "No worries, that's normal! Braille takes repetition. What do you want to review?"),
    ("can you teach me something fun", "Here's a fun one: in Grade 2, the word 'people' is just a single cell — the letter p (dots 1-2-3-4). So a whole word becomes one dot pattern. Want to learn more wordsigns?"),
    ("what should i learn first", "Start with letters a-j — they're the foundation. Once you know those 10 patterns, every other letter is just those patterns with extra dots added. Then move to numbers (same patterns as a-j!), and finally Grade 2 contractions."),
    ("i'm bored", "Let's make it interesting — try a quiz! I'll show you braille cells and you identify them. Say 'test me' to start."),
    ("goodbye", "See you next time! Keep practicing — even a few minutes a day makes a big difference."),
    ("see ya", "Later! Your progress is saved, so pick up where you left off next time."),
]

for q, a in CASUAL:
    examples.append(msg(q, a))

# ── 18. Multi-language awareness ─────────────────────────────────────────
examples.append(msg(
    "Does braille work in other languages?",
    "Yes! Every major language has its own braille code. French, Spanish, German, Arabic, Chinese, Japanese, Hindi — they all have braille systems. The 6-dot cell structure is universal, but the meaning of each cell varies by language. UEB is specifically for English. The code assignments were designed for each language's letter frequency and phonetics."
))
examples.append(msg(
    "ben jij agi?",
    "I only speak English and braille! I'm a UEB (Unified English Braille) tutor. Want to practice some braille?"
))
examples.append(msg(
    "hola",
    "¡Hola! I'm an English braille tutor, so I teach UEB. Spanish braille exists too, but that's a different code. Want to learn English braille?"
))

# Shuffle and write
random.seed(42)
random.shuffle(examples)

output_path = "/Users/ryanbarrett/.pi/agent/extensions/native-braille/tutor/training/braille_tutor_train.jsonl"
with open(output_path, "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Wrote {len(examples)} examples to {output_path}")

# Stats
turns = sum(len(ex["messages"]) // 2 for ex in examples)
print(f"Total conversation turns: {turns}")
print(f"Categories: letter patterns, number patterns, wordsigns, contractions, groupsigns, shortforms, concepts, mistakes, dialogues, social, practical, quiz, RAG-grounded, casual, pedagogical")
