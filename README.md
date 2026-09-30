# braille tutor

Interactive braille learning app with Duolingo-style spaced repetition and a fine-tuned AI tutor. Single HTML file, no build step, no account required.

**[Try it live](https://elevate-foundry.github.io/braille-tutor/)**

## What it teaches

- **Letters a-z** -- dot patterns, reading and writing
- **Numbers 0-9** -- numeric indicator + digit cells
- **Uncontracted words** -- spell out words cell by cell
- **Grade 2 wordsigns** -- whole words as single cells (but=b, the, and, for, of, with...)
- **Grade 2 groupsigns** -- contractions inside words (ch, sh, th, ing, er, ed...)
- **Grade 2 shortforms** -- abbreviated words (about=ab, friend=fr, children=chn...)

## How it works

Type `test me` and press Enter. The tutor alternates between:

- **Reading**: shows braille cells, you type the print answer
- **Writing**: shows a print character/word, you enter dot numbers (e.g. `1-2`) or tap the dot grid
- **Retry**: wrong answers invite you to try again -- no lockouts

Natural language works: "teach me numbers", "practice shortforms", "I want to learn grade 2".

### Commands

| Command | What it does |
|---------|-------------|
| `test me` | Start a 5-question round |
| `hint` | Show the answer (no mastery XP) |
| `explain` | Explain the current pattern |
| `next` | Move to the next question |
| `skip` | Skip without answering |
| `review` | Practice missed and due items |
| `help` | List all commands |

You can also say `ok`, `got it`, or `thanks` to advance, and `i give up` to skip.

### AI tutor (RAG-enhanced)

Open-ended questions are answered by a fine-tuned LLM with retrieval-augmented generation (RAG). The model is Qwen2.5-3B, QLoRA-trained on 448 UEB tutoring examples, with a 223-entry knowledge base injected at query time for factual accuracy:

- "What is the letter f in braille?" -- **dots 1-2-4** (verified by RAG, not hallucinated)
- "What does dots 1-2-5 represent?" -- **the letter h** (reverse lookup via RAG)
- "What is the difference between d and f?" -- detailed dot-pattern comparison with correct cells
- "How do numbers work?" -- conceptual explanation with all 10 digits listed correctly
- "What are the strong groupsigns?" -- lists them with correct dot patterns and Unicode
- Off-topic questions get redirected back to braille

The AI shows a "Thinking..." indicator while loading. First request after idle may take ~30s (cold start); subsequent requests are 2-5s. If the AI is unavailable, the tutor falls back to its local rule engine -- quiz grading and spaced repetition always work offline.

### Spaced repetition

Each skill (reading a letter, writing a letter) is tracked independently. Correct recalls schedule reviews at 1, 3, then 7 days. Mistakes and hints reset the interval. Progress is stored in `localStorage`.

### Other features

- **Remembers your name** -- say "i'm Ryan" and it greets you next time
- **XP and streaks** -- earned per correct answer, tracked across sessions
- **Dot grid input** -- tap dots or use keyboard (Tab + Space) to build cells
- **Braille-first mode** -- hides print feedback for immersive practice
- **Sound feedback** -- tones for correct/wrong answers
- **Mobile-friendly** -- works at 390px viewport width
- **Learn tab** -- structured lesson road with 9 lessons and stars
- **Practice tab** -- free-play quiz with level selection
- **Translation sandbox** -- experimental G1/G2 translator

## Architecture

```
                    +------------------+
User input ------->| Intent detection |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
     Quiz / commands               Open-ended questions
     (local, instant)              (LLM + RAG on Modal)
              |                             |
     +--------+--------+      +-----------+-----------+
     | Rule engine      |      | 1. RAG retrieval      |
     | XP, spaced rep,  |      |    223-entry UEB KB   |
     | grading, topics  |      | 2. Inject top-8 facts |
     +-----------------+      |    into system prompt  |
                               | 3. Qwen2.5-3B answers |
                               |    with verified data  |
                               +-----------------------+
```

Quiz grading is always local and deterministic. The LLM only handles conversation. RAG (retrieval-augmented generation) ensures dot patterns, contraction rules, and shortforms are accurate by injecting verified facts from the knowledge base into the model's context at query time.

## Run locally

```
python3 -m http.server 8042
```

Open http://localhost:8042. That's it -- single HTML file, no build step.

## Training

The AI model is fine-tuned using the scripts in `training/`:

```bash
# Generate the training dataset (448 examples)
python3 training/build_dataset.py

# Build the RAG knowledge base (223 entries)
python3 training/build_rag_kb.py

# Train on Modal (requires Modal account + A10G GPU)
cd training && modal run modal_app.py::train

# Deploy the inference endpoint with RAG
cd training && modal deploy modal_app.py
```

### Files

| File | Purpose |
|------|---------|
| `build_dataset.py` | Generates 448 training examples (letters, numbers, contractions, RAG-grounded, casual, pedagogical) |
| `build_rag_kb.py` | Generates 223-entry knowledge base from UEB specification (letters, numbers, punctuation, contractions, shortforms, concepts) |
| `braille_tutor_train.jsonl` | The training dataset (ChatML format) |
| `ueb_knowledge.json` | The RAG knowledge base (JSON, keyword-indexed) |
| `modal_app.py` | Modal app: QLoRA training + RAG-enhanced inference endpoint |

### How RAG works

At inference time, the endpoint:
1. Extracts keywords from the user's question (including dot patterns like "1-2-5" and Unicode braille characters)
2. Scores all 223 knowledge base entries by keyword overlap
3. Injects the top 8 matching facts into the system prompt as "REFERENCE FACTS"
4. The model is trained to prefer these facts over its own knowledge

This eliminates dot-pattern hallucination -- the main failure mode of small language models on braille data.

## Data

All learning progress stays in your browser's `localStorage`. The only network request is to the Modal AI endpoint for open-ended questions. No personal data is sent -- just the conversation text.

## License

MIT
