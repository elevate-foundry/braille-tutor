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

### AI tutor

Open-ended questions are answered by a fine-tuned LLM (Qwen2.5-3B, QLoRA-trained on 315 UEB tutoring examples, served on [Modal](https://modal.com)):

- "Who invented braille?" -- real historical answer
- "What is the difference between d and f?" -- detailed dot-pattern comparison
- "How do numbers work?" -- conceptual explanation
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
     (local, instant)              (LLM on Modal, 2-30s)
              |                             |
     +--------+--------+          +--------+--------+
     | Rule engine      |          | Qwen2.5-3B      |
     | XP, spaced rep,  |          | QLoRA fine-tuned |
     | grading, topics  |          | braille expert   |
     +-----------------+          +-----------------+
```

Quiz grading is always local and deterministic. The LLM only handles conversation.

## Run locally

```
python3 -m http.server 8042
```

Open http://localhost:8042. That's it -- single HTML file, no build step.

## Training

The AI model is fine-tuned using the scripts in `training/`:

```bash
# Generate the dataset
python3 training/build_dataset.py

# Train on Modal (requires Modal account)
cd training && modal run modal_app.py::train

# Deploy the inference endpoint
cd training && modal deploy modal_app.py
```

The training dataset covers UEB letter patterns, numbers, Grade 2 contractions, wordsigns, groupsigns, shortforms, conceptual explanations, common mistakes, multi-turn tutoring dialogues, and social responses.

## Data

All learning progress stays in your browser's `localStorage`. The only network request is to the Modal AI endpoint for open-ended questions. No personal data is sent -- just the conversation text.

## License

MIT
