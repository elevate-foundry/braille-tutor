# braille tutor

Interactive braille learning app with Duolingo-style spaced repetition. No accounts, no backend, no dependencies — runs entirely in your browser.

**[Try it live](https://elevate-foundry.github.io/braille-tutor/)**

## What it teaches

- **Letters a-z** — dot patterns, reading and writing
- **Numbers 0-9** — numeric indicator + digit cells
- **Uncontracted words** — spell out words cell by cell
- **Grade 2 wordsigns** — whole words as single cells (but=b, the, and, for, of, with...)
- **Grade 2 groupsigns** — contractions inside words (ch, sh, th, ing, er, ed...)
- **Grade 2 shortforms** — abbreviated words (about=ab, friend=fr, children=chn...)

## How it works

Type `test me` and press Enter. The tutor alternates between:

- **Reading**: shows braille cells, you type the print answer
- **Writing**: shows a print character/word, you enter dot numbers (e.g. `1-2`) or tap the dot grid

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

### Spaced repetition

Each skill (reading a letter, writing a letter) is tracked independently. Correct recalls schedule reviews at 1, 3, then 7 days. Mistakes and hints reset the interval. Progress is stored in `localStorage` — no server, no account.

### Other features

- **XP and streaks** — earned per correct answer, tracked across sessions
- **Dot grid input** — tap dots or use keyboard (Tab + Space) to build cells
- **Braille-first mode** — hides print feedback for immersive practice
- **Sound feedback** — tones for correct/wrong answers
- **Mobile-friendly** — works at 390px viewport width
- **Translation sandbox** — experimental G1/G2 translator (not the quiz answer key)
- **Learn tab** — structured lesson road with stars
- **Practice tab** — free-play quiz with level selection

## Run locally

```
python3 -m http.server 8042
```

Open http://localhost:8042. That's it — single HTML file, no build step.

## Data

All progress stays in your browser's `localStorage`. Nothing is sent to any server. Clearing browser data resets progress.

## License

MIT
