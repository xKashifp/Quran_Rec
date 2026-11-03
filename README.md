# Ayah Finder

You recite some Quran into the mic, and the app tells you which ayah it was. It shows the surah, the ayah number, the Uthmani text, a match score and the top 3 guesses.

Recognition is done only by Quran-Lab's zipformer_p-arabic-v3.1 model, running locally with sherpa-onnx. No Whisper or any other speech-to-text.

## Running it

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
hf auth login
python scripts/download_model.py
python scripts/build_quran_text.py
cd backend
uvicorn app:app --port 8000
```

The model repo is gated, so `hf auth login` has to use an account that's been approved on the model page. After that, open http://localhost:8000, press Start, recite, and press Stop. You can also upload an audio file instead of using the mic.

If something isn't being recognised, start the server with `QURANLAB_SAVE_RECORDINGS=1`. It then saves each recording, plus what the model heard, into `recordings/`. That's how I found out my first tries failed because my mic was off.

## Finding v3.1

The repo has no tags, so I went through the commit history to find v3.1.

The file called `zipformer_p_arabic_v3.1.onnx` on `main` hasn't changed since August 17. That's when the madd fine-tune was uploaded and renamed to v3.1. On October 31, the maintainers said this export used a different SwooshL activation setting than the one the model was trained with. They uploaded a corrected "as trained" export, renamed it to v3.2 the same day, and then removed v3.2 and v3.3 on November 2. The card now just says v3.1 is the latest.

So there were two candidates. I ran both on the same 76 test clips:

| Model file | Full | Partial | Across two ayat | Mean score of correct matches |
|---|---|---|---|---|
| Official v3.1 | 40/40 | 18/20 | 12/16 | 96.5% |
| Corrected export | 40/40 | 17/20 | 11/16 | 94.4% |

The official file did slightly better. That fits the maintainers' own October numbers: the corrected export was better at catching mistakes, but slightly worse on correct recitation. So I use the official file, and the download script is pinned to commit `5a115aa9`.

## What I learned about the model's output

My full notes are in [docs/phonetic_alphabet.md](docs/phonetic_alphabet.md). The script `scripts/study_alphabet.py` generates them from the model's own files.

The model outputs one of 251 symbols every 40 ms, and one of those symbols is the CTC blank. Decoding takes the best symbol for each frame, merges repeats, and drops the blanks. The blank is also how the model says the same sound twice in a row. Use `tokens.txt` for the ids, not `phoneme_units.json`, which is off by one.

Those 250 symbols are built from just 42 characters. Most of them are chunks, like a letter plus its vowel (`بَ`) or a whole madd (`اااا`). So the same sound can be split into tokens more than one way, which is why I compare characters instead of tokens.

The script writes pronunciation, not spelling:

- **Letters and vowels.** Letters are plain Arabic letters, and fatha, damma and kasra follow them. Sukun has no symbol: a letter with no vowel after it is sakin. For example, بِسْمِ is written `بِسمِ`.
- **Shadda** is a doubled letter: رَبِّ is `رَببِ`. A doubled noon or meem also gets its ghunna, so it appears 4 times: إِنَّ is `ءِننننَ`.
- **Tanween** is written as a real noon, which then follows the noon rules. At a stop, tanween fath becomes `اا`.
- **Madd** is the number of repeated long vowels (`ا`, `ۥ`, `ۦ`), one per beat. The reference only uses 2, 4 and 6: natural madd is 2, the connected and separated madd are 4, and madd lazim is 6.
- **Hidden noon (ikhfaa)** is `ںںں`. Meem before ب, which covers both ikhfaa shafawi and iqlab, is `۾۾۾`.
- **Idgham with ghunna** into و or ي triples that letter: مَن يَقُولُ becomes `مَيييَقُۥۥلُ`.
- **Qalqalah** is a `ڇ` after a sakin ق ط ب ج د.
- **Rare cases.** A few rare Hafs features have their own marks: saktah `ۜ`, tasheel `ٲ` and imalah `۪`.

Words also merge across boundaries the way they're recited, so the phonetic words don't line up with the written words.

## How the search works, and why

I joined the phonemes of all 6,236 ayat into one long sequence. A recitation is then just a stretch of that sequence with a few mistakes in it. That's why partial ayat and recitations running across two ayat needed no special handling: they're just shorter or longer stretches. Matching word by word wouldn't work, because the phonetic words don't line up with the written ones.

The search has two steps:

1. **A fast index finds likely places.** Every 4-character window of the recitation votes for where it appears in the Quran, and the real location collects lots of votes in one spot. Repeated characters are collapsed first, so madd length doesn't matter at this stage.
2. **Edit distance checks the best places properly.** The whole recitation has to match, but it can start and end anywhere in the reference.

In the alignment, a missing or extra repeat of a character costs 0.3, while a wrong letter costs 1. That's because madd and ghunna length is the thing that varies most between reciters. A stop madd, for example, can be 2, 4 or 6 beats, while the reference always writes 4.

The score is how much of the recitation matched, with 100% being a perfect match. If the match covers more than one ayah, the page shows all of them.

Some phrases appear word-for-word in more than one place, like فَبِأَيِّ ءَالَآءِ رَبِّكُمَا تُكَذِّبَانِ in Ar-Rahman. Audio alone can't tell those apart, so the guesses tie and the page says the result is ambiguous.

## How well it works

I tested with EveryAyah recordings of Alafasy, Husary, Abdul Basit and Minshawy, using `tests/test_recitations.py`:

| Case | Correct first guess | In top 3 |
|---|---|---|
| Full ayah | 40/40 | 40/40 |
| Part of an ayah | 18/20 | 20/20 |
| Across two ayat | 12/16 | 14/16 |

Every miss was a duplicate phrase:

- **Part of an ayah.** The middle of 3:190 is word-for-word the same as part of 2:164.
- **Across two ayat.** The join of 36:2 and 36:3 mostly leaves ءِنَّكَ لَ, which shows up all over the Quran.

On my laptop, recognition runs about 17 times faster than real time, and the search takes under 0.15 seconds.

## Known limits

- **A'udhu billah.** It isn't in the reference, so saying it first lowers the score.
- **Qira'at.** The model and reference are Hafs 'an 'Asim only.
- **Repeated phrases.** If you repeat a phrase, the score drops, but it usually still finds the right ayah.
- **Short clips.** A few words can easily be ambiguous.
- **Not a teacher.** Automatic recognition can be wrong and doesn't replace a qualified teacher. The model's license (NPL-1.2) asks every app to say this.

## Credits

- **Model and reference phonemes:** Quran-Lab zipformer_p-arabic-v3.1, under the Quran-Lab No-Profit License 1.2.
- **Uthmani text and surah names:** Tanzil Project (https://tanzil.net), under CC BY 3.0. The text is unmodified.
