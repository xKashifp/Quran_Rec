# Ayah Finder

A small web app. You recite part of the Quran into the microphone, and it tells you which ayah it was. It shows the surah, the ayah number, the Uthmani text, a match score, and the top 3 candidates.

Recognition uses only the Quran-Lab **zipformer_p-arabic-v3.1** phoneme model, run locally with sherpa-onnx. No other speech-to-text is used.

## Setup

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

1. **Access.** `hf auth login` needs an account that has been approved on the gated model page.
2. **Model files.** `download_model.py` fetches the v3.1 model, `tokens.txt` and the reference phonemes into `models/`. It is pinned to commit `5a115aa9`.
3. **Quran text.** `build_quran_text.py` fetches the Tanzil Uthmani text.
4. **Run.** Open http://localhost:8000, press Start, recite, and press Stop. You can also upload an audio file.

To debug recognition, set `QURANLAB_SAVE_RECORDINGS=1` before starting the server. Each request's audio and recognised phonemes are then saved in `recordings/`.

## Which file is v3.1

The repo has no tags, so I went through the commit history and compared file hashes.

- **August 17.** The madd fine-tune of v3 was uploaded and renamed `zipformer_p_arabic_v3.1.*`. The `zipformer_p_arabic_v3.1.onnx` on `main` today is byte-identical to that upload.
- **October 31.** The card said the released v3.1 ONNX ran one activation (SwooshL) with a different setting from the one used in training. It added a corrected "as trained" export of the same weights. That export was renamed v3.2 the same day, and v3.3 followed.
- **November 2.** v3.2 and v3.3 were removed, and the card went back to "Latest: v3.1".

So I use `zipformer_p_arabic_v3.1.onnx` from `main`. I also tested the corrected export, which is still in commit `3858bdc2`, on the same 76 clips.

| Model file | Full | Partial | Across two ayat | Mean score of correct matches |
|---|---|---|---|---|
| Official v3.1 | 40/40 | 18/20 | 12/16 | 96.5% |
| Corrected export | 40/40 | 17/20 | 11/16 | 94.4% |

The official file is slightly better here. That matches the October card's own numbers: the corrected export caught more recitation mistakes, but was slightly worse on correct recitation.

## The model's phonetic alphabet

Full notes are in [docs/phonetic_alphabet.md](docs/phonetic_alphabet.md), generated from the data by `scripts/study_alphabet.py`.

- **Tokens and CTC.** `tokens.txt` has 250 symbols plus `<blank>` at id 250. The model outputs one prediction every 40 ms. Decoding keeps the best symbol per frame, merges repeats, and removes blanks. A blank between two identical symbols is how the model says a sound twice, for example a long madd. `phoneme_units.json` puts the blank at 0, so its ids are off by one and are not used.
- **Built from 42 characters.** The 250 symbols are made of 42 characters. Most tokens are merged chunks: a letter with its vowel (`بَ`), a doubled letter with its vowel (`ببَ`), or a whole madd (`اااا`). One sound can be split into tokens in different ways, so the app compares characters, not tokens.
- **Letters.** Each consonant is written as its Arabic letter. Hamza is `ء`, and a hamzat al-wasl is written only when it is actually pronounced.
- **Short vowels.** Fatha, damma and kasra follow their letter. Sukun has no symbol: a letter with no vowel after it is sakin. For example, بِسْمِ is written `بِسمِ`.
- **Shadda** is a doubled letter: رَبِّ is `رَببِ`. With noon and meem it also carries ghunna, so they appear 4 times: إِنَّ is `ءِننننَ`.
- **Tanween** is written as the noon it sounds like, then follows the noon rules. سَوَآءٌ عَلَيْهِمْ is `سَوَااااءُن عَلَيهِم`. At a stop, tanween fath becomes a 2-beat alif: رَقِيبًا is `رَقِۦۦبَاا`.
- **Madd.** `ا`, `ۥ` and `ۦ` are long a, u and i, one symbol per beat. The reference uses only runs of 2, 4 and 6. These are natural madd at 2, the connected and separated madd at 4 (Hafs via ash-Shatibiyyah), and madd lazim at 6. A stop on a long vowel, as in ٱلرَّحِيمِ, gives `ۦۦۦۦ`, which is 4.
- **Ghunna and ikhfaa.**
  - Hidden noon is `ںںں`, always 3 symbols.
  - Meem before ب, both ikhfaa shafawi and iqlab, is `۾۾۾`. For example, أَنۢبَآءِ is `ءَ۾۾۾بَااااءِ`.
  - Idgham with ghunna into و or ي triples that letter: مَن يَقُولُ is `مَيييَقُۥۥلُ`.
- **Qalqalah** is `ڇ` written after a sakin ق ط ب ج د: رَزَقْنَـٰهُمْ is `رَزَقڇنَااهُم`.
- **Rare Hafs features** have their own marks: saktah `ۜ` at 75:27 and 83:14, tasheel `ٲ` at 41:44, and imalah `۪` at 11:41.
- **Pronunciation, not spelling.** The reference writes each ayah as it is pronounced in connected recitation, with the stopping form at its end. Words therefore merge: غِشَـٰوَةٌ وَلَهُمْ is written as one chunk, `غِشَااوَتُوووَلَهُم`.

## How it works

```
mic ─► browser: 16 kHz mono WAV ─► FastAPI ─► sherpa-onnx, zipformer2 CTC, greedy
                                                   │ phoneme characters
                                                   ▼
                              search all 6,236 ayat (652k phoneme characters)
                                                   │
                                                   ▼
                     top 3 + scores ─► page: surah, ayah, Uthmani text, score
```

The browser records raw audio and converts it to 16 kHz WAV itself, so the server does not need ffmpeg.

### The search, and why I chose it

The model gives phonemes, so I compare phonemes directly. I never convert back to Arabic words.

**The whole Quran as one sequence.** All ayat are joined in order into one long phoneme sequence. A recitation is then a piece of that sequence with a few errors. That covers all three cases from the brief:

- a full ayah
- part of an ayah
- a recitation that runs across two or more ayat

None of them needs special handling. Matching word by word would not work well, because the phonetic words merge across word boundaries, as shown above.

**Step 1: find candidate places.** Every 4-character window of the recitation is looked up in an index of the whole Quran. Each hit votes for a position. The true position collects many votes in the same spot, while chance matches are spread out. Before indexing, repeated characters are collapsed (`اااا` becomes `ا`), so madd length does not affect this step. This takes milliseconds and survives recognition errors, because only some windows need to match.

**Step 2: score the best places precisely.** For each of the 12 to 24 strongest places, the recitation is aligned to the reference with edit distance. The whole recitation has to be matched, but it may start and end anywhere in the reference ("fitting" alignment). One extra or missing repeat of the same character costs 0.3 instead of 1. This keeps madd and ghunna length differences cheap, because those are what differ most between reciters. For example, a stop madd may be 2, 4 or 6 beats, while the reference always writes 4. A wrong letter or vowel costs the full 1.

**Score.** The score is `1 − alignment cost ÷ number of recognised phonemes`, where 100% is a perfect match. The matched span is mapped back to the ayat it covers. The page then shows whether the recitation covered one ayah, part of one, or several.

**Repeated wording.** Some phrases appear word-for-word in several places. فَبِأَيِّ ءَالَآءِ رَبِّكُمَا تُكَذِّبَانِ occurs 31 times in Surah Ar-Rahman. Audio alone cannot tell those apart. The candidates then tie, and the page marks the result as ambiguous.

## Results

`tests/test_recitations.py` runs EveryAyah recordings of four reciters through the full pipeline: Alafasy, Husary, Abdul Basit and Minshawy. Each reciter has 10 full ayat, 5 middle slices covering 30% to 65% of an ayah, and 4 joins of the second half of one ayah with the first half of the next.

| Query type | Top-1 | Top-3 |
|---|---|---|
| Full ayah | 40/40 | 40/40 |
| Part of an ayah | 18/20 | 20/20 |
| Across two ayat | 12/16 | 14/16 |

Recognition runs about 17 times faster than real time on a laptop CPU, and the search takes 25 to 140 ms.

- **Part-of-ayah misses.** Both are the middle of 3:190. That passage is word-for-word identical to part of 2:164, and 3:190 is still in the top 3.
- **Across-two-ayat misses.** All are the join of 36:2 and 36:3. Half of 36:2's audio is mostly its final madd and silence. What remains is ءِنَّكَ لَ, which occurs in many places.
- **Reciter repetition.** Alafasy repeats a phrase inside 18:10. The score drops to 75%, but the right ayah still wins.

`tests/simulate_search.py` tests only the search. It needs no model, and it injects random errors into stand-in queries.

## Limitations

- Isti'adha (أعوذ بالله) is not in the reference. Reciting it lowers the score.
- The model and reference follow Hafs 'an 'Asim. Other qira'at will score lower.
- Very short recitations are often ambiguous.
- Automatic recognition can be wrong and does not replace a qualified teacher. The model's license (NPL-1.2) requires this notice.

## Credits

- Model and reference phonemes: Quran-Lab zipformer_p-arabic-v3.1, under the Quran-Lab No-Profit License 1.2.
- Uthmani text and surah metadata: Tanzil Project (https://tanzil.net), CC BY 3.0. The text is unmodified.
