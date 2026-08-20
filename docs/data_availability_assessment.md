# Is there enough EthSL data to train an isolated-word model?

Assessment date: 2026-08-20. Every access claim below was tested, not assumed.

## Bottom line

**No.** There is no publicly downloadable Ethiopian Sign Language corpus adequate
for training an isolated-word recogniser. The only open download is alphabet
fingerspelling. Every word-level corpus is request-gated, and the one official
EthSL video dictionary is currently offline.

This does not kill the project. It means the data has to be recorded, and that
recording — not modelling — is the critical path.

## Step 1 — existing labelled datasets

| Source | Content | Public? | Verified how |
|---|---|---|---|
| [Mendeley: Amharic Sign Language Data Sets](https://data.mendeley.com/datasets/5d3nkyhsrf/1) — Isayas Feyera, AASTU, 2020, CC BY 4.0 | `Amharic_Sign_Language.rar`, 313 MB. Amharic **alphabet** fingerspelling. A downstream repo reports 1,172 RGB images over **7 alphabet characters** | Open | Pulled file id from the public API and started the download. API reports `file_downloads: 0` — nobody had ever downloaded it |
| [CESL](https://zenodo.org/records/10800699) (DOI 10.5281/zenodo.10800699) | 1,320 videos, 22 signers, 30 **sentences**, 65-word vocab, 1080p/25fps, 92.7 GB, CC BY 4.0 | **Restricted** — record public, files gated | Fetched record: "publicly accessible, but files are restricted". 7 downloads total |
| [CESLR code](https://github.com/ethio-artifical/CESLR) | ResNet18 + BiLSTM + CTC training code | Code only, **no data** | Read repo |
| [Sci Reports 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12533181/) | 20 isolated words, 5,600 clips, 7 signers, MediaPipe Holistic 543 landmarks | Raw withheld "owing to privacy and institutional restrictions"; skeletons on request | Read data-availability statement |
| [Heliyon 2024](https://www.cell.com/heliyon/fulltext/S2405-8440%2824%2914296-X) | 2,430 samples, 15 signers, 7 derived alphabets | "Available from the corresponding author on reasonable request" | Read statement |
| [ESL-Recognition-CNN](https://github.com/yohannis-abraham/Ethiopian-Sign-Language-Recognition-CNN) | CNN code, 7 alphabet classes | Code only; points back at the Mendeley set | Read repo |
| [AAU EthSL digital dictionary](https://ethsl.aau.edu.et/) | **1,322 signs with video**, 25 headings. Official, US State Dept funded | **Server down** — `ethsld.aau.edu.et/dictionary/` returns HTTP 502 | Requested 4x over ~10 min, 502 every time. No Wayback snapshots exist |
| Kaggle | — | Nothing EthSL-specific | Searched |
| Hugging Face | — | Nothing EthSL-specific (Amharic speech/text only) | Searched |

### The dictionary is not a training set, even when it works

1,322 signs sounds transformative. It is not, for training: a dictionary holds
roughly **one video per sign**. One example per class trains nothing. Its real
value is different and still high — it is the authority on which handshape is
correct, so your signers record the right sign and your gloss list is defensible.
Treat it as the reference, and budget zero training clips from it.

## Step 2 — YouTube as raw material

| Source | Format | Extraction difficulty |
|---|---|---|
| "Learn Ethiopian Sign Language" playlists (2 found: `PL-An2pU9EfLcJMNhz28GSmclBhw_YB87s`, `PLRChqvgJmnXBMhTXXVcRUxBUnbk996Bu_`) — topic lessons: Verbs, Countries, School, Important Phrases | Lesson compilations, many signs per video, spoken/captioned labels | **Medium-hard.** Needs manual temporal segmentation per sign |
| [Visions Global Empowerment](https://www.visionsethiopia.org/video) — EthSL stories: *Walking Together*, *Teju's Shadow*, *Friends*, *Shongololo's Shoes*, *7 Colors of a Rainbow* | Continuous narrative signing | **Hardest.** Continuous, co-articulated, no word boundaries |
| [EthioSign](https://ethiosign.org/) | App, video lessons, early access | Login-gated, no bulk access, copyrighted |

The honest problem: **almost none of it is isolated single-sign demonstration.**
It is compilations and continuous narrative. Extracting isolated words means a
human scrubbing a timeline and marking start/end for every instance — precisely
the manual cost the project can least afford.

Two further constraints. YouTube's terms prohibit bulk downloading, so building a
corpus this way is a decision the team should make deliberately, and
redistributing it is a separate and harder question. And a lesson video gives one
signer per sign, so it cannot supply the signer diversity that generalisation
actually needs.

## Step 3 — verdict for a 3-week timeline

**Viable path to a real EthSL isolated-word dataset from existing public data in 3 weeks? No.**

Assembling one from YouTube would require manual segmentation of hundreds of
instances, with a single signer per sign and unresolved licensing. That consumes
the whole three weeks and yields a corpus too narrow to generalise.

**Viable path to a working EthSL model in 3 weeks? Yes — by recording it.**

```
30 glosses x 25 takes           = 750 clips per signer
~4 s per take including reset   = ~50 min per signer
10 signers                      = ~8-9 hours total recording
landmark extraction             = automated
```

Eight or nine hours of recording spread across sessions is achievable in three
weeks. The binding constraint is **recruiting ~10 signers and verifying the signs
are correct EthSL** — an access-and-people problem, not an engineering one. Below
5 signers you cannot evaluate signer-invariance honestly, and the result is a demo
that works for whoever recorded it and fails on strangers. Published EthSL work
saw exactly this: 94% signer-dependent vs 73% signer-invariant with 7 signers.

Pursue CESL access in parallel — it is the largest real EthSL corpus and costs one
email. It is *continuous* sentence data, so it does not drop into an isolated-word
pipeline unchanged, but access takes days while recording takes weeks. Start both.

### On switching to ASL / WLASL

WLASL is the safer engineering bet and the worse product. Be clear about which one
is being optimised.

- It is genuinely large: ~2,000 glosses, ~21,000 videos, well documented.
- It is **not** frictionless. WLASL ships YouTube *URLs*, not videos, and suffers
  ongoing link rot; the maintainers acknowledge dead links and distribute the
  complete pre-processed set only through a request form with a terms agreement and
  a roughly 7-day turnaround. Budget for that, not for a clean `git clone`.
- Most importantly: **ASL is a different language from EthSL.** Training on WLASL
  does not produce an Ethiopian Sign Language translator. It produces an ASL
  translator. That is a change of product, not a change of dataset.

Recommendation: record a 20–30 gloss EthSL set yourselves — the only route to
something that is genuinely an Ethiopian translator inside the timeline — and use
WLASL only as optional pretraining if time allows. If the requirement is "a
working sign recognition demo" and signer access falls through, switching to ASL
is defensible; state it plainly in the writeup rather than presenting an ASL model
as an EthSL one.

### Immediate actions

1. Request CESL access at the Zenodo record — must come from the team, in your own name.
2. Recruit signers. This is the critical path; start before any more code is written.
3. Confirm the 30 glosses with a fluent signer, and re-check the AAU dictionary periodically to see whether it returns.
4. Prepare consent forms. This is identifiable face and body video and the repo is public.
