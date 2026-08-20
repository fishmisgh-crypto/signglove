# Every route to EthSL training data, ranked

Companion to [data_availability_assessment.md](data_availability_assessment.md),
which establishes that nothing sufficient is openly downloadable. This lists
every remaining way to get data, ordered by return on effort.

## A. Ask the people who already have it

Cheapest by far. Each is one email. Send all of them the same week — they run in
parallel and cost nothing but a paragraph.

| # | Target | What you get | Why they may say yes |
|---|---|---|---|
| **A1** | Authors of the [Sci Reports 2025 skeleton framework](https://pmc.ncbi.nlm.nih.gov/articles/PMC12533181/) | **20 isolated words, 5,600 clips, 7 signers, as MediaPipe Holistic landmarks** | They state anonymised skeletal data "can be shared upon request to ensure reproducibility" |
| **A2** | Anteneh Yehalem — `artificialintel02@gmail.com` ([CESL](https://ethio-artifical.github.io/index.html)) | 1,320 videos, 22 signers, 92.7 GB | Zenodo record is public with restricted files — the request path is the intended one |
| **A3** | Corresponding author, [Heliyon 2024](https://www.cell.com/heliyon/fulltext/S2405-8440%2824%2914296-X) | 2,430 alphabet samples, 15 signers | Paper says "available from the corresponding author on reasonable request" |
| **A4** | Isayas Feyera, AASTU ([Mendeley set](https://data.mendeley.com/datasets/5d3nkyhsrf/1)) | Alphabet data; possibly unpublished word data | Already published openly once |
| **A5** | [AAU EthSL & Deaf Culture Study Unit](https://ethsl.aau.edu.et/) | The 1,322-sign dictionary corpus; pointers to 100+ undergrad and 10 postgrad projects | They built the dictionary; their server is down and they may share directly |

**A1 is the single highest-value action in this entire document.** It is already
isolated words, already MediaPipe landmarks, already the exact representation
`extract_landmarks.py` produces. If they say yes, you skip weeks of recording and
get 7 signers of diversity you cannot easily recruit yourself. Write it today.

What to say: who you are, that it is a student project, the specific artefact you
want, that you will cite them, and that you will not redistribute. Short.

## B. Ethiopian institutions and the Deaf community

Slower than email, but this is where signers come from — and you need signers
regardless of whether any dataset request succeeds.

| # | Target | Why |
|---|---|---|
| **B1** | Ethiopian National Association of the Deaf (ENAD) | Gatekeeper to the community; can legitimise the project and reach signers |
| **B2** | Schools for the Deaf — Mekanissa (Addis Ababa), Hosanna, Hawassa | Concentrations of fluent signers and teachers |
| **B3** | Ethiopian Center for Disability and Development (ECDD) | Named as a data source in prior AAU sign language theses |
| **B4** | [Visions Ethiopia Center for the Deaf](https://www.visionsethiopia.org/our-work), Hawassa | Already produces EthSL video; may collaborate rather than just be scraped |
| **B5** | AAU Department of Linguistics | Source of EthSL expertise for verifying your glosses are correct |
| **B6** | EBC / broadcast sign language interpreters | Professional signers, high fluency |

Approach these as collaborators, not data sources. A project that gives something
back — a working app, shared results, credit — gets access that a scraping
request does not. Ethical point, and also the practical one.

## C. Record it yourself — the reliable path

The only route fully under your control, and the only one guaranteed to produce
correct EthSL with the signer diversity that generalisation needs.

| # | Method | Notes |
|---|---|---|
| **C1** | Your own webcam | Working now. `python ML/record_clips.py --signer S01` |
| **C2** | Teammates | Each teammate is a signer id. Fastest way to 3–4 signers |
| **C3** | Recruited Deaf signers via B1–B4 | The ones that make the model actually work |
| **C4** | Remote/crowdsourced recording | Ship `record_clips.py` to a signer, they return the clip folder. Scales past who is physically nearby |
| **C5** | Phone cameras | Published EthSL work used "different Smart Mobile Phones" deliberately — device variety is a feature, not a defect |

Do not let the whole corpus be your team signing. A model trained on four
classmates learns those four people. C3 is what separates a demo from a result.

## D. Media-derived

Available, but expensive and legally unsettled. See the assessment doc for the
YouTube ToS and licensing constraints.

| # | Source | Difficulty |
|---|---|---|
| **D1** | "Learn Ethiopian Sign Language" YouTube playlists | Medium-hard — lesson compilations need manual segmentation |
| **D2** | EthSL story videos (Visions Global Empowerment) | Hardest — continuous, co-articulated signing |
| **D3** | TV news with interpreter inset | Hardest — continuous, small interpreter window, low resolution |
| **D4** | [EthioSign](https://ethiosign.org/) app | Login-gated, copyrighted, no bulk access |

Only worth it as a **reference for verifying signs**, not as a training corpus.
Every one of these gives a single signer per sign.

## E. Multiply what you have

These create no new information. They make a small corpus train better; they do
not substitute for signers.

| # | Technique | Caution |
|---|---|---|
| **E1** | Temporal resampling / speed variation | Safe and effective — signing speed genuinely varies |
| **E2** | Scale and translation jitter in landmark space | Safe |
| **E3** | Small rotation | Keep it small; large rotations create poses no camera would see |
| **E4** | Frame dropout | Simulates detection failures the app will actually hit |
| **E5** | Shoulder-width normalisation | Not augmentation but essential — removes body-size differences between signers |
| **E6** | Horizontal mirroring | **Use with care.** Flipping swaps dominant and non-dominant hand. For some signs this is fine (left-handed signers exist and you want robustness); for asymmetric two-handed signs it can produce something that is not the sign. Decide per-gloss, do not blanket-apply |

## F. Cross-lingual transfer

Borrow structure from other sign languages, then fine-tune on EthSL. Useful for
squeezing accuracy out of few EthSL clips; it does not make your model EthSL on
its own.

| # | Source | Status |
|---|---|---|
| **F1** | [WLASL](https://dxli94.github.io/WLASL/) (ASL, ~2,000 glosses, ~21k videos) | Ships YouTube URLs with real link rot; complete set via request form, ~7-day turnaround. Contact `dongxu.li@anu.edu.au` |
| **F2** | Kenyan Sign Language word-based pose dataset | Word-level, pose format, African SL — worth checking availability |
| **F3** | [sign-language-processing/datasets](https://github.com/sign-language-processing/datasets) | TFDS loaders across many SL corpora, `.pose` format — the fastest way to survey what else exists |
| **F4** | Any large isolated-SL corpus (AzSLD, Chinese SL, etc.) | Pretraining only |

The pretrain-then-fine-tune pattern is legitimate and worth doing if time allows.
Presenting a model trained only on ASL as an Ethiopian translator is not.

## Recommended plan

Run these concurrently, not in sequence:

1. **Today** — send A1, A2, A3 (three emails, ~20 minutes total).
2. **Today** — start recording C1/C2 with your team. Do not wait for replies.
3. **This week** — open B1/B2 conversations to recruit real signers.
4. **Ongoing** — apply E1–E5 once you have a first corpus.
5. **If time remains** — F1/F3 as pretraining.

The plan is deliberately not dependent on any single request succeeding. If A1
lands you are far ahead; if nothing lands, C still gets you a working model.
