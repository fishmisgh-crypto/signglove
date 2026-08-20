# Data request drafts

Requests 1, 2, 3 and 5 were **sent on 2026-08-20** from fishmisgh@gmail.com.
Request 4 is still unsent — no public address exists for the AAU unit.

Kept here as the record of what was asked and as templates for follow-ups.
Signature style: institution in the opening line only, not repeated at the end.

---

## 1. EthSL isolated-word skeletal data — highest value

**To:** Dagne Walle Girmaw (corresponding author), Department of Information
Technology, Haramaya University — email is on the paper's author page.
**Cc:** Million Meshesha (Addis Ababa University), Deme Kuma Gonfa.

Paper: *A deep learning framework for Ethiopian sign language recognition using
skeleton-based representation*, Scientific Reports 15:36181 (2025),
DOI 10.1038/s41598-025-19937-0.

Why this one first: they hold **20 isolated words, 5,600 clips, 7 signers**,
already extracted as MediaPipe Holistic landmarks — the exact representation our
pipeline produces. The paper explicitly offers it.

> **Subject:** Request for anonymised skeletal data — EthSL recognition (Sci Rep 15:36181)
>
> Dear Dr Girmaw,
>
> I am Fisehatsion Misganaw, a summer camp student at the Ethiopian Artificial Intelligence Institute, working on a student project
> building an Ethiopian Sign Language to text and speech application for isolated
> word recognition.
>
> I read your Scientific Reports paper on skeleton-based EthSL recognition with
> great interest, particularly the signer-invariant evaluation. The paper notes
> that anonymised skeletal representations derived from MediaPipe Holistic can be
> shared on request for reproducibility. I would like to request access to that
> data for the 20 isolated words.
>
> Our pipeline extracts MediaPipe Holistic landmarks in the same way, so your
> representation would be directly usable and would let us compare against your
> reported results rather than starting from nothing.
>
> To be clear about intent: this is an unfunded student project. We would cite
> the paper in any writeup, would not redistribute the data, and are happy to
> agree to any conditions you set on its use. We can also share our results back
> with you.
>
> Thank you for considering this, and for making the work available.
>
> Fisehatsion Misganaw
> fishmisgh@gmail.com

---

## 2. CESL continuous corpus

**To:** Anteneh Yehalem — `artificialintel02@gmail.com`
Record: <https://zenodo.org/records/10800699> (files restricted)

1,320 videos, 22 signers, 92.7 GB. Continuous sentences, not isolated words — it
will not drop into an isolated-word pipeline unchanged, but 22 signers of real
EthSL is worth having and the request costs nothing.

> **Subject:** Access request — Continuous Ethiopian Sign Language (CESL) dataset
>
> Dear Mr Yehalem,
>
> I am Fisehatsion Misganaw, a summer camp student at the Ethiopian Artificial Intelligence Institute. We are building an Ethiopian
> Sign Language recognition application as a student project.
>
> I found the CESL dataset on Zenodo (DOI 10.5281/zenodo.10800699) and would like
> to request access to the files, which are currently restricted. Our work is on
> isolated word recognition rather than continuous signing, so we would primarily
> use the corpus for its signer diversity and as a reference for how signs are
> produced in context.
>
> This is an unfunded student project. We would cite the dataset, would not
> redistribute it, and will follow any conditions you place on its use.
>
> If a smaller subset is easier to share than the full 92.7 GB, that would be very
> welcome — our connection is slow.
>
> Thank you for creating and publishing this resource.
>
> Fisehatsion Misganaw
> fishmisgh@gmail.com

---

## 3. Derived Amharic alphabet dataset

**To:** corresponding author, Heliyon (2024), DOI 10.1016/j.heliyon.2024.e38265

2,430 samples, 15 signers, 7 derived alphabet signs. Alphabet rather than words,
so lower priority — but it is fingerspelling data you cannot get elsewhere, and
the paper says it is available on request.

> **Subject:** Data request — derived Amharic alphabet sign language dataset
>
> Dear Dr Salau,
>
> I am Fisehatsion Misganaw, a summer camp student at the Ethiopian Artificial Intelligence Institute, working on an Ethiopian Sign
> Language recognition project.
>
> Your Heliyon paper on derived Amharic alphabet sign recognition notes that the
> dataset is available from the corresponding author on reasonable request. I
> would like to request a copy for use in a student project on EthSL recognition.
>
> We would cite the paper, would not redistribute the data, and are happy to
> accept any conditions on its use.
>
> Thank you for your time.
>
> Fisehatsion Misganaw
> fishmisgh@gmail.com

---

## 4. AAU EthSL & Deaf Culture Study Unit

**To:** Addis Ababa University EthSL unit — contact via <https://ethsl.aau.edu.et/>

Two purposes: their digital dictionary server (`ethsld.aau.edu.et`) has been
returning HTTP 502, and they are the authority who can confirm our gloss list is
correct EthSL. Note Million Meshesha (AAU) is also a co-author on request 1.

> **Subject:** EthSL digital dictionary unavailable — and a question on sign verification
>
> Dear EthSL Study Unit,
>
> I am Fisehatsion Misganaw, a summer camp student at the Ethiopian Artificial Intelligence Institute, working on an Ethiopian Sign
> Language recognition project.
>
> Two things. First, the digital dictionary at ethsld.aau.edu.et/dictionary/ has
> been returning a server error (HTTP 502) whenever we have tried it. I wanted to
> flag it in case you were not aware.
>
> Second, we are assembling a 30-word vocabulary for a recognition prototype and
> want to be certain we are recording correct EthSL rather than approximations.
> Is there someone in the unit who could review our word list, or could you point
> us to an authoritative reference while the dictionary is down?
>
> We would be glad to share what we build.
>
> Thank you,
>
> Fisehatsion Misganaw
> fishmisgh@gmail.com

---

## 5. WLASL preprocessed videos

**Primary route:** the video request Google Form linked from the
[WLASL README](https://github.com/dxli94/WLASL) — submit that first, it is the
process the maintainers actually run and it carries the terms agreement.
**Follow-up only:** Dongxu Li — `dongxu.li@anu.edu.au`, the address the README
gives for urgent cases. Do not email before submitting the form.

Why we need it rather than the downloader script: WLASL ships URLs, not video,
and from our network 11 of the 13 hosts behind our 29 overlapping glosses are
unreachable — YouTube times out, signingsavvy returns 403, and most dictionary
sites do not respond at all. Only about 13% of the clips are fetchable here.
The preprocessed set sidesteps all of it, and it resolves the licensing
question too: those clips belong to third-party dictionary sites, which is
exactly why WLASL cannot redistribute them itself.

**What was actually sent** differed from the draft below: the version emailed on
2026-08-20 was rewritten as a first contact, because claiming a form submission
that had not happened would have been false. It asks Dr Li to point us at the
form if one is required.

> **Subject:** WLASL video request follow-up — Fisehatsion Misganaw
>
> Dear Dr Li,
>
> I submitted the WLASL video request form on [date] under the name Fisehatsion Misganaw
> (fishmisgh@gmail.com) and wanted to follow up briefly.
>
> I am a summer camp student at the Ethiopian Artificial Intelligence Institute working on an isolated-word sign
> language recognition project. We are studying the overlap between ASL and
> Ethiopian Sign Language, which shares some historical lexicon with ASL through
> the deaf schools where EthSL developed, and WLASL is the reference corpus for
> the ASL side.
>
> We tried the start_kit downloader first. From our network most of the source
> hosts are unreachable — YouTube requests time out, signingsavvy returns 403,
> and several dictionary sites do not respond — so we can retrieve only a small
> fraction of the clips. We would be grateful for access to the preprocessed
> videos.
>
> We need roughly 29 glosses rather than the full 2,000: everyday vocabulary
> such as hello, water, help, doctor, hospital, mother, father. A subset would
> be very welcome if that is easier to share than the whole corpus.
>
> We accept the terms of use, will cite the WACV 2020 paper, and will not
> redistribute the data.
>
> Thank you for maintaining WLASL and for making it available.
>
> Fisehatsion Misganaw
> fishmisgh@gmail.com

---

## Tracking

| # | Target | Address | Sent | Reply |
|---|---|---|---|---|
| 1 | Girmaw — 5,600 clips, 7 signers, landmarks | dagnewalle143@gmail.com (verified via PLOS One) | 2026-08-20 | |
| 2 | Yehalem — CESL, 22 signers | artificialintel02@gmail.com | 2026-08-20 | |
| 3 | Salau +2 — 2,430 alphabet samples | ayodejisalau98@gmail.com | 2026-08-20 | |
| 4 | AAU EthSL unit — dictionary + sign verification | **none found** | not sent | |
| 5 | Li — WLASL preprocessed clips | dongxu.li@anu.edu.au | 2026-08-20 | |

A correction was sent on each of 1, 2, 3 and 5: the first round went out under an
incorrect name.

Requests 1–4 are for **EthSL** data and are what make this an Ethiopian
translator. Request 5 is for **ASL** data, useful only for the glosses a fluent
signer confirms are shared. Do not let 5 arriving first quietly turn the project
into an ASL demo.

Request 4 still needs a recipient. Routes: the contact form at
<https://ethsl.aau.edu.et/>, the AAU School of Information Science Facebook page,
or an introduction from Million Meshesha, who co-authored the paper behind
request 1.

Request 4 is not admin. It is the one that asks a fluent signer to confirm the 30
glosses are real EthSL rather than ASL borrowings — the check that decides whether
the downloaded ASL clips are training data or decoration.
