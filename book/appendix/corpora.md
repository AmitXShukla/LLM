---
title: Where to find Sanskrit and Urdu text
short_title: Corpora
---

# Where to find Sanskrit and Urdu text

A starting list. Check the licence on everything before you use it, and record
where each file came from as you go — see [Step 6](../06-collect-data.md).

:::{note} Keep this page updated
This is the page most likely to go stale. If you find a source that is not
listed, or one that has moved, please open a pull request.
:::

---

## Sanskrit

### Digital text archives

- **GRETIL** (Göttingen Register of Electronic Texts in Indian Languages) — the
  largest single collection of digitised Sanskrit texts. Variable encoding
  quality; check each text.
- **SARIT** (Search and Retrieval of Indic Texts) — carefully marked-up
  scholarly editions. Smaller but cleaner.
- **The Digital Corpus of Sanskrit (DCS)** — importantly, this one is
  **morphologically annotated**, which makes it valuable for the grammatical
  tasks in [Step 10](../10-evaluation.md) and
  [Step 15](../15-panini-neurosymbolic.md).
- **Sanskrit Wikipedia and Wikisource** — modern and classical, freely licensed,
  though the total volume is small.

### Institutional projects

Large digitisation efforts are running, including a major collaboration in
Chennai involving IIT Madras and Madras Sanskrit College, working through more
than 110,000 rare manuscripts.

These projects usually publish their datasets and benchmarks openly. **Check
what they have released before you start collecting.** They are doing the
expensive part — digitisation — and you can build on it.

### Existing models worth studying

Not for using directly, but for reading their data and tokenizer decisions:

- **Paramanu-Sanskrit** — a small from-scratch Sanskrit model, around 139
  million parameters
- **ByT5-Sanskrit** — focused on lower-level tasks like sandhi splitting and
  parsing
- Various Llama-family fine-tunes adapted for Sanskrit

---

## Urdu

- **Urdu Wikipedia** dumps
- **Common Crawl** language splits — large and messy
- **Rekhta** and similar poetry archives — check terms of use carefully
- Urdu news archives
- **Roman Urdu** datasets, if you decided in [Step 5](../05-urdu-tokenizer.md)
  that Roman Urdu is in scope

---

## Multilingual collections that include both

- **FineWeb-2** language splits
- **OSCAR**
- **IndicCorp** and related AI4Bharat collections
- **CulturaX**

---

## Related projects worth knowing about

- **AI4Bharat** (IIT Madras) — IndicTrans2, multilingual Indic models,
  benchmarks such as MILU
- **Bhashini** — the Indian government's national language technology mission,
  covering 22-plus scheduled languages
- **BharatGen** and other indigenous model efforts

These treat each language as one among many. That is exactly the gap described
in the [introduction](../intro.md), and exactly why their open data is useful to
you while their models are not the last word.

---

## Before you use any of it

1. **Check the licence.** Free to read is not the same as free to redistribute
   or free to train on.
2. **Record the source.** Every file, every time.
3. **Check the encoding.** Older archives use a variety of transliteration
   schemes and legacy encodings.
4. **Deduplicate across sources** — see [Step 7](../07-clean-data.md). The same
   verse appears everywhere.
