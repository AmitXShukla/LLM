"""
01_make_dataset.py  —  STEP 1 of weekend 2
==========================================
In weekend 1 you trained a model to *continue* text. That's called pretraining:
the model learns the shape of the language. But a pretrained model doesn't know
it's supposed to be helpful — hand it "Translate this verse" and it may just
ramble more Sanskrit at you.

Fine-tuning (specifically *supervised fine-tuning*, SFT) fixes that. We show the
model thousands of (instruction -> good answer) pairs until it learns the
*behaviour* "when asked X, respond with Y." The base model already knows the
language; we're only teaching it manners and a task.

So the unit of data changes. Weekend 1 wanted a big blob of raw text. Weekend 2
wants **pairs**:

    { "prompt": "<what the user asks>", "completion": "<the answer we want>" }

That's it. This file hand-builds a tiny set of such pairs so your pipeline runs
today, and shows you the format so you can scale it up with your real corpus.

Why "prompt" + "completion" as two separate fields (instead of one glued
string)? Because the trainer can then compute the loss on the *completion only*
— it scores the model on the answer, not on re-typing the question. More on that
in 02 and in TEACHING_finetune.md. This is the single most important idea in SFT,
and choosing this data format is what unlocks it.

Run:  python 01_make_dataset.py
Out:  sanskrit_sft.jsonl   (one JSON object per line)
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# A small, honest, hand-written dataset. Every verse here is ancient and
# public-domain; every English gloss is a plain-sense translation.
#
# Notice the VARIETY of instruction types. A model fine-tuned only on "translate"
# will only learn to translate. Mixing tasks (translate / explain / complete /
# define) teaches a more general "Sanskrit assistant" behaviour. This variety
# matters far more than raw count when your dataset is small.
# ---------------------------------------------------------------------------
PAIRS = [
    # --- task type 1: translate a verse ---------------------------------
    ("Translate this Sanskrit verse into English:\nकर्मण्येवाधिकारस्ते मा फलेषु कदाचन।",
     "You have a right to your action alone, never to its fruits."),
    ("Translate this Sanskrit verse into English:\nसर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः।",
     "May all beings be happy; may all be free from illness."),
    ("Translate this Sanskrit line into English:\nअहिंसा परमो धर्मः।",
     "Non-violence is the highest virtue."),
    ("Translate this Sanskrit line into English:\nयोगः कर्मसु कौशलम्।",
     "Yoga is skill in action."),
    ("Translate this Sanskrit verse into English:\nयत्र नार्यस्तु पूज्यन्ते रमन्ते तत्र देवताः।",
     "Where women are honoured, there the gods rejoice."),

    # --- task type 2: explain the meaning -------------------------------
    ("Explain the meaning of this verse in simple English:\nविद्या ददाति विनयं विनयाद्याति पात्रताम्।",
     "Knowledge gives humility; from humility one gains worthiness. It describes a "
     "chain in which true learning first makes a person humble, and that humility "
     "in turn makes them deserving of respect and success."),
    ("Explain the meaning of this line in simple English:\nउद्यमेन हि सिध्यन्ति कार्याणि न मनोरथैः।",
     "Tasks are accomplished through effort, not through mere wishes. It teaches "
     "that wanting something is not enough; only sustained action produces results."),
    ("What is the core teaching of this verse?\nवसुधैव कुटुम्बकम्।",
     "It means 'the whole world is one family.' It expresses a vision of universal "
     "kinship, encouraging us to treat all people as our own."),

    # --- task type 3: complete the shloka -------------------------------
    ("Complete this shloka:\nकर्मण्येवाधिकारस्ते मा फलेषु कदाचन।",
     "मा कर्मफलहेतुर्भूर्मा ते सङ्गोऽस्त्वकर्मणि॥"),
    ("Complete this shloka:\nसर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः।",
     "सर्वे भद्राणि पश्यन्तु मा कश्चिद्दुःखभाग्भवेत्॥"),

    # --- task type 4: define a word -------------------------------------
    ("What does the Sanskrit word 'धर्म' mean?",
     "'Dharma' means righteous duty, moral law, or the right way of living. It has "
     "no single English equivalent and covers duty, virtue, and cosmic order at once."),
    ("What does the Sanskrit word 'विद्या' mean?",
     "'Vidya' means knowledge or learning, especially knowledge that is cultivated "
     "and leads to wisdom."),
    ("What does the Sanskrit word 'योग' mean?",
     "'Yoga' means union or discipline — a method of joining or steadying the mind, "
     "and in the Gita, skillful and balanced action."),

    # --- task type 5: answer a question about Sanskrit ------------------
    ("In one sentence, what is a 'shloka'?",
     "A shloka is a classical Sanskrit verse, most often written in a two-line "
     "couplet of sixteen syllables per line."),
    ("What is the 'danda' (।) used for in Sanskrit writing?",
     "The danda (।) marks the end of a line or clause, and a double danda (॥) marks "
     "the end of a full verse — together they work like Sanskrit's full stops."),
]


def main():
    out = Path("sanskrit_sft.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for prompt, completion in PAIRS:
            # ensure_ascii=False keeps the Devanagari readable in the file
            f.write(json.dumps({"prompt": prompt, "completion": completion},
                               ensure_ascii=False) + "\n")

    print(f"Wrote {len(PAIRS)} examples -> {out}\n")
    print("Peek at the first example:")
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    print("  prompt    :", first["prompt"].replace("\n", " ⏎ "))
    print("  completion:", first["completion"])
    print("\nScaling up (this is your real weekend-2 work):")
    print("  • Aim for a few hundred to a few thousand pairs — quality over quantity.")
    print("  • Mix task types so the model generalises, not memorises.")
    print("  • You can bootstrap pairs from your OCR'd corpus + a strong model to")
    print("    draft translations/explanations, then hand-check them. (One published")
    print("    Sanskrit team generated ~11,000 synthetic Q&A pairs this way.)")


if __name__ == "__main__":
    main()
