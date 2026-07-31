---
title: "Step 25 — Release your model responsibly"
short_title: "25. Release"
---

# Step 25 — Release your model responsibly

**Goal:** put your work into the world in a way that helps people and does not
create problems for you.

---

## Why this step matters

Open release is how a lone builder gets leverage. Publish your weights,
tokenizer, cleaned data, and evaluation set, and people who care about the
language will find you. Scholars will correct you. Contributors will bring data
you could not have collected alone.

That only works if the release is honest and well documented.

---

## What you do

### 1. Test safety in Sanskrit and Urdu, not in English

:::{warning} Most people do not know this
Safety training transfers across languages **far worse** than general ability
does.

A model that correctly refuses a harmful request in English will often comply
with the same request in Urdu. The refusal behaviour was trained mostly in
English, and it does not follow the language across.

If you only tested in English, you have not tested.
:::

### 2. Red-team your own model

Try to make it fail. Write down what worked. For a model dealing with religious
and philosophical text, pay particular attention to confident fabrication — a
made-up verse presented as authentic is a serious failure.

### 3. Sort out your licences

You have four to consider:

1. Your **data's** licence
2. Your **base model's** licence
3. The licence of any **teacher model** whose outputs you used
4. The licence **you** release under

They stack, and the most restrictive one wins. Work this out before release,
not after.

### 4. Write an honest model card

Include:

- What it does well
- What it does badly
- What data it saw
- How you evaluated it
- The failure modes you found in step 2

A model card that lists real weaknesses is more trustworthy, not less. It also
saves you from people discovering those weaknesses publicly and loudly.

### 5. Release properly

- Weights, in both full and quantized forms
- The tokenizer
- Your evaluation results **and** the evaluation set itself
- The training configuration
- The data recipe, if you can

### 6. Make it trivially easy to run locally

Publish GGUF versions. Make sure it works with the common local tools. If
someone has to fight your model for an hour, they will not.

### 7. Make it reproducible

Random seeds, exact library versions, hardware used, and the exact command you
ran.

### 8. Release the evaluation set and the fertility numbers separately

Honestly, these may be more useful to the community than your model. Your model
will be superseded within a year. A careful, native-checked evaluation set for
classical Sanskrit will still be useful in ten.

### 9. Document the journey

If you write, record, or stream, do it. Not for vanity — for recruitment.

Sanskrit and Urdu both have passionate communities who are not AI engineers but
who have exactly the domain knowledge you lack. A public build log is how they
find you. Some of the most valuable contributions to a project like this come
from people who cannot write a training loop but can tell you instantly that
your model's output is grammatically wrong.

---

## You are ready to move on when

Someone you have never met has downloaded your model, run it, and told you
something about it that you did not know.

---

:::{seealso} Related
- [Step 10](10-evaluation.md) — the evaluation set you are releasing
- [Contributing](https://github.com/YOUR_GITHUB_USERNAME/sanskrit-llm-book/blob/main/CONTRIBUTING.md)
:::
