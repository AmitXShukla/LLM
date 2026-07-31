# Contributing

Thank you for wanting to help. This book gets better when people who actually
tried the steps report back.

## The most useful contributions

Ranked by how much they help:

1. **"I followed Step N and it did not work."** Tell us what you ran, what you
   expected, and what happened. Negative results save other people days.
2. **Fertility measurements** for a language not yet covered. See Step 3 for
   the method. A table of tokenizer versus tokens-per-word on real text is a
   real contribution.
3. **Evaluation tasks written by native speakers.** See Step 10. These barely
   exist for most of the languages in this book.
4. **Corrections.** Technical errors, broken links, and unclear wording.
5. **New chapters or sections.** Please open an issue first so we can agree on
   the shape before you write it.

## House style

This book is written for readers whose first language may not be English, and
for readers who are new to this field. Please match the style:

- **Short sentences.** One idea per sentence.
- **Plain words.** Write "use" not "utilise". Write "about" not "approximately".
- **Explain every technical term the first time it appears.** Then add it to
  the glossary in `book/appendix/glossary.md`.
- **Use the chapter template.** Every step chapter has the same five sections:
  Goal, Why this step matters, What you do, Where people usually get stuck,
  You are ready to move on when.
- **Prefer a plain analogy over a formal definition** when introducing an idea.
- **Be honest about costs and failures.** If something is expensive, slow, or
  usually goes wrong, say so.

## Writing a new chapter

1. Copy `book/_template.md` to `book/NN-your-chapter.md`.
2. Fill in the five sections.
3. Add one line for the file in the `toc:` section of `myst.yml`.
4. Run `myst start` and check that it renders.
5. Open a pull request.

## Adding reference material

Put PDFs, papers, and notes in `docs/`. See `docs/README.md` for the rules.

**Only commit files you have the right to redistribute.** When in doubt, add a
link and a short summary in `docs/references.md` instead of the file itself.

## Adding code

Runnable examples go in `code/`, in a folder named after the step, for example
`code/step-01-tiny-transformer/`. Each folder should have its own short
`README.md` saying how to run it and roughly how long it takes.

## Before you open a pull request

- Run `myst build --html` and confirm it completes without errors.
- Check that any new term you introduced is in the glossary.
- Keep pull requests focused. One chapter, or one fix, per pull request.
