# docs — reference material

This folder is for reference material that supports the book but is **not**
part of the book itself. MyST does not build anything in here, so nothing you
put here will appear on the website.

## What goes here

- Research papers as PDFs
- Technical reports from model releases
- Grammar references, tokenizer studies, corpus documentation
- Your own notes, diagrams, and working files
- Anything you want to keep near the project but not publish

## How to organise it

```
docs/
├── papers/          # research papers, named author-year-topic.pdf
├── reports/         # model and technical reports
├── grammar/         # Sanskrit and Urdu language references
├── notes/           # your own working notes
└── references.md    # an index of everything, with links
```

Create these folders as you need them.

## Naming

Use lowercase with hyphens, and start with the year:

```
2024-shao-grpo-deepseekmath.pdf
2025-guo-deepseek-r1.pdf
2026-rana-indic-supertokenizer.pdf
```

This keeps the folder sorted by date automatically.

## Important: only commit what you may redistribute

Many papers are free to read but **not** free to redistribute. Publisher PDFs
in particular usually may not be copied into a public repository.

Safe:
- arXiv preprints (check the specific licence on the paper's arXiv page)
- Papers you wrote
- Openly licensed reports and documentation
- Your own notes

Not safe:
- Paywalled journal PDFs
- Scanned copyrighted books
- Anything marked "all rights reserved"

**When in doubt, do not commit the file.** Add a link and a short summary to
`references.md` instead. A good summary of a paper is often more useful to a
reader than the paper itself.

Large PDFs also bloat the repository permanently, because Git keeps every
version forever. If a file is over about 10 MB, link to it rather than commit
it.
