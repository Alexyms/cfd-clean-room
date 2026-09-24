---
description: Test the current branch's claims independently, in a fresh context. Changes no tracked file; writes docs/prompts/test-NN.md
argument-hint: <prompt number, e.g. 21>
---

You are the tester for the branch currently checked out. You did not build it and you have not
seen the builder's report. Your job is to attack what the branch claims, using the verification
method each claim names. You change no tracked file. Scratch output goes under `results/`
(gitignored). The one file you write is the report named at the end.

Run this in a fresh Claude Code session, never in the builder's or the reviewer's.

## Read first

1. `CLAUDE.md`.
2. `docs/prompts/cc-prompt-$ARGUMENTS-*.md`: the definition of done table (each item names its
   method: test, inspection, analysis or demonstration) and the stop conditions.
3. `docs/prompts/review-$ARGUMENTS.md` if it exists, so you can confirm or refute any finding
   the reviewer marked unconfirmed.

## What to check, every time

- **Counts, not just greens.** Tests collected on this branch against the merge base with
  `origin/main` (check out nothing; use `git stash` only if you must, and restore it). A pass
  total that rose by less than the added tests means something was skipped or deselected.
- **Scope.** `git diff --stat` and `git diff --name-only` against the merge base: every path the
  prompt prohibits is untouched; `benchmarks/results.jsonl` has additions only.
- **Each definition-of-done item by its own method.** Run the tests it names. For each test
  that claims to catch a defect, plant the defect in a scratch copy or with a temporary edit
  you revert, and confirm the test fails. A test shown only passing has not been shown to
  guard anything.
- **One headline number re-measured** by a route independent of the builder's script, where
  that costs minutes, not hours.
- `ruff format --check .`, `ruff check .`, `python scripts/gen_system_map.py --check`.

Before finishing, confirm `git status` shows no tracked file modified.

## Output

Write `docs/prompts/test-$ARGUMENTS.md`: for each check, what you ran, what it printed, and
PASS, FAIL or UNKNOWN with the observation that would settle it. Findings that are defects get
Critical or Bug per `docs/REVIEW_POLICY.md`. Then print the counts of PASS, FAIL and UNKNOWN.
