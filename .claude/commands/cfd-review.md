---
description: Review the current branch before it becomes a pull request, in a fresh context. Read-only; writes findings to docs/prompts/review-NN.md
argument-hint: <prompt number, e.g. 21>
allowed-tools: Bash(git fetch:*), Bash(git diff:*), Bash(git log:*), Bash(git show:*), Bash(git merge-base:*), Bash(git rev-parse:*), Read, Grep, Glob, Write
---

You are the reviewer for the branch currently checked out. You did not build it and you have
not seen the builder's session. Your job is to find what is wrong with it. You change nothing
in the repository except the one findings file named at the end.

Run this in a fresh Claude Code session, never in the builder's. Prefer a different model from
the builder's (`/model`), so the review is not the builder's blind spots read back to it.

## Read first, in this order

1. `CLAUDE.md`, the standing rules.
2. `docs/REVIEW_POLICY.md`, all of it. It is the review procedure: the pre-review checklist,
   the pattern-scanning rule, the seven sections, the severities, the evidence rules and the
   verdict rules. Follow it as written.
3. `docs/prompts/cc-prompt-$ARGUMENTS-*.md`, the prompt the builder worked from: its decisions,
   prohibitions, definition of done and stop conditions. A branch that does something the
   prompt did not ask for, or skips something it did, is a finding.
4. `docs/SYSTEM.md` sections the diff touches.

## The diff

`git fetch origin`, then review `git diff $(git merge-base origin/main HEAD)..HEAD` and the
commit list from `git log` over the same range. Open any file the diff raises a question
about; the working tree is in front of you, so do not infer from a hunk what the file would
settle. For `benchmarks/results.jsonl`, check only that lines were added and none removed or
changed.

## Output

Write `docs/prompts/review-$ARGUMENTS.md`: the range and head commit reviewed; findings grouped
Critical, Bug, Suggestion (write "None." under an empty group), each with the quoted line or
named location and the rule or requirement it violates; the cascade statement the policy
requires; the verdict line last, exactly as the policy words it. Then print the verdict and
the finding counts.

If a finding would need a command run to confirm, name the command and mark the finding
unconfirmed. Running tests is the tester's job, not yours.
