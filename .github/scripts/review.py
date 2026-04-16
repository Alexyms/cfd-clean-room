"""Automated code review using Claude Sonnet with Opus advisor.

Reads the PR diff and project context documents, sends them to Claude
for review, and posts the review as a GitHub PR review. Requests changes
if issues are found, approves if the PR is clean.
"""

import json
import os
import sys
from pathlib import Path

import anthropic
from github import Github


def load_file(path: str) -> str:
    """Load a file and return its contents, or empty string if missing."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def load_system_prompt() -> str:
    """Load the review system prompt from the prompts directory."""
    prompt_path = Path(".github/prompts/review_system_prompt.md")
    if not prompt_path.exists():
        print("ERROR: review_system_prompt.md not found")
        sys.exit(1)
    return prompt_path.read_text(encoding="utf-8")


def build_review_message(
    diff: str,
    pr_meta: dict,
    context_docs: dict[str, str],
    truncated: bool,
) -> str:
    """Assemble the user message for the review request."""
    sections = []

    sections.append(f"# Pull Request: {pr_meta.get('title', 'Unknown')}")
    sections.append(f"**Branch:** {pr_meta.get('headRefName', 'unknown')}")

    pr_body = pr_meta.get("body", "")
    if pr_body:
        sections.append(f"**PR Description:**\n{pr_body}")

    sections.append("---")

    for doc_name, doc_content in context_docs.items():
        if doc_content.strip():
            sections.append(f"## Context: {doc_name}\n\n{doc_content}")

    sections.append("---")
    sections.append("## PR Diff\n")

    if truncated:
        sections.append(
            "*Note: This diff was truncated to fit context limits. "
            "Review may be incomplete for very large PRs.*\n"
        )

    sections.append(f"```diff\n{diff}\n```")

    return "\n\n".join(sections)


def run_review(message: str, system_prompt: str) -> str:
    """Call Claude API with Sonnet executor and Opus advisor."""
    client = anthropic.Anthropic()

    tools = [
        {
            "type": "advisor_20260301",
            "name": "advisor",
            "model": "claude-opus-4-6",
        }
    ]

    response = client.beta.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        betas=["advisor-tool-2026-03-01"],
        system=system_prompt,
        tools=tools,
        messages=[{"role": "user", "content": message}],
        output_config={"effort": "high"},
    )

    # Extract text content from response, handling advisor tool blocks
    review_text = ""
    for block in response.content:
        if block.type == "text":
            review_text += block.text

    return review_text


def parse_verdict(review_text: str) -> str:
    """Extract the review verdict from the review text.

    Looks for VERDICT: APPROVE or VERDICT: REQUEST_CHANGES in the
    review output. Defaults to REQUEST_CHANGES if unclear.
    """
    upper = review_text.upper()
    if "VERDICT: APPROVE" in upper:
        return "APPROVE"
    return "REQUEST_CHANGES"


def post_review(review_text: str, verdict: str, pr_number: int) -> None:
    """Post the review as a GitHub PR review."""
    gh = Github(os.environ["GITHUB_TOKEN"])
    repo = gh.get_repo(os.environ["REPO_FULL_NAME"])
    pr = repo.get_pull(pr_number)

    # GitHub API review events
    event = "APPROVE" if verdict == "APPROVE" else "REQUEST_CHANGES"

    pr.create_review(
        body=review_text,
        event=event,
    )

    print(f"Review posted with verdict: {event}")


def main() -> None:
    """Run the full review pipeline."""
    pr_number = int(os.environ["PR_NUMBER"])
    diff_truncated = os.environ.get("DIFF_TRUNCATED", "false") == "true"

    # Load inputs
    diff = load_file("pr_diff.txt")
    if not diff.strip():
        print("No diff found. Skipping review.")
        return

    pr_meta = {}
    meta_path = Path("pr_meta.json")
    if meta_path.exists():
        pr_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Load context documents
    context_docs = {
        "claude.md (Coding Standards)": load_file("review_context/claude.md"),
        "SYSTEM.md (Architecture)": load_file("review_context/SYSTEM.md"),
        "PROJECT_PLAN.md (Phase Tracking)": load_file("review_context/PROJECT_PLAN.md"),
    }

    # Build prompt
    system_prompt = load_system_prompt()
    message = build_review_message(diff, pr_meta, context_docs, diff_truncated)

    # Check message size (rough token estimate: 4 chars per token)
    estimated_tokens = len(message) // 4
    print(f"Estimated input size: {estimated_tokens} tokens")
    if estimated_tokens > 150000:
        print("WARNING: Input may exceed context window. Review may be degraded.")

    # Run review
    print("Sending to Claude for review...")
    review_text = run_review(message, system_prompt)

    if not review_text.strip():
        print("ERROR: Empty review response")
        sys.exit(1)

    # Post review
    verdict = parse_verdict(review_text)
    post_review(review_text, verdict, pr_number)


if __name__ == "__main__":
    main()
