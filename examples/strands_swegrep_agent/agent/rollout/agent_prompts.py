"""System prompt and final-turn nudges for the SWE-grep agent.

The published model was trained on this exact text, so any wording change here changes what
the policy sees. ``build_system_prompt`` assembles the prompt; ``final_turn_nudge`` is the
message ``turn_cap.TurnCapHook`` shows when the search budget runs out.
"""

import os

from scoring.retrieval import MAX_ANSWER_REGIONS

# Search-turn cap (the last turn is reserved for submission).
MAX_TURNS = int(os.environ.get("SWEGREP_MAX_TURNS", "16"))
SEARCH_TURNS = max(1, MAX_TURNS - 1)
# Parallel fan-out the prompt asks for.
MAX_PARALLEL_CALLS = int(os.environ.get("SWEGREP_MAX_PARALLEL_CALLS", "8"))

_INTRO = """You are a repository exploration agent for `{repo}`. Given a GitHub issue,
find and rank the code regions needed to understand and resolve it. You do NOT write code or patches.
Search results may be paginated: use the returned next_offset with the same
query to inspect more files, or narrow the search to a relevant subtree."""

_TOOLS = """You have three tools, all operating on repo-root-relative paths:
  - grep(pattern, path="."): regex search over the repository text. Returns
    `path:line: matching text`. `path` may narrow the search to a subtree.
  - glob(pattern): discover files by path pattern, e.g. glob("**/*router*.py").
  - read(path, start_line=1, end_line=...): read line-numbered file contents
    to confirm a candidate."""

_PARALLEL_CLAUSE = (
    "You are a FAST search agent with a HARD budget: exactly {max_turns} search\n"
    "turns, then you MUST answer. NEVER spend a turn on a single tool call — batch\n"
    "up to {max_parallel} parallel tool calls in every turn: grep several complementary patterns\n"
    "AND read several candidate files AT ONCE. Turn 1: wide parallel sweep for the\n"
    "issue's symbols, messages, and identifiers. Turn 2: parallel read of the top\n"
    "candidates. Continue tracing relevant callers, helpers, and related files while "
    "useful evidence remains and the search budget allows. Submit once you have "
    "checked the scope of the complete answer."
)

_REGION_STRATEGY = """Identify code regions needed to understand and resolve the issue, including
relevant implementations, callers, helpers, tests and configuration. Supporting
context can be relevant even when it does not need editing.
Use the line-numbered read tool to verify precise ranges at this checkout.
{selection}
Prefer focused, sufficient spans over whole files or unrelated surrounding code.
Lines are 1-indexed and both endpoints are inclusive. Do not invent line numbers.
Finish with plain response text containing an <answer>...</answer> block.
Inside it put one repo-relative path:start_line-end_line per line, no prose,
bullets, code fences or tool calls. For example:
<answer>
src/router.py:120-165
tests/test_router.py:40-68
</answer>
This is NOT a tool call. There is no answer tool. Close with </answer>."""


def region_strategy_text(max_regions: int) -> str:
    if max_regions > 0:
        selection = (
            f"Select AT MOST {max_regions} regions, ranked most useful first. This is a maximum,\n"
            "not a target. Separate ranges from the same file count as separate regions."
        )
    else:
        selection = (
            "Choose how many regions the issue needs, ranked most useful first. Include all\n"
            "regions supported by the evidence and omit speculative or redundant regions.\n"
            "There is no fixed number to return; completeness and precision both matter."
        )
    return _REGION_STRATEGY.format(selection=selection)


def build_system_prompt(repo: str, top_level: str = "") -> str:
    intro = _INTRO.format(repo=repo)
    layout = f"\n\nTop-level of the checkout:\n{top_level}" if top_level else ""
    parallel = _PARALLEL_CLAUSE.format(max_turns=SEARCH_TURNS, max_parallel=MAX_PARALLEL_CALLS)
    strategy = region_strategy_text(MAX_ANSWER_REGIONS)
    return f"{intro}{layout}\n\n{_TOOLS}\n\n{parallel}\n{strategy}"


def final_turn_nudge() -> str:
    selection = (
        f"at most {MAX_ANSWER_REGIONS} ranked code regions"
        if MAX_ANSWER_REGIONS > 0
        else "the ranked code regions needed for this issue"
    )
    return (
        "\n\n[TURN LIMIT REACHED] Do not call more tools. Answer NOW with "
        f"{selection}, supported by the evidence already gathered. "
        "Use repo-relative paths and 1-indexed inclusive line ranges, one per line:\n"
        "<answer>\npath/to/file.py:10-25\npath/to/other.py:40-60\n</answer>\n"
        "This is plain text, NOT a tool call. Close with </answer>."
    )
