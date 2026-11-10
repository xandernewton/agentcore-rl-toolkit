"""The retrieval task: parse an agent's answer and score it against gold files.

Single source of the scoring maths: the training reward (``reward``) and the benchmark
harness (``eval/full_benchmark.py``) both import from here so they cannot drift.

Answers are ranked line regions; their distinct file paths feed the file-level diagnostics.
"""

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any

from scoring.regions import Region, parse_regions, validate_regions

# Also accept a bare "path:line" as a one-line region instead of marking it invalid.
ACCEPT_SINGLE_LINE = os.environ.get("SWEGREP_ACCEPT_SINGLE_LINE", "1") == "1"
# Only the first N submitted regions are kept (0 = no cap).
MAX_ANSWER_REGIONS = int(os.environ.get("SWEGREP_MAX_ANSWER_REGIONS", "0"))
if MAX_ANSWER_REGIONS < 0:
    raise ValueError("SWEGREP_MAX_ANSWER_REGIONS must be nonnegative (0 = no cap)")

# The answer format: the text inside an <answer> block.
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def normalize_path(p: str, repo_root: str = "") -> str:
    """Canonicalize for comparison against repo-root-relative gold: strip an
    absolute ``repo_root`` prefix, then posix-normalize."""
    p = str(p or "").strip().replace("\\", "/")
    root = str(repo_root or "").strip().replace("\\", "/").rstrip("/")
    if root and p.startswith(root + "/"):
        p = p[len(root) + 1 :]
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def normalize_list(paths, repo_root: str = "") -> list[str]:
    """Normalized paths in submission order, first occurrence wins, blanks dropped."""
    return list(dict.fromkeys(n for p in paths or [] if (n := normalize_path(p, repo_root))))


def normalize_set(paths, repo_root: str = "") -> set[str]:
    return set(normalize_list(paths, repo_root))


@dataclass
class Answer:
    files: list[str] = field(default_factory=list)
    submitted: bool = False
    num_turns: int = 0  # assistant turns holding a search tool call
    max_parallel_calls: int = 0  # widest single turn
    regions: list[Region] = field(default_factory=list)
    invalid_regions: list[str] = field(default_factory=list)


def _tool_calls(content: Any) -> list[dict]:
    blocks = content if isinstance(content, list) else []
    return [block["toolUse"] or {} for block in blocks if isinstance(block, dict) and "toolUse" in block]


def search_call_count(message: dict[str, Any]) -> int:
    """Tool calls (all of them searches) in an assistant message."""
    return len(_tool_calls(message.get("content", [])))


def _text_answer(content: Any) -> str | None:
    """The last <answer> block in an assistant message's text."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and "text" in b)
    else:
        text = ""
    # Native SGLang emits reasoning in the text block. The opening <think>
    # can belong to the prompt, so a closing delimiter alone is sufficient.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.split("<think>", 1)[0]
    if "<tool_call>" in text:
        return None
    matches = ANSWER_RE.findall(text)
    return matches[-1] if matches else None


def _final_block(message: dict[str, Any]) -> str | None:
    """The raw answer in the last assistant message. Only that message can answer,
    so the prompt's own example echoed mid-conversation never parses."""
    content = message.get("content", [])
    return None if _tool_calls(content) else _text_answer(content)


def extract_answer(
    messages: list[dict[str, Any]],
    *,
    max_regions: int | None = None,
    repo_root: str = "",
    allow_single_line: bool | None = None,
) -> Answer:
    """Count search turns and extract the final assistant message.

    The answer is the last <answer> block of the final assistant turn, if that turn makes no
    tool call. Reasoning blocks and narration alongside search calls are never answers.
    """
    max_regions = MAX_ANSWER_REGIONS if max_regions is None else max_regions
    allow_single_line = ACCEPT_SINGLE_LINE if allow_single_line is None else allow_single_line
    ans = Answer()
    assistant = [m for m in messages if m.get("role") == "assistant"]
    for message in assistant:
        if search_calls := search_call_count(message):
            ans.num_turns += 1
            ans.max_parallel_calls = max(ans.max_parallel_calls, search_calls)

    block = _final_block(assistant[-1]) if assistant else None
    if block is None:
        return ans
    ans.submitted = True
    # Every nonempty submitted entry consumes a slot, including malformed
    # or invalid paths. Entries after K cannot replace failed top-K guesses.
    ranked = [line for line in block.splitlines() if line.strip()]
    if max_regions > 0:
        ranked = ranked[:max_regions]
    ans.regions, ans.invalid_regions = parse_regions("\n".join(ranked), allow_single_line=allow_single_line)
    if repo_root:
        ans.regions, invalid = validate_regions(ans.regions, repo_root)
        ans.invalid_regions.extend(invalid)
    ans.files = normalize_list([r.path for r in ans.regions], repo_root)
    return ans


def fbeta(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def score_files(predicted, gold, beta: float, repo_root: str = "") -> dict[str, float]:
    """The graded quantity. Takes raw path lists, normalizes both sides, returns
    precision / recall / f_beta / exact_match plus set sizes. Empty prediction or
    empty gold scores 0."""
    pred = normalize_set(predicted, repo_root)
    gold_set = normalize_set(gold, repo_root)
    hits = len(pred & gold_set)
    precision = hits / len(pred) if pred and gold_set else 0.0
    recall = hits / len(gold_set) if pred and gold_set else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f_beta": fbeta(precision, recall, beta),
        "exact_match": float(bool(gold_set) and pred == gold_set),
        "num_predicted": float(len(pred)),
        "num_gold": float(len(gold_set)),
    }


def recall_at_k(predicted_ordered, gold, k: int, repo_root: str = "") -> float:
    """Share of reachable gold in the first k submitted files (order kept, duplicates dropped).

    The denominator is min(k, |gold|), so a list that puts gold first scores 1.0 even
    when gold is larger than k.
    """
    gold_set = normalize_set(gold, repo_root)
    if not gold_set:
        return 0.0
    pred = normalize_list(predicted_ordered, repo_root)[:k]
    return sum(f in gold_set for f in pred) / min(k, len(gold_set))


def ndcg_at_k(predicted_ordered, gold, k: int, repo_root: str = "") -> float:
    """NDCG@k with binary relevance over the submitted ORDER.

    Gold at rank i earns 1/log2(i+1); the ideal list puts min(k, |gold|) gold
    files first. Junk inside the top k costs by pushing gold down; anything
    past rank k is invisible here.
    """
    pred = normalize_list(predicted_ordered, repo_root)[:k]
    gold_set = normalize_set(gold, repo_root)
    if not gold_set or not pred:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, f in enumerate(pred) if f in gold_set)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold_set))))
    return dcg / ideal
