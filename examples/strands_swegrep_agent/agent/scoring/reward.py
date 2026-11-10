"""Training reward for the SWE-grep retrieval agent.

Entry point: ``compute_reward(messages, gold_files, rollout_exception, ...) -> (float, dict)``.
The reward is region F-beta over unique cited lines, less a capped penalty for invalid regions.
The file-level and search numbers in the metrics are diagnostics, never reward terms.
"""

import json
import math
import os
from typing import Any

from scoring.retrieval import (
    extract_answer,
    fbeta,
    ndcg_at_k,
    normalize_path,
    normalize_set,
    recall_at_k,
    score_files,
)
from scoring.search_metrics import (
    calls_by_turn,
    idle_turns,
    search_quality,
    surfaced_recall,
    tool_result_blob,
    zero_hit_calls,
)

# Recall weighting of the F-beta. The trained model used 2; the ablation recipes vary it.
BETA = float(os.environ.get("SWEGREP_BETA", "2"))
if not math.isfinite(BETA) or BETA <= 0:
    raise ValueError("SWEGREP_BETA must be finite and positive")
# Score for no or empty answer; no submission may score below it or the policy stalls.
UNSUBMITTED_PENALTY = 0.3
# Cost of each distinct invalid region, and the most they can cost in total (at most the above).
INVALID_REGION_PENALTY = 0.05
INVALID_REGION_PENALTY_CAP = 0.3
# Rank cut-offs for the ndcg and recall_at_top metrics.
NDCG_K = 5
RECALL_TOP_K = 3


def merge_spans(entries, repo_root=""):
    """(path, start, end) entries -> {path: sorted, merged spans}, rejecting bad intervals."""
    grouped = {}
    for path, start, end in entries:
        path = normalize_path(path, repo_root)
        # type() rather than isinstance(): bool is an int subclass and must be rejected.
        if not path or type(start) is not int or type(end) is not int or start < 1 or end < start:  # noqa: E721
            raise ValueError("Invalid region reference interval")
        grouped.setdefault(path, []).append((start, end))
    result = {}
    for path, spans in grouped.items():
        merged = []
        for start, end in sorted(spans):
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        result[path] = merged
    return result


def line_count(paths):
    """Lines covered by {path: merged spans}."""
    return sum(end - start + 1 for spans in paths.values() for start, end in spans)


def _overlap(p, g):
    """Lines two {path: merged spans} share, walking both sorted span lists together."""
    hits = 0
    for path, predicted in p.items():
        reference = g.get(path, [])
        i = j = 0
        while i < len(predicted) and j < len(reference):
            ps, pe = predicted[i]
            gs, ge = reference[j]
            hits += max(0, min(pe, ge) - max(ps, gs) + 1)
            if pe <= ge:
                i += 1
            else:
                j += 1
    return hits


def score_regions(regions, gold_lines, repo_root=""):
    """Unique-line overlap with interval unions; never expand large spans into sets."""
    gold = json.loads(gold_lines) if isinstance(gold_lines, str) else gold_lines
    if not isinstance(gold, dict) or not gold:
        raise ValueError("region_f1 requires nonempty gold_lines reference labels")
    g = merge_spans(((path, start, end) for path, spans in gold.items() for start, end in spans), repo_root)
    p = merge_spans(((r.path, r.start_line, r.end_line) for r in regions), repo_root)
    pn, gn = line_count(p), line_count(g)
    if not gn:
        raise ValueError("region_f1 requires nonempty gold_lines reference labels")
    hits = _overlap(p, g)
    return {
        "region_precision": hits / pn if pn else 0.0,
        "region_recall": hits / gn,
        "region_f1": 2 * hits / (pn + gn),
        "num_predicted_lines": float(pn),
        "num_gold_lines": float(gn),
    }


def compute_reward(
    messages: list[dict[str, Any]],
    gold_files: list[str],
    rollout_exception: bool,
    *,
    gold_lines: dict | str | None,
    repo_root: str = "",
    infra_exception: bool = False,
) -> tuple[float, dict[str, float]]:
    """Score a rollout against its region labels.

    infra_exception scores 0.0, not -1.0 (a -1 shrinks every GRPO sibling advantage);
    an exception after a valid submission is still scored on that answer.
    """
    ans = extract_answer(messages, repo_root=repo_root)
    gold = normalize_set(gold_files, repo_root)
    scores = score_files(ans.files, gold_files, BETA, repo_root)
    region_scores = score_regions(ans.regions, gold_lines, repo_root)
    region_scores["region_fbeta"] = fbeta(region_scores["region_precision"], region_scores["region_recall"], BETA)

    if infra_exception:
        reward = 0.0
    elif not (ans.submitted and scores["num_predicted"] > 0):
        reward = -1.0 if rollout_exception else -UNSUBMITTED_PENALTY
    else:
        distinct_invalid = len({entry.strip() for entry in ans.invalid_regions})
        penalty = min(INVALID_REGION_PENALTY_CAP, INVALID_REGION_PENALTY * distinct_invalid)
        reward = region_scores["region_fbeta"] - penalty

    metrics = {
        "reward": float(reward),
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f_beta": scores["f_beta"],
        "ndcg": ndcg_at_k(ans.files, gold_files, NDCG_K, repo_root),
        "recall_at_top": recall_at_k(ans.files, gold_files, RECALL_TOP_K, repo_root),
        "exact_match": scores["exact_match"],
        "num_predicted": scores["num_predicted"],
        "num_gold": scores["num_gold"],
        "complete_hit": float(scores["num_gold"] > 0 and scores["recall"] == 1.0),
        "submitted": float(ans.submitted),
        "empty_submission": float(ans.submitted and scores["num_predicted"] == 0),
        "num_turns": float(ans.num_turns),
        "max_parallel_calls": float(ans.max_parallel_calls),
        # Upper bound on what the search reached.
        "surfaced_recall": surfaced_recall(tool_result_blob(messages), gold),
        "zero_hit_calls": float(zero_hit_calls(messages)),
        "idle_turns": float(idle_turns(calls_by_turn(messages))),
        **search_quality(messages, gold),
        "rollout_exception": float(rollout_exception),
        "infra_exception": float(infra_exception),
        **region_scores,
        "invalid_regions": float(len(ans.invalid_regions)),
    }
    return float(reward), metrics
