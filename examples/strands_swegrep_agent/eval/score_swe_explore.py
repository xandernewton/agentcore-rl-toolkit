"""Line scoring of saved SWE-Explore region answers against the published gold; no model calls.

  python eval/score_swe_explore.py --evaluation OUT/swe-explore --out OUT/line-scores

Gold is SWE-Explore's read_core_regions (bench.final.public.jsonl at a pinned revision), with full
denominators, inclusive lines and overlaps deduplicated. Precision, recall and F1 are line-set
arithmetic, which equals the official ExploreEvaluator's P/R/F1 (checked on all 848 tasks). Unsubmitted
answers count as zeros; the macro mean is over all 848 tasks. `first5_submitted_regions` re-scores only
the first five cited regions (a post-hoc diagnostic).
"""

import argparse
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import full_benchmark as harness
from huggingface_hub import hf_hub_download
from rollout import repo_store

GOLD = (
    "SWE-Explore-Bench/SWE-Explore-Bench",
    "bench.final.public.jsonl",
    "bdb0ae45d7c337d9e1dc3ebfe2a0af6bc7c1fbd9",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert (args.evaluation / "COMPLETED").exists(), "Evaluation must finish first"
    source = args.evaluation / "results.jsonl"
    results = {}
    for line in source.read_text().splitlines():
        row = json.loads(line)
        results[(row["rep"], row["instance_id"])] = row
    gold_path = Path(hf_hub_download(GOLD[0], GOLD[1], repo_type="dataset", revision=GOLD[2]))
    gold = {r["instance_id"]: r for r in map(json.loads, gold_path.read_text().splitlines())}
    assert len(results) == len(gold) == 848
    assert {iid for rep, iid in results} == set(gold)
    assert all(rep == 0 and not r["error"] for (rep, _), r in results.items())
    single_line = harness.accepts_single_line(json.loads((args.evaluation / "manifest.json").read_text())["agent_env"])
    scored = []
    for (_, iid), record in sorted(results.items()):
        gt = gold[iid]["ground_truth"]
        gold_lines = {(g["path"], n) for g in gt["read_core_regions"] for n in range(g["start"], g["end"] + 1)}
        assert gold_lines, iid
        with gzip.open(args.evaluation / f"{record['session_id']}.json.gz", "rt") as f:
            raw = json.load(f)
        payload = raw["payload"]
        assert payload["instance_id"] == iid and set(payload["gold_files"]) == set(gt["read_core_files"])
        root = repo_store.ensure_checkout(payload["repo"], payload["base_commit"])
        messages = [t["message"] for t in raw["turns"]]
        modes = {}
        for mode, cap in [("uncapped", 0), ("first5_submitted_regions", 5)]:
            answer, files = harness.parse_messages(
                messages,
                str(root),
                max_regions=cap,
                allow_single_line=single_line,
            )
            regions = [asdict(r) for r in answer.regions]
            if cap == 0:
                assert regions == record["predicted_regions"] and files == record["predicted"], f"Parser drift: {iid}"
            predicted = {(r["path"], n) for r in regions for n in range(r["start_line"], r["end_line"] + 1)}
            hits = len(predicted & gold_lines)
            modes[mode] = {
                "regions": regions,
                "scores": {
                    "precision": hits / len(predicted) if predicted else 0.0,
                    "recall": hits / len(gold_lines),
                    "f1_score": 2 * hits / (len(predicted) + len(gold_lines)),
                    "num_regions": len(regions),
                    "num_files": len(set(files)),
                    "predicted_lines": len(predicted),
                    "gold_lines": len(gold_lines),
                    "overlapping_lines": hits,
                    "invalid_entries": len(answer.invalid_regions),
                    "submitted": float(answer.submitted),
                },
            }
        scored.append({"instance_id": iid, "data_source": record["data_source"], "modes": modes})
    groups = defaultdict(list)
    for row in scored:
        groups["all"].append(row)
        groups[row["data_source"]].append(row)
    summary = {
        mode: {
            name: {
                "n": len(rows),
                **{
                    k: statistics.mean(r["modes"][mode]["scores"][k] for r in rows)
                    for k in rows[0]["modes"][mode]["scores"]
                },
            }
            for name, rows in groups.items()
        }
        for mode in ["uncapped", "first5_submitted_regions"]
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scored))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "tasks": len(scored),
        "results_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "ground_truth": f"{GOLD[0]}@{GOLD[2]}/{GOLD[1]}",
        "ground_truth_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "COMPLETED").write_text(manifest["created_at"] + "\n")
    s = summary["uncapped"]["all"]
    print(f"SWE-Explore line F1 {100 * s['f1_score']:.2f} " f"(P {100 * s['precision']:.1f} R {100 * s['recall']:.1f})")


if __name__ == "__main__":
    main()
