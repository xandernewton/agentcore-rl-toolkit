"""Score final region citations against the published ContextBench Lite gold; no model calls.

  python eval/score_contextbench.py OUT        (reads OUT/contextbench, writes OUT/scores.{json,jsonl})

Gold is gold_context of contextbench_verified.parquet at a pinned revision, paths made repo-relative.
This scores the final answer's lines, not ContextBench's trajectory-context metric.
"""

import gzip
import importlib.util
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

import full_benchmark as harness
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from rollout import repo_store

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_benchmarks", ROOT / "data/build_benchmarks.py")
build_benchmarks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_benchmarks)

out = Path(sys.argv[1])
ev = out / "contextbench"
assert (ev / "COMPLETED").exists()
rows = {r["instance_id"]: r for r in map(json.loads, (ev / "results.jsonl").read_text().splitlines())}
payloads = {
    r["instance_id"]: r
    for r in map(json.loads, (ROOT / "data/benchmarks/contextbench-lite.jsonl").read_text().splitlines())
}
assert set(rows) == set(payloads) and len(rows) == 500 and all(not r["error"] for r in rows.values())
single_line = harness.accepts_single_line(json.loads((ev / "manifest.json").read_text())["agent_env"])
repo_id, revision, filename = build_benchmarks.CONTEXTBENCH
table = pq.read_table(hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision))
gold = {r["instance_id"]: json.loads(r["gold_context"]) for r in table.to_pylist()}

scored = []
for iid, r in rows.items():
    pay = payloads[iid]
    lines = {
        (build_benchmarks.canonical(g["file"], pay["repo"]), n)
        for g in gold[iid]
        for n in range(g["start_line"], g["end_line"] + 1)
    }
    assert lines and {p for p, n in lines} == set(pay["gold_files"]), iid
    with gzip.open(ev / f"{r['session_id']}.json.gz", "rt") as f:
        raw = json.load(f)
    root = str(repo_store.ensure_checkout(pay["repo"], pay["base_commit"]))
    modes = {}
    for mode, cap in [("uncapped", 0), ("first5_regions", 5)]:
        ans, files = harness.parse_messages(
            [t["message"] for t in raw["turns"]],
            root,
            max_regions=cap,
            allow_single_line=single_line,
        )
        regions = [asdict(v) for v in ans.regions]
        if cap == 0:
            assert regions == r["predicted_regions"] and files == r["predicted"], iid
        pred = {(v["path"], n) for v in regions for n in range(v["start_line"], v["end_line"] + 1)}
        hits = len(lines & pred)
        fs = harness.score(files, pay["gold_files"])
        modes[mode] = {
            "line_precision": hits / len(pred) if pred else 0.0,
            "line_recall": hits / len(lines),
            "line_f1": 2 * hits / (len(pred) + len(lines)),
            "file_precision": fs["precision"],
            "file_recall": fs["recall"],
            "file_f1": fs["f1"],
            "predicted_lines": len(pred),
            "gold_lines": len(lines),
            "num_files": len(files),
            "num_regions": len(regions),
            "submitted": float(ans.submitted),
            "invalid_entries": len(ans.invalid_regions),
        }
    scored.append({"instance_id": iid, "lite": True, "source": r["data_source"], "modes": modes})
summary = {
    "lite": {
        mode: {
            "n": len(scored),
            **{k: statistics.mean(r["modes"][mode][k] for r in scored) for k in scored[0]["modes"][mode]},
        }
        for mode in ["uncapped", "first5_regions"]
    }
}
(out / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scored))
(out / "scores.json").write_text(json.dumps(summary, indent=2) + "\n")
(out / "SCORING_COMPLETED").write_text("completed\n")
s = summary["lite"]["uncapped"]
print(
    f"ContextBench Lite line F1 {100 * s['line_f1']:.2f} "
    f"(P {100 * s['line_precision']:.1f} R {100 * s['line_recall']:.1f})"
)
