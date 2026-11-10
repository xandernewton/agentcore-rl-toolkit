"""Merge an exported PEFT LoRA adapter into the base Qwen3.5 snapshot, keeping its exact layout.

  .venv/bin/python scripts/merge_adapter.py exports/<adapter> <out_dir> [--base <snapshot>]

Works on the safetensors shards directly: W += (lora_alpha / r) * B @ A for every adapted module,
every other tensor and file copied unchanged. The output therefore has the base model's key names,
shard layout and processor files (the base snapshot serves in vLLM as-is), avoiding the
module-tree pitfalls of a transformers round trip. Needs torch + safetensors (the `merge` extra).
The merged weights are BF16, so small LoRA deltas round away (about half of them for the published
adapter); the published benchmark numbers were measured on this merged model.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

BASE = Path(
    os.environ.get("MODEL_DIR")
    or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    / "hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
PREFIX = "base_model.model."


def base_key(key, weight_map):
    """Map a text-only Transformers adapter back onto Qwen's full HF snapshot."""
    if key in weight_map:
        return key
    if key.startswith("model.layers."):
        candidate = key.replace("model.", "model.language_model.", 1)
        if candidate in weight_map:
            return candidate
    raise ValueError(f"Adapted module not in the base checkpoint: {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify every key and tensor shape without writing weights",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")

    config = json.loads((args.adapter / "adapter_config.json").read_text())
    if any(
        config.get(k)
        for k in [
            "use_dora",
            "use_rslora",
            "fan_in_fan_out",
            "rank_pattern",
            "alpha_pattern",
            "modules_to_save",
        ]
    ):
        raise SystemExit("This merger supports plain uniform-rank LoRA only")
    scale = config["lora_alpha"] / config["r"]
    lora = {}
    with safe_open(str(args.adapter / "adapter_model.safetensors"), "pt") as f:
        for key in f.keys():  # noqa: SIM118 - safe_open is not iterable
            if not key.startswith(PREFIX) or not key.endswith((".lora_A.weight", ".lora_B.weight")):
                raise SystemExit(f"unexpected adapter key {key}")
            module, part = key[len(PREFIX) :].rsplit(".lora_", 1)
            lora.setdefault(f"{module}.weight", {})[part[0]] = f.get_tensor(key)

    index = json.loads((args.base / "model.safetensors.index.json").read_text())
    mapped = {}
    for key, parts in lora.items():
        target = base_key(key, index["weight_map"])
        if target in mapped or set(parts) != {"A", "B"}:
            raise SystemExit(f"Duplicate target or incomplete LoRA pair: {target}")
        mapped[target] = parts
    if args.dry_run:
        for shard in {index["weight_map"][key] for key in mapped}:
            with safe_open(str(args.base / shard), "pt") as f:
                for key, parts in mapped.items():
                    if index["weight_map"][key] != shard:
                        continue
                    a, b = parts["A"], parts["B"]
                    if (
                        a.ndim != 2
                        or b.ndim != 2
                        or a.shape[0] != b.shape[1]
                        or list(f.get_slice(key).get_shape()) != [b.shape[0], a.shape[1]]
                    ):
                        raise SystemExit(f"Incompatible adapter shape: {key}")
        print(
            json.dumps(
                {
                    "verified_modules": len(mapped),
                    "scale": scale,
                    "remapped_text_modules": sum(base_key(k, index["weight_map"]) != k for k in lora),
                    "base": str(args.base),
                    "adapter": str(args.adapter),
                    "weights_written": False,
                }
            )
        )
        return
    lora = mapped

    args.out.mkdir(parents=True)
    merged = 0
    deltas = {}
    for shard in sorted(set(index["weight_map"].values())):
        tensors = {}
        with safe_open(str(args.base / shard), "pt") as f:
            for key in f.keys():  # noqa: SIM118 - safe_open is not iterable
                weight = f.get_tensor(key)
                if key in lora:
                    a, b = lora[key]["A"].float(), lora[key]["B"].float()
                    delta = scale * (b @ a)
                    if delta.shape != weight.shape:
                        raise SystemExit(f"{key}: delta {tuple(delta.shape)} vs weight {tuple(weight.shape)}")
                    new = (weight.float() + delta).to(weight.dtype)
                    deltas[key] = (new.float() - weight.float()).abs().max().item()
                    weight = new
                    merged += 1
                tensors[key] = weight.contiguous()
        save_file(tensors, str(args.out / shard), metadata={"format": "pt"})
        print(f"wrote {shard} ({len(tensors)} tensors)")
    if merged != len(lora):
        raise SystemExit(f"merged {merged} of {len(lora)} adapted modules")
    if not all(d > 0 for d in deltas.values()):
        raise SystemExit(f"{sum(d == 0 for d in deltas.values())} modules unchanged after merge (bf16 rounding?)")

    for item in args.base.iterdir():
        if not item.name.endswith(".safetensors"):
            shutil.copy(item, args.out / item.name)  # config, tokenizer, chat template, processors, index
    (args.out / "MERGED_FROM.json").write_text(
        json.dumps(
            {
                "adapter": str(args.adapter.resolve()),
                "base": str(args.base),
                "scale": scale,
                "modules": merged,
                "adapter_sha256": hashlib.sha256((args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
                "max_abs_delta": max(deltas.values()),
                "source": (args.adapter / "SOURCE.txt").read_text().strip()
                if (args.adapter / "SOURCE.txt").exists()
                else None,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"merged {merged} modules (scale {scale}); max |delta| {max(deltas.values()):.3g}; wrote {args.out}")


if __name__ == "__main__":
    main()
