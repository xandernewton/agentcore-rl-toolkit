"""Export a SkyRL Tinker training checkpoint as a Hugging Face PEFT LoRA adapter.

  .venv/bin/python scripts/export_adapter.py tinker://<model>/weights/<name> <out_dir>

SkyRL's training checkpoints (save_state -> backend save_checkpoint) are Megatron distributed shards.
A *named* sampler save (save_weights_for_sampler(name) -> backend save_sampler_checkpoint with
persist=True) tars the live HF PEFT adapter instead (adapter_model.safetensors + adapter_config.json),
at $SKYRL_STATE_DIR/checkpoints/<model_id>/sampler_weights/<name>.tar.gz. So: load the training
checkpoint into a fresh LoRA client, save a named sampler checkpoint, extract that archive, and check
it holds non-zero LoRA B weights (a fresh adapter has B = 0).
"""

import argparse
import json
import os
import tarfile
from pathlib import Path

import tinker
from dotenv import load_dotenv


def inspect_lora_b(path: Path) -> tuple[list[str], int]:
    """Tensor names and the number of lora_B tensors with any non-zero element (torch-free)."""
    with path.open("rb") as stream:
        header_len = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(header_len))
        header.pop("__metadata__", None)
        nonzero = 0
        for name, info in header.items():
            if "lora_B" not in name:
                continue
            start, end = info["data_offsets"]
            stream.seek(8 + header_len + start)
            data = stream.read(end - start)
            # Any set bit outside the sign bits means a non-zero float (bf16/fp16/fp32 alike).
            if data.strip(b"\x00\x80"):
                nonzero += 1
    return list(header), nonzero


def main():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # TINKER_PORT, as in swegrep_train.py
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("TINKER_ENDPOINT") or f"http://127.0.0.1:{os.environ.get('TINKER_PORT', '18080')}",
    )
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.environ.get("SKYRL_STATE_DIR", "~/skyrl-state")),
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")

    service = tinker.ServiceClient(base_url=args.endpoint, api_key=os.environ.get("TINKER_API_KEY") or "tml-dummy")
    base_model = service.get_server_capabilities().supported_models[0].model_name
    trainer = service.create_lora_training_client(base_model=base_model, rank=args.lora_rank, seed=42)
    trainer.load_state(args.checkpoint).result()  # must be the new model's first request
    model_id = trainer._guaranteed_model_id()
    print(f"loaded {args.checkpoint} into {model_id}")
    name = f"export-{args.out.name}"
    saved = trainer.save_weights_for_sampler(name).result()
    print(f"saved sampler checkpoint {saved.path}")

    archive = args.state_dir.expanduser() / "checkpoints" / model_id / "sampler_weights" / f"{name}.tar.gz"
    if not archive.exists():
        raise SystemExit(f"no archive at {archive}")
    args.out.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(args.out, filter="data")
    if not (args.out / "adapter_model.safetensors").exists():
        raise SystemExit(f"{archive} holds {sorted(p.name for p in args.out.iterdir())}, not a PEFT adapter")

    config = json.loads((args.out / "adapter_config.json").read_text())
    keys, b_nonzero = inspect_lora_b(args.out / "adapter_model.safetensors")
    print(
        json.dumps(
            {
                k: config.get(k)
                for k in (
                    "r",
                    "lora_alpha",
                    "target_modules",
                    "base_model_name_or_path",
                )
            }
        )
    )
    print(f"{len(keys)} tensors, e.g. {keys[0]}; lora_B tensors with non-zero values: {b_nonzero}")
    if not b_nonzero:
        raise SystemExit("adapter has zero lora_B weights: the checkpoint did not load")
    (args.out / "SOURCE.txt").write_text(f"{args.checkpoint}\n")
    print(f"exported to {args.out}")


if __name__ == "__main__":
    main()
