import json
from pathlib import Path

from email_miles.launch import ROOT, SMOKE, miles_args

RECIPE = json.loads((ROOT / "recipes/qwen3_8b.json").read_text())


def args(recipe=RECIPE):
    return miles_args(recipe, "/model", Path("/run"))


def value(argv, flag):
    return argv[argv.index(flag) + 1]


def test_recipe_becomes_miles_arguments():
    argv = args()
    assert value(argv, "--rollout-batch-size") == "24" and value(argv, "--n-samples-per-prompt") == "16"
    assert value(argv, "--lr") == "5e-05" and value(argv, "--num-rollout") == "20"
    # Qwen3-8B's context is 40,960 tokens; a longer session limit makes SGLang refuse to start.
    assert value(argv, "--max-seq-len") == value(argv, "--seq-length") == "40960"
    assert "--colocate" in argv and "--load" not in argv


def test_sglang_gets_the_parsers_miles_binds_to_the_model_family():
    argv = args()
    assert (value(argv, "--tito-model"), value(argv, "--sglang-tool-call-parser")) == ("qwen3", "qwen25")
    assert (value(args(RECIPE | {"tito_model": "qwen35"}), "--sglang-tool-call-parser")) == "qwen3_coder"


def test_everything_else_is_left_to_miles():
    # Sampling, batch arithmetic, the optimizer and parallelism keep Miles' defaults. The fused
    # log-prob op relies on the default unrestricted sampling and on its in-place backward.
    left_to_miles = ("--rollout-temperature", "--rollout-top-p", "--rollout-top-k", "--eval-temperature",
                     "--global-batch-size", "--optimizer", "--tensor-model-parallel-size", "--sglang-context-length",
                     "--recompute-loss-function")  # fmt: skip
    assert not set(left_to_miles) & set(args())


def test_smoke_run_uses_the_smoke_data():
    argv = args(RECIPE | SMOKE)
    assert value(argv, "--num-rollout") == "1" and value(argv, "--prompt-data").endswith("data/smoke_train.jsonl")
