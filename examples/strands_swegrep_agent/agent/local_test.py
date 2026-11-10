"""Local test run for the explorer: dummy repos, Bedrock as the LLM, real tools + reward.

Needs Bedrock creds:

    cd agent && ../.venv/bin/python local_test.py
    LOCAL_TEST_MODEL=jp.anthropic.claude-haiku-4-5-20251001-v1:0 \
        ../.venv/bin/python local_test.py
"""

import os
import tempfile
from pathlib import Path

from rollout.agent_prompts import build_system_prompt
from rollout.repo_tools import build_repo_tools
from rollout.turn_cap import TurnCapHook
from scoring.retrieval import extract_answer
from scoring.reward import compute_reward
from strands import Agent
from strands.models import BedrockModel

MODEL_ID = os.environ.get("LOCAL_TEST_MODEL", "global.anthropic.claude-sonnet-5-5")
REGION = os.environ.get("AWS_REGION", "us-east-1")


def _make_repo(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def run_scenario(name: str, files: dict[str, str], problem: str, gold_files: list[str]) -> bool:
    tmp = Path(tempfile.mkdtemp(prefix=f"swegrep_{name}_"))
    _make_repo(tmp, files)

    model = BedrockModel(model_id=MODEL_ID, region_name=REGION, temperature=0.7, max_tokens=2048)
    agent = Agent(
        model=model,
        tools=build_repo_tools(tmp),
        system_prompt=build_system_prompt(f"dummy/{name}"),
        hooks=[TurnCapHook()],
        callback_handler=None,
    )

    rollout_exception = False
    try:
        agent(problem)
    except Exception as exc:  # noqa: BLE001
        rollout_exception = True
        print(f"  ROLLOUT EXCEPTION: {exc}")

    tool_calls = [
        b["toolUse"].get("name")
        for m in agent.messages
        if m.get("role") == "assistant" and isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and "toolUse" in b
    ]
    answer = extract_answer(list(agent.messages))
    reward, metrics = compute_reward(
        list(agent.messages),
        gold_files,
        rollout_exception,
        repo_root=str(tmp),
        gold_lines={f: [[1, len(files[f].splitlines())]] for f in gold_files},
    )

    print(f"\n===== {name} =====")
    print(f"  repo:           {tmp}")
    print(f"  tool calls:     {tool_calls}")
    print(f"  predicted:      {answer.files}")
    print(f"  gold:           {gold_files}")
    print(
        f"  reward={reward:.3f}  f_beta={metrics['region_fbeta']:.2f}  "
        f"precision={metrics['region_precision']:.2f} recall={metrics['region_recall']:.2f}  "
        f"exact={metrics['exact_match']:.0f}  submitted={metrics['submitted']:.0f}  "
        f"turns={metrics['num_turns']:.0f}  surfaced={metrics['surfaced_recall']:.2f}"
    )
    return metrics["submitted"] == 1.0 and not rollout_exception


SCENARIOS = [
    {
        "name": "router-bug",
        "problem": (
            "Title: Router.match returns the first route even when the path doesn't match\n\n"
            "When I register two routes and request a path that only matches the second, "
            "`Router.match` still returns the first route. It looks like the matching loop "
            "never compares the request path against each route's pattern. Steps: register "
            "'/users' and '/posts', then call match('/posts') — it returns the '/users' handler."
        ),
        "gold": ["src/router.py"],
        "files": {
            "README.md": "# Demo web framework\nA tiny router + config loader.\n",
            "src/router.py": (
                "class Route:\n    def __init__(self, pattern, handler):\n"
                "        self.pattern = pattern\n        self.handler = handler\n\n"
                "class Router:\n    def __init__(self):\n        self.routes = []\n\n"
                "    def add(self, pattern, handler):\n        self.routes.append(Route(pattern, handler))\n\n"
                "    def match(self, path):\n"
                "        # BUG: ignores `path`, always returns the first route\n"
                "        for route in self.routes:\n            return route.handler\n"
                "        return None\n"
            ),
            "src/config.py": "import os\n\ndef load_config():\n    return {'debug': os.environ.get('DEBUG') == '1'}\n",
            "src/server.py": "from .router import Router\n\napp = Router()\n",
            "tests/test_router.py": "from src.router import Router\n\ndef test_match():\n    assert True\n",
        },
    },
    {
        "name": "auth-expiry",
        "problem": (
            "Title: Sessions never expire\n\n"
            "`Session.is_valid()` always returns True, so expired sessions are still accepted. "
            "The expiry timestamp is stored but never checked against the current time."
        ),
        "gold": ["auth/session.py"],
        "files": {
            "auth/login.py": "from .session import Session\n\ndef login(user):\n    return Session(user)\n",
            "auth/session.py": (
                "import time\n\nclass Session:\n    def __init__(self, user, ttl=3600):\n"
                "        self.user = user\n        self.expires_at = time.time() + ttl\n\n"
                "    def is_valid(self):\n"
                "        # BUG: never compares expires_at to now\n        return True\n"
            ),
            "utils/hashing.py": "import hashlib\n\ndef sha(s):\n    return hashlib.sha256(s.encode()).hexdigest()\n",
            "README.md": "# Auth service\n",
        },
    },
]


def main() -> None:
    print(f"model={MODEL_ID} region={REGION}")
    ok = 0
    for scenario in SCENARIOS:
        if run_scenario(scenario["name"], scenario["files"], scenario["problem"], scenario["gold"]):
            ok += 1
    print(f"\n{ok}/{len(SCENARIOS)} scenarios produced a submitted answer without exception.")


if __name__ == "__main__":
    main()
