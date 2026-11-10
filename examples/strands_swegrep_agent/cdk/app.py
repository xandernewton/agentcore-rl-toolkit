#!/usr/bin/env python3
import os
from pathlib import Path

import aws_cdk as cdk
import yaml
from swegrep_agent_stack import SweGrepAgentCoreStack

app = cdk.App()

# Deploy config (account, VPC, runtime name, buckets) lives in config.yaml. Fold it
# into CDK context so the stack's self.node.try_get_context(...) reads it with no
# stack changes. A CLI override (`-c key=value`) always wins: we only set a key
# when it wasn't already provided on the command line. `-c config=<file>` picks
# another config file.
cfg: dict = {}
config_path = Path(__file__).parent / (app.node.try_get_context("config") or "config.yaml")
if config_path.exists():
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

for key, value in cfg.items():
    if key in ("awsAccount", "awsRegion"):
        continue  # consumed below for the deploy Environment, not stack context
    if app.node.try_get_context(key) is None:
        app.node.set_context(key, value)

# from_lookup (VPC) needs a concrete account + region. Prefer config.yaml, then
# the CDK_DEFAULT_* env vars the AWS CLI sets from the active credentials.
account = cfg.get("awsAccount") or os.environ.get("CDK_DEFAULT_ACCOUNT")
region = cfg.get("awsRegion") or os.environ.get("CDK_DEFAULT_REGION")

SweGrepAgentCoreStack(
    app,
    "SweGrepAgentCoreStack",
    env=cdk.Environment(account=account, region=region),
    description=(
        "SWE-grep code-region retrieval agent on AWS Bedrock AgentCore Runtime "
        "for RL training (agentcore-rl-toolkit)."
    ),
)

app.synth()
