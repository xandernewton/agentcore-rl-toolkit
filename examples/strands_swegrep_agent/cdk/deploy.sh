#!/usr/bin/env bash
# Deploy the AgentCore Runtime via CDK, then write the runtime ARN, region and the result and
# corpus bucket names into ../.env for the trainer, the benchmarks and scripts/prepare_corpus.py.
#
# Prereqs: Docker running, AWS creds, and `cdk bootstrap` already run for this
# account/region. Pass extra cdk args after the script name, e.g.:
#   bash deploy.sh -c corpusBucket=my-existing-corpus-bucket
set -euo pipefail
cd "$(dirname "$0")"

STACK="SweGrepAgentCoreStack"
OUT="cdk-outputs.json"
TRAIN_ENV="../.env"

[ -f config.yaml ] || { echo "FATAL: copy config.example.yaml to config.yaml and fill it in" >&2; exit 1; }
CDK=$(command -v cdk || echo "npx --yes aws-cdk@2.1147.0")

# Build the ARM64 image asset, deploy the stack, dump outputs to JSON.
$CDK deploy "$STACK" --require-approval never --outputs-file "$OUT" "$@"

output() { jq -er ".$STACK.$1" "$OUT"; }
ARN=$(output AgentRuntimeArn)
BUCKET=$(output ResultBucketName)
CORPUS=$(output CorpusBucketName)

# Upsert the values into ../.env (created from ../.env.example on the first deploy).
[ -f "$TRAIN_ENV" ] || cp ../.env.example "$TRAIN_ENV"
set_env() {
  { grep -v "^$1=" "$TRAIN_ENV" || true; echo "$1=$2"; } >"$TRAIN_ENV.tmp" && mv "$TRAIN_ENV.tmp" "$TRAIN_ENV"
}
set_env AGENTCORE_AGENT_ARN "$ARN"
set_env AGENTCORE_S3_BUCKET "$BUCKET"
set_env SWEGREP_CORPUS_BUCKET "$CORPUS"
set_env AWS_REGION "$(cut -d: -f4 <<<"$ARN")"

echo "Wrote to $TRAIN_ENV:"
echo "  AGENTCORE_AGENT_ARN=$ARN"
echo "  AGENTCORE_S3_BUCKET=$BUCKET"
echo "  SWEGREP_CORPUS_BUCKET=$CORPUS"
echo
echo "Next: populate the corpus bucket ->  uv run python scripts/prepare_corpus.py --workers 16"
