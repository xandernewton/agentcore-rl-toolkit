"""CDK stack: the SWE-grep agent on AWS Bedrock AgentCore Runtime (VPC mode).

Builds the ARM64 image from ../agent, creates the Runtime in the gateway host's VPC
(the gateway base_url is a private IP, so the runtime ENIs must share that VPC),
grants the execution role S3 result-bucket write + corpus-bucket read, and emits
the runtime ARN + bucket names as outputs that deploy.sh writes to ../.env.

Context overrides (cdk deploy -c key=value; config.yaml supplies defaults):
  vpcId                     VPC of the GPU training/gateway host (default: the account's default VPC)
  subnetType                public | private-egress | private-isolated (default: private-egress)
  gatewaySecurityGroupId    SG on the gateway host; inbound gateway and benchmark vLLM
                            (:8399) rules from the runtime SG are added automatically when set
  gatewayPort               gateway port to open (default: 9090)
  createVpcEndpoints        true to create the ecr/logs/s3 VPC endpoints; false to
                            reuse ones already in the VPC (default: true)
  endpointSecurityGroupId   when reusing endpoints, the shared interface-endpoint SG
                            to open on :443 from this runtime's SG
  resultBucket              existing S3 bucket for rollout results (default: create one)
  corpusBucket              existing S3 bucket holding repo snapshots (default: create one)
  runtimeName               AgentCore runtime name (default: swegrep_agent_rl)
  imageRepository           existing same-region ECR repo to run instead of building
                            ../agent (with imageTag; default: build the asset)
  imageTag                  tag in imageRepository
  recipe                    recipe whose swegrep.runtime_env (reward and agent settings) the runtime gets,
                            relative to the repository root (default: recipes/final.json)
"""

import json
import os

from aws_cdk import (
    CfnOutput,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_bedrockagentcore as agentcore
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_s3 as s3
from constructs import Construct

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
AGENT_DIR = os.path.join(REPO_ROOT, "agent")

_SUBNET_TYPES = {
    "public": ec2.SubnetType.PUBLIC,
    "private-egress": ec2.SubnetType.PRIVATE_WITH_EGRESS,
    "private-isolated": ec2.SubnetType.PRIVATE_ISOLATED,
}


def _as_bool(value, default: bool) -> bool:
    """Coerce a CDK context value to bool. config.yaml yields a real bool; a CLI
    ``-c key=value`` override yields a string."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("false", "0", "no")


def _recipe_runtime_env(recipe: str) -> dict[str, str]:
    """A recipe's ``swegrep.runtime_env``: the agent settings (reward beta, answer parsing, ...)
    the run trains with. swegrep_train.py refuses to start if the runtime's differ."""
    with open(os.path.join(REPO_ROOT, recipe)) as f:
        env = json.load(f)["swegrep"]["runtime_env"]
    if not all(key.startswith("SWEGREP_") for key in env):
        raise ValueError(f"{recipe}: runtime_env may only set SWEGREP_* variables")
    return {key: str(value) for key, value in env.items()}


# Benchmark vLLM port the agent calls directly; fixed in endpoint/serve_eval.sh and eval/.
EVAL_ENGINE_PORT = 8399


class SweGrepAgentCoreStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ctx = self.node.try_get_context
        runtime_name = ctx("runtimeName") or "swegrep_agent_rl"
        # Context only, never the environment: deploy.sh writes the stack's own bucket names to .env,
        # and reading them back from a sourced .env would turn those buckets into imports.
        existing_bucket = ctx("resultBucket")
        existing_corpus = ctx("corpusBucket")
        vpc_id = ctx("vpcId")
        subnet_type = _SUBNET_TYPES[ctx("subnetType") or "private-egress"]
        gateway_sg_id = ctx("gatewaySecurityGroupId")
        gateway_port = int(ctx("gatewayPort") or 9090)
        # A VPC holds one S3 gateway endpoint per route table / one private-DNS endpoint per
        # service: set false and reuse when another stack already owns them.
        create_endpoints = _as_bool(ctx("createVpcEndpoints"), True)
        # When reusing endpoints, their interface-endpoint SG must admit this runtime's SG on 443.
        shared_endpoint_sg_id = ctx("endpointSecurityGroupId")
        # AgentCore pins the runtime ENIs ~8h after delete, so a stack that DELETES the runtime SG
        # hangs on destroy: reuse a persistent SG (runtimeSecurityGroupId) or RETAIN the created one.
        runtime_sg_id = ctx("runtimeSecurityGroupId")
        retain_runtime_sg = _as_bool(ctx("retainRuntimeSg"), True)
        recipe = ctx("recipe") or "recipes/final.json"

        # S3 bucket the agent writes each rollout result to and the trainer reads back.
        if existing_bucket:
            result_bucket = s3.Bucket.from_bucket_name(self, "ResultBucket", existing_bucket)
        else:
            result_bucket = s3.Bucket(
                self,
                "ResultBucket",
                removal_policy=RemovalPolicy.RETAIN,  # rollout artifacts; don't auto-delete
                enforce_ssl=True,
            )

        # S3 bucket holding the pre-materialized {repo}@{base_commit} snapshots that
        # agent/rollout/repo_store.py fetches at rollout time. Populated by scripts/prepare_corpus.py.
        if existing_corpus:
            corpus_bucket = s3.Bucket.from_bucket_name(self, "CorpusBucket", existing_corpus)
        else:
            corpus_bucket = s3.Bucket(
                self,
                "CorpusBucket",
                removal_policy=RemovalPolicy.RETAIN,  # expensive to rebuild; keep
                enforce_ssl=True,
            )

        if ctx("imageRepository"):
            # Reuse a pushed image (e.g. one copied from another region) so runtimes stay in parity.
            repo = ecr.Repository.from_repository_name(self, "AgentImageRepo", ctx("imageRepository"))
            artifact = agentcore.AgentRuntimeArtifact.from_ecr_repository(repo, ctx("imageTag") or "latest")
        else:
            # AgentCore requires linux/arm64; on x86 this needs QEMU (tonistiigi/binfmt --install arm64).
            artifact = agentcore.AgentRuntimeArtifact.from_asset(
                AGENT_DIR,
                platform=ecr_assets.Platform.LINUX_ARM64,
            )

        # --- VPC networking: put the runtime ENIs in the gateway host's VPC ---
        # from_lookup needs a concrete account/region (set in app.py via CDK_DEFAULT_*).
        if vpc_id:
            vpc = ec2.Vpc.from_lookup(self, "GatewayVpc", vpc_id=vpc_id)
        else:
            vpc = ec2.Vpc.from_lookup(self, "GatewayVpc", is_default=True)

        # Runtime-ENI SG: import a persistent one if given, else create and RETAIN (see above).
        if runtime_sg_id:
            runtime_sg = ec2.SecurityGroup.from_security_group_id(self, "RuntimeSg", runtime_sg_id, mutable=True)
        else:
            runtime_sg = ec2.SecurityGroup(
                self,
                "RuntimeSg",
                vpc=vpc,
                allow_all_outbound=True,
                description=f"AgentCore swegrep runtime ENIs (reach gateway :{gateway_port})",
            )
            if retain_runtime_sg:
                runtime_sg.apply_removal_policy(RemovalPolicy.RETAIN)

        network_configuration = agentcore.RuntimeNetworkConfiguration.using_vpc(
            self,
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=subnet_type),
            security_groups=[runtime_sg],
        )

        # --- VPC endpoints: runtime ENIs have only private IPs, so without these (or a NAT)
        # the container cannot pull ECR, ship logs, or reach S3 (502 with empty logs).
        # ecr.api + ecr.dkr + logs (interface) and s3 (gateway) are all required.
        endpoint_subnets = ec2.SubnetSelection(subnet_type=subnet_type)
        if create_endpoints:
            endpoint_sg = ec2.SecurityGroup(
                self,
                "VpcEndpointSg",
                vpc=vpc,
                allow_all_outbound=True,
                description="HTTPS from the swegrep runtime ENIs to interface VPC endpoints",
            )
            endpoint_sg.add_ingress_rule(
                ec2.Peer.security_group_id(runtime_sg.security_group_id),
                ec2.Port.tcp(443),
                "HTTPS from swegrep AgentCore runtime",
            )
            for endpoint_id, service in [
                ("EcrApiEndpoint", ec2.InterfaceVpcEndpointAwsService.ECR),
                ("EcrDkrEndpoint", ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER),
                ("LogsEndpoint", ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS),
            ]:
                vpc.add_interface_endpoint(
                    endpoint_id,
                    service=service,
                    subnets=endpoint_subnets,
                    security_groups=[endpoint_sg],
                    private_dns_enabled=True,
                )
            # S3 must be a GATEWAY endpoint (route-table wired), not an interface endpoint.
            vpc.add_gateway_endpoint(
                "S3Endpoint",
                service=ec2.GatewayVpcEndpointAwsService.S3,
                subnets=[endpoint_subnets],
            )
        elif shared_endpoint_sg_id:
            # Reuse another stack's endpoints: only their SG needs to admit this runtime on 443.
            shared_endpoint_sg = ec2.SecurityGroup.from_security_group_id(
                self, "SharedEndpointSg", shared_endpoint_sg_id, mutable=True
            )
            shared_endpoint_sg.add_ingress_rule(
                ec2.Peer.security_group_id(runtime_sg.security_group_id),
                ec2.Port.tcp(443),
                "HTTPS from swegrep AgentCore runtime (reusing shared VPC endpoints)",
            )
        # else: existing endpoints already admit this runtime, or a NAT path exists.

        # The recipe's agent settings, then the infrastructure the agent needs.
        env_vars = {
            **_recipe_runtime_env(recipe),
            "AGENTCORE_S3_BUCKET": result_bucket.bucket_name,
            "SWEGREP_CORPUS_BUCKET": corpus_bucket.bucket_name,
            "AWS_REGION": self.region,
            "OTEL_METRICS_EXPORTER": "none",
        }

        runtime = agentcore.Runtime(
            self,
            "SweGrepAgentRuntime",
            runtime_name=runtime_name,
            agent_runtime_artifact=artifact,
            network_configuration=network_configuration,
            environment_variables=env_vars,
            # execution_role omitted -> CDK creates one; augmented with S3 grants below.
        )

        # Open the gateway port to the runtime SG (else add the rule yourself from the output).
        if gateway_sg_id:
            gateway_sg = ec2.SecurityGroup.from_security_group_id(self, "GatewaySg", gateway_sg_id, mutable=True)
            gateway_sg.add_ingress_rule(
                ec2.Peer.security_group_id(runtime_sg.security_group_id),
                ec2.Port.tcp(gateway_port),
                "rollout gateway from AgentCore swegrep runtime",
            )
            # Benchmarks: the agent calls vLLM's OpenAI endpoint on the same host directly.
            gateway_sg.add_ingress_rule(
                ec2.Peer.security_group_id(runtime_sg.security_group_id),
                ec2.Port.tcp(EVAL_ENGINE_PORT),
                "benchmark vLLM from AgentCore swegrep runtime",
            )

        # The auto-created execution role lacks these; add them.
        result_bucket.grant_read_write(runtime.role)
        corpus_bucket.grant_read(runtime.role)
        # deploy.sh reads these by output key; keep the names stable.
        CfnOutput(
            self,
            "AgentRuntimeArn",
            value=runtime.agent_runtime_arn,
            description="AGENTCORE_AGENT_ARN for .env",
        )
        CfnOutput(
            self,
            "ResultBucketName",
            value=result_bucket.bucket_name,
            description="AGENTCORE_S3_BUCKET for .env",
        )
        CfnOutput(
            self,
            "CorpusBucketName",
            value=corpus_bucket.bucket_name,
            description="SWEGREP_CORPUS_BUCKET for .env (populate with scripts/prepare_corpus.py)",
        )
        CfnOutput(
            self,
            "RuntimeSecurityGroupId",
            value=runtime_sg.security_group_id,
            description=f"Allow inbound :{gateway_port} and :{EVAL_ENGINE_PORT} from this SG on the gateway host",
        )
