# Training image: the Miles v0.2.0 release image (Miles at /root/miles, with its patched SGLang and
# Megatron-LM) plus the AgentCore RL toolkit, which the rollout hooks use to call the agent.
FROM radixark/miles:v0.2.0@sha256:c5e9ebe1dbb02faa094cf5036b57c4f3aadac9a53992f4d143823a6c20fb377c
RUN python -m pip install \
    "agentcore-rl-toolkit[rollout]==0.2.0" \
    "strands-agents[openai]==1.52.0" python-dotenv pytest pytest-asyncio
ENV PYTHONPATH=/root/miles:/root/Megatron-LM PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1
