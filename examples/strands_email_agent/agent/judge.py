"""Grades an answer against the reference with Claude on Amazon Bedrock."""

import logging
import os
import re

import boto3

logger = logging.getLogger(__name__)

MODEL = os.environ.get("JUDGE_MODEL", "global.anthropic.claude-haiku-5-5")
# The model may reason before it answers (up to ~450 tokens seen). With a small limit the reply is
# cut off empty and reads as "no verdict".
MAX_TOKENS = 2048
VOTES = 3
PROMPT = """You will be given an question and two different answers to the question, the correct answer and the answer given by an AI. Your job is to determine if the answer given by the AI is correct. Return True if the answer is semantically similar to the correct answer, and False otherwise. Dates, times, names and numbers in the AI answer must agree with the correct answer; an answer that omits or contradicts such a required detail is False. Return only the word True or False, no other text.

Question: {question}
Correct answer: {reference}
AI answer: {answer}"""  # noqa: E501

_client = None


def ask(prompt):
    """One verdict: True, False, or None when the call fails or the reply is not a verdict."""
    global _client
    _client = _client or boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION"))
    try:
        response = _client.converse(
            modelId=MODEL,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": MAX_TOKENS},
        )
    except Exception:
        logger.exception("Judge call failed")
        return None
    reply = "".join(block.get("text", "") for block in response["output"]["message"]["content"]).lower()
    said_true, said_false = bool(re.search(r"\btrue\b", reply)), bool(re.search(r"\bfalse\b", reply))
    return said_true if said_true != said_false else None


def judge(question, reference, answer):
    """Whether `answer` is correct: the majority of up to VOTES verdicts, or None without one.

    The model takes no temperature setting and flips on about 1 in 10 borderline answers, so an
    answer gets two votes and a third on a split.
    """
    prompt = PROMPT.format(question=question, reference=reference, answer=answer)
    votes = []
    for _ in range(VOTES):
        votes.append(ask(prompt))
        for verdict in (True, False):
            if votes.count(verdict) * 2 > VOTES:
                return verdict
    return None
