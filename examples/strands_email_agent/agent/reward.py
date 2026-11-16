"""The reward for one rollout, and the facts about the search it is built from.

    correct answer                      1.0 + 0.3 x citation overlap + 0.1 if it opened a right email
                                            + 0.1 / number of cited emails
    wrong answer                       -1.0 + partial credit
    declined, or no answer              0.0 + partial credit
    ran out of turns or context        -2.0 + partial credit

    partial credit (at most 0.5)        0.2 a right email was in a search result
                                      + 0.15 it opened a right email
                                      + 0.05 it never tried to open an email that does not exist
                                      + 0.1 x citation overlap

Running out of turns scores below a wrong answer on purpose: scored at about 0, never answering
beats a wrong answer and the policy learns to repeat a tool call until the turn cap.
"""

import json
import re

# An answer that says the emails do not contain the answer is declining, not guessing.
DECLINED = re.compile(
    r"\b(not (specified|available|mentioned|provided|found|stated|included|indicated|identified)|"
    r"no (specific|information|details?|mention|record)|cannot (be )?determine|unable to (determine|find)|"
    r"(does|do) not (specify|mention|state|contain|provide|indicate)|"
    r"i don'?t know)\b",
    re.IGNORECASE,
)


def declined(answer):
    return bool(DECLINED.search(answer))


def search_facts(messages, right_emails):
    """What the agent's tool calls did, read from a Strands transcript."""
    results = {}  # toolUseId -> the tool's parsed JSON result
    for message in messages:
        for block in message.get("content") or []:
            if "toolResult" in block:
                text = "".join(part.get("text", "") for part in block["toolResult"].get("content") or [])
                try:
                    results[block["toolResult"]["toolUseId"]] = json.loads(text)
                except ValueError:
                    pass
    facts = {"found_right_email": False, "read_right_email": False, "invalid_read": False, "turns": 0}
    for message in messages:
        calls = [block["toolUse"] for block in message.get("content") or [] if "toolUse" in block]
        facts["turns"] += bool(calls)
        for call in calls:
            result = results.get(call["toolUseId"])
            if call["name"] == "search_inbox" and isinstance(result, list):
                facts["found_right_email"] |= any(row.get("message_id") in right_emails for row in result)
            elif call["name"] == "read_email":
                facts["read_right_email"] |= (call.get("input") or {}).get("message_id") in right_emails
                facts["invalid_read"] |= isinstance(result, dict) and "error" in result
    return facts


def score(facts, answer, sources, right_emails, correct, out_of_budget):
    """(reward, metrics). `correct` is the judge's verdict, or None when there was nothing to judge."""
    cited, right = set(sources), set(right_emails)
    overlap = len(cited & right) / len(cited | right) if cited | right else 0.0
    partial = (
        0.2 * facts["found_right_email"]
        + 0.15 * facts["read_right_email"]
        + 0.05 * (not facts["invalid_read"])
        + 0.1 * overlap
    )
    if out_of_budget:
        reward = -2.0 + partial
    elif correct:
        reward = 1.0 + 0.3 * overlap + 0.1 * facts["read_right_email"] + (0.1 / len(cited) if cited else 0.0)
    elif correct is None:
        reward = partial
    else:
        reward = -1.0 + partial
    metrics = {
        "answer_correct": float(bool(correct)),
        "answered": float(correct is not None),
        "declined": float(bool(answer) and declined(answer)),
        "out_of_budget": float(out_of_budget),
        "found_right_email": float(facts["found_right_email"]),
        "read_right_email": float(facts["read_right_email"]),
        "invalid_read": float(facts["invalid_read"]),
        "citation_overlap": overlap,
        "num_sources": float(len(cited)),
        "num_turns": float(facts["turns"]),
    }
    return reward, metrics
