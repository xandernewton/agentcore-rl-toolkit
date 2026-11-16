"""The model's Miles session family against the model's own chat template.

The session keeps each sampled assistant turn as tokens and renders only the messages appended
after it, so the prompt Miles builds turn by turn must equal the model's native rendering of the
whole conversation. Run it for any model you add a recipe for.
"""

from pathlib import Path

import pytest
from email_miles.launch import model_path
from miles.utils.chat_template_utils import tito_tokenizer
from miles.utils.chat_template_utils.template import extract_tool_dicts
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType, get_tito_tokenizer
from transformers import AutoTokenizer

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_inbox",
            "description": "Search the inbox.",
            "parameters": {
                "type": "object",
                "properties": {"keywords": {"type": "array", "items": {"type": "string"}}},
                "required": ["keywords"],
            },
        },
    }
]
# (HF repo, Miles session family, the reasoning the model emits before its tool call)
MODELS = [
    ("Qwen/Qwen3-8B", "qwen3", "The user asks about a meeting; search for it."),
]


@pytest.fixture(scope="module", params=MODELS, ids=[m[0] for m in MODELS])
def model(request):
    repo, family, reasoning = request.param
    try:
        path = model_path(repo)
    except FileNotFoundError as missing:
        pytest.skip(str(missing))
    # Miles serves and tokenizes with its own fixed copy of the family's template, not the model's.
    fixed = TITOTokenizerType.get_tokenizer_class(TITOTokenizerType(family)).FIXED_TEMPLATE.template
    session_tokenizer = AutoTokenizer.from_pretrained(path)
    session_tokenizer.chat_template = (Path(tito_tokenizer.__file__).parent / "templates" / fixed).read_text()
    return AutoTokenizer.from_pretrained(path), get_tito_tokenizer(session_tokenizer, family), reasoning


def native(tokenizer, messages, tokenize=True, add_generation_prompt=True):
    """The model's own template; tools serialized as SGLang serves them (through its Tool model)."""
    return tokenizer.apply_chat_template(
        messages,
        tools=extract_tool_dicts(TOOLS),
        add_generation_prompt=add_generation_prompt,
        tokenize=tokenize,
        return_dict=False,
    )


def test_incremental_prompt_matches_native_template(model):
    tokenizer, tito, reasoning = model
    args = tito.default_template_args(TOOLS)
    history = [
        {"role": "system", "content": "You are an email search agent."},
        {"role": "user", "content": "When is the meeting?"},
    ]
    prompt = tito.apply_chat_template(history, add_generation_prompt=True, tokenize=True, template_args=args)
    assert prompt == native(tokenizer, history)

    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "search_inbox", "arguments": {"keywords": ["meeting"]}},
    }
    assistant = {"role": "assistant", "content": "", "tool_calls": [call]}
    if reasoning:
        assistant["reasoning_content"] = reasoning
    tool = {"role": "tool", "tool_call_id": "c1", "content": '[{"message_id": "<m1>"}]'}

    # What the model samples for this turn: the native rendering of the assistant message after
    # the generation prompt, ending at <|im_end|> (the model does not emit the trailing newline).
    prompt_text = native(tokenizer, history, tokenize=False)
    turn_text = native(tokenizer, history + [assistant], tokenize=False, add_generation_prompt=False)
    assert turn_text.startswith(prompt_text)
    sampled = tokenizer.encode(turn_text[len(prompt_text) :].rstrip("\n"), add_special_tokens=False)

    merged = tito.merge_tokens(history + [assistant], history + [assistant, tool], prompt + sampled, template_args=args)
    expected = native(tokenizer, history + [assistant, tool])
    assert tokenizer.decode(merged) == tokenizer.decode(expected)
    assert merged == expected
