"""Live one-call smoke: validates the key, model slug, and usage telemetry.

Usage: uv run python scripts/smoke_provider.py [model-slug]
"""

import sys

from context_eval.env import load_env
from context_eval.provider import OpenRouterClient
from context_eval.tokens import MESSAGE_OVERHEAD_TOKENS, default_counter
from context_eval.types import PromptMessage, Role


def main() -> None:
    load_env()
    model = sys.argv[1] if len(sys.argv) > 1 else OpenRouterClient().model
    prompt = "Reply with exactly: ok"
    client = OpenRouterClient(model=model)
    result = client.complete(
        [{"role": "user", "content": prompt}],
        # Reasoning models burn completion tokens on thinking first.
        max_tokens=512,
    )

    print("requested model:", model)
    print("resolved model: ", result.model)
    print("provider:       ", result.provider or "<not reported>")
    print("finish_reason:  ", result.finish_reason)
    print("text:           ", repr(result.text.strip()[:80]))
    print(
        f"usage:           prompt={result.usage.prompt_tokens}"
        f" completion={result.usage.completion_tokens}"
        f" reasoning={result.usage.reasoning_tokens}"
    )

    counter = default_counter()
    local = counter.count_messages([PromptMessage(role=Role.USER, content=prompt)])
    print(
        f"token check:     local={local} (incl. {MESSAGE_OVERHEAD_TOKENS} framing) "
        f"vs reported prompt={result.usage.prompt_tokens}"
    )


if __name__ == "__main__":
    main()
