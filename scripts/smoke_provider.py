"""Live one-call smoke: validates the key, model slug, and usage telemetry.

Usage: uv run python scripts/smoke_provider.py [model-slug]
"""

import sys

from context_eval.env import load_env
from context_eval.provider import OpenRouterClient


def main() -> None:
    load_env()
    model = sys.argv[1] if len(sys.argv) > 1 else OpenRouterClient().model
    client = OpenRouterClient(model=model)
    result = client.complete(
        [{"role": "user", "content": "Reply with exactly: ok"}],
        max_tokens=8,
    )
    print("requested model:", model)
    print("resolved model: ", result.model)
    print("text:           ", repr(result.text.strip()))
    print("usage:          ", result.usage)


if __name__ == "__main__":
    main()
