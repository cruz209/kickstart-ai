
from typing import Dict, Any
import json

try:
    from openai import OpenAI
except ImportError:  # v0: soft dependency
    OpenAI = None


class OpenAIBackend:
    """Simple OpenAI Chat Completions backend for LiftOff v0."""

    def __init__(self, api_key: str) -> None:
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAIBackend")
        self.client = OpenAI(api_key=api_key)

    def generate_project(self, meta_prompt: str, metadata: Dict[str, Any]) -> Dict[str, str]:
        # v0: single call, no fancy retries
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a code scaffolding engine that outputs JSON only."},
                {"role": "user", "content": meta_prompt},
            ],
            temperature=0.2,
        )
        raw = completion.choices[0].message.content
        # Expect raw to be pure JSON
        try:
            file_tree = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"OpenAI backend returned non-JSON output: {e}: {raw[:200]}")
        if not isinstance(file_tree, dict):
            raise ValueError("Expected JSON object mapping file paths to contents")
        return {str(k): str(v) for k, v in file_tree.items()}
