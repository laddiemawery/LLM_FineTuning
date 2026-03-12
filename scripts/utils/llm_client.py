"""LLM API client wrapper for dataset generation."""

import json
import time

from anthropic import Anthropic

from .config import Config


class LLMClient:
    """Wrapper around the Anthropic API for generating training data."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.client = Anthropic(api_key=self.config.anthropic_api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send a prompt to Claude and return the response text."""
        response = self.client.messages.create(
            model=self.config.llm_model,
            max_tokens=max_tokens or self.config.llm_max_tokens,
            temperature=temperature if temperature is not None else self.config.llm_temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[dict]:
        """Generate and parse a JSON response. Returns a list of dicts."""
        raw = self.generate(system_prompt, user_prompt, max_tokens, temperature)

        # Extract JSON from the response (handle markdown code blocks)
        text = raw.strip()
        if text.startswith("```"):
            # Remove code block markers
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        # Try parsing as a JSON array first
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
            return [result]
        except json.JSONDecodeError:
            pass

        # Try extracting individual JSON objects
        results = []
        depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(text[start : i + 1])
                        results.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = None

        return results

    def generate_batch(
        self,
        system_prompt: str,
        user_prompts: list[str],
        max_tokens: int | None = None,
        temperature: float | None = None,
        delay: float = 0.5,
    ) -> list[list[dict]]:
        """Generate JSON responses for multiple prompts with rate limiting."""
        results = []
        for prompt in user_prompts:
            result = self.generate_json(system_prompt, prompt, max_tokens, temperature)
            results.append(result)
            time.sleep(delay)
        return results
