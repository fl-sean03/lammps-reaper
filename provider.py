"""Anthropic provider with retry logic for LAMMPS Reaper."""

import asyncio
import os
from typing import Optional

import anthropic


class AnthropicProvider:
    """Minimal Anthropic provider with retry logic."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # seconds

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key. If not provided, reads from
                     ANTHROPIC_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set in ANTHROPIC_API_KEY environment variable"
            )
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = os.environ.get("CLAUDE_MODEL", self.DEFAULT_MODEL)

    async def create_message(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4096,
    ) -> str:
        """Create a message with retry logic.

        Args:
            system_prompt: System prompt for the conversation.
            user_message: User message content.
            max_tokens: Maximum tokens in the response.

        Returns:
            The response content as a string.

        Raises:
            anthropic.APIError: If all retries are exhausted.
        """
        last_exception = None

        for attempt in range(self.MAX_RETRIES):
            try:
                # Run synchronous API call in thread pool
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )
                # Extract text content from response
                return response.content[0].text

            except anthropic.RateLimitError as e:
                last_exception = e
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_DELAY * (2**attempt)  # 1s, 2s, 4s
                    await asyncio.sleep(delay)
                continue

            except anthropic.APIStatusError as e:
                # For other API errors, retry with backoff
                last_exception = e
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_DELAY * (2**attempt)
                    await asyncio.sleep(delay)
                continue

        raise last_exception

    async def health_check(self) -> bool:
        """Perform a simple health check.

        Returns:
            True if the API is healthy and responding, False otherwise.
        """
        try:
            response = await self.create_message(
                system_prompt="You are a health check assistant.",
                user_message="Say OK",
                max_tokens=10,
            )
            return "OK" in response.upper()
        except Exception:
            return False
