"""
LLM Service for the GitHub Issue Solver.

Unified provider routing via LiteLLM — supports Gemini, Claude, Grok, OpenAI,
Ollama, and 100+ other providers through a single interface.
"""

import os
from loguru import logger
from langchain_core.language_models.chat_models import BaseChatModel


# ── Provider registry ──────────────────────────────────────────
# (litellm_prefix, api_key_env_var, default_model)
PROVIDERS = {
    "gemini":  ("gemini",    "GOOGLE_API_KEY",    "gemini-2.5-flash"),
    "claude":  ("anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-4-5-20241022"),
    "grok":    ("xai",       "XAI_API_KEY",       "grok-3-mini"),
    "openai":  ("openai",    "OPENAI_API_KEY",    "gpt-4o-mini"),
    "ollama":  ("ollama",    None,                "llama3.1"),
}


def list_providers() -> list[str]:
    """Return all supported provider names."""
    return list(PROVIDERS.keys())


def get_provider_info(provider: str) -> dict:
    """Return provider metadata (prefix, env var, default model)."""
    if provider not in PROVIDERS:
        return {}
    prefix, env_key, default_model = PROVIDERS[provider]
    return {
        "prefix": prefix,
        "api_key_env": env_key,
        "default_model": default_model,
        "needs_api_key": env_key is not None,
    }


class LLMService:
    """Unified LLM service powered by LiteLLM."""

    def __init__(self, config):
        self.config = config
        self._llm = None

    def get_llm(self) -> BaseChatModel:
        """Get or create the LLM instance (lazy init)."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self) -> BaseChatModel:
        """Create LLM via LiteLLM router."""
        from litellm import validate_environment
        from langchain_community.chat_models import ChatLiteLLM

        provider = self.config.llm_provider

        if provider not in PROVIDERS:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Supported: {', '.join(PROVIDERS)}"
            )

        prefix, env_key, default_model = PROVIDERS[provider]

        # Validate API key is set (skip for local providers like Ollama)
        if env_key:
            api_key = os.environ.get(env_key)
            if not api_key:
                raise ValueError(
                    f"{env_key} is required for provider '{provider}'. "
                    f"Set it in your .env or environment."
                )

        # Build the litellm model string: "prefix/model"
        model_name = self.config.llm_model_name or default_model
        if prefix and not model_name.startswith(f"{prefix}/"):
            model_string = f"{prefix}/{model_name}"
        else:
            model_string = model_name

        logger.info(f"Initializing LLM via LiteLLM: {model_string}")

        return ChatLiteLLM(
            model=model_string,
            temperature=0.1,
            max_retries=2,
            request_timeout=120,
        )

    @property
    def provider_name(self) -> str:
        return self.config.llm_provider

    @property
    def model_name(self) -> str:
        provider = self.config.llm_provider
        if provider in PROVIDERS:
            return self.config.llm_model_name or PROVIDERS[provider][2]
        return self.config.llm_model_name or "unknown"
