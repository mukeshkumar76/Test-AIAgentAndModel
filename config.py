"""Configuration loader for the AI Agent with OpenAI and Ollama fallback support."""

import json
import os
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    # OpenAI Settings
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7"))
    )
    max_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS", "2048"))
    )

    # Local Ollama Fallback Settings
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    )
    ollama_model: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "gemma4")
    )
    enable_ollama_fallback: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_OLLAMA_FALLBACK", "true").lower() in {"true", "1", "yes"}
    )

    # Agent System Prompt
    system_prompt: str = Field(
        default=(
            "You are a helpful, intelligent, and capable AI Agent. "
            "You have access to tools that you can execute when needed to assist the user. "
            "Always be concise, accurate, and structured in your explanations."
        )
    )

    def is_api_key_configured(self) -> bool:
        """Check if a valid OpenAI API key is configured."""
        placeholder_values = {"", "your_openai_api_key_here", "sk-...", "your_actual_openai_api_key_here"}
        return bool(self.openai_api_key and self.openai_api_key.strip() not in placeholder_values)

    def is_ollama_running(self) -> bool:
        """Check if local Ollama server is accessible."""
        # Derive base host: e.g. http://localhost:11434 from http://localhost:11434/v1
        base_host = self.ollama_base_url.replace("/v1", "").rstrip("/")
        tags_url = f"{base_host}/api/tags"
        try:
            req = urllib.request.Request(tags_url, headers={"User-Agent": "AIAgent/1.0"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_ollama_models(self) -> list:
        """Retrieve list of locally downloaded models in Ollama."""
        base_host = self.ollama_base_url.replace("/v1", "").rstrip("/")
        tags_url = f"{base_host}/api/tags"
        try:
            req = urllib.request.Request(tags_url, headers={"User-Agent": "AIAgent/1.0"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []


# Global settings instance
settings = Settings()
