"""OpenAI Agent Implementation with Local Ollama Fallback and Autonomous Tool Execution."""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from openai import OpenAI, OpenAIError

from config import Settings, settings
from tools import TOOL_DEFINITIONS, TOOL_REGISTRY, execute_tool

logger = logging.getLogger(__name__)


class OpenAIAgent:
    """An AI Agent that leverages OpenAI API with an automatic fallback to local Ollama (gemma4)."""

    def __init__(
        self,
        config: Optional[Settings] = None,
        custom_system_prompt: Optional[str] = None,
        preferred_provider: str = "auto",  # 'auto', 'openai', or 'ollama'
        mock_mode: bool = False,
    ):
        self.config = config or settings
        self.system_prompt = custom_system_prompt or self.config.system_prompt
        self.preferred_provider = preferred_provider.lower()
        self.mock_mode = mock_mode

        self.tools = list(TOOL_DEFINITIONS)
        self.tool_registry = dict(TOOL_REGISTRY)

        # Active runtime provider information and fallback state
        self.active_provider: str = "openai"
        self.active_model: str = self.config.openai_model
        self._fallback_active: bool = (self.preferred_provider == "ollama")

        # Initialize conversation history with system prompt
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        self._openai_client: Optional[OpenAI] = None
        self._ollama_client: Optional[OpenAI] = None

    @property
    def openai_client(self) -> Optional[OpenAI]:
        """Lazy-loaded OpenAI official client."""
        if self._openai_client is None and self.config.is_api_key_configured():
            self._openai_client = OpenAI(
                api_key=self.config.openai_api_key,
                timeout=5.0,  # fast fail if network or API unreachable
            )
        return self._openai_client

    @property
    def ollama_client(self) -> OpenAI:
        """Client configured for local Ollama OpenAI-compatible endpoint."""
        if self._ollama_client is None:
            self._ollama_client = OpenAI(
                base_url=self.config.ollama_base_url,
                api_key="ollama",  # dummy key required by SDK
                timeout=60.0,
            )
        return self._ollama_client

    def register_tool(
        self,
        name: str,
        function: Callable[..., Any],
        schema: Dict[str, Any],
    ) -> None:
        """Register a custom tool function and its schema."""
        self.tool_registry[name] = function
        self.tools.append(schema)

    def reset(self) -> None:
        """Clear conversation history and restore system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        if self.preferred_provider != "ollama":
            self._fallback_active = False

    def save_history(self, file_path: str = "chat_history.json") -> str:
        """Export current conversation history to JSON."""
        target = Path(file_path)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, indent=2, ensure_ascii=False)
        return f"Saved conversation history ({len(self.messages)} entries) to '{file_path}'."

    def load_history(self, file_path: str = "chat_history.json") -> str:
        """Load conversation history from JSON."""
        target = Path(file_path)
        if not target.exists():
            return f"Error: File '{file_path}' does not exist."
        with open(target, "r", encoding="utf-8") as f:
            self.messages = json.load(f)
        return f"Loaded conversation history ({len(self.messages)} entries) from '{file_path}'."

    def _call_completion(self, client: OpenAI, model: str) -> Any:
        """Perform a single completion call with tools."""
        return client.chat.completions.create(
            model=model,
            messages=self.messages,
            tools=self.tools if self.tools else None,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    def _get_completion_with_fallback(self) -> tuple[Any, str, str]:
        """Obtain completion, falling back to local Ollama (gemma4) if OpenAI fails."""
        # Case 1: Ollama requested or already latched to fallback
        if self.preferred_provider == "ollama" or self._fallback_active:
            response = self._call_completion(self.ollama_client, self.config.ollama_model)
            return response, "ollama", self.config.ollama_model

        # Case 2: Try OpenAI
        openai_available = self.config.is_api_key_configured() and self.openai_client is not None

        if openai_available:
            try:
                response = self._call_completion(self.openai_client, self.config.openai_model)
                return response, "openai", self.config.openai_model
            except Exception as openai_err:
                logger.warning(f"OpenAI completion error: {openai_err}")
                if not self.config.enable_ollama_fallback:
                    raise openai_err
                print(f"\n[Fallback Triggered]: OpenAI connection failed ({type(openai_err).__name__}).")
                print(f"[Fallback]: Switching to local Ollama (model: {self.config.ollama_model})...\n")
                self._fallback_active = True
        else:
            if not self.config.enable_ollama_fallback:
                raise RuntimeError("OpenAI API key is missing and Ollama fallback is disabled.")
            self._fallback_active = True

        # Fallback to local Ollama
        try:
            response = self._call_completion(self.ollama_client, self.config.ollama_model)
            return response, "ollama", self.config.ollama_model
        except Exception as ollama_err:
            raise RuntimeError(
                f"Both OpenAI and local Ollama ({self.config.ollama_model}) failed to respond.\n"
                f"Ollama error: {ollama_err}\n"
                "Make sure Ollama is running (`ollama serve`) and the model is downloaded (`ollama pull gemma4`)."
            ) from ollama_err

    def chat(self, user_input: str, max_tool_iterations: int = 10) -> str:
        """Send a message to the agent and receive the response.

        Handles reasoning, tool calling, and automatic OpenAI -> Ollama fallback.
        """
        # Append user message
        self.messages.append({"role": "user", "content": user_input})

        iteration = 0
        while iteration < max_tool_iterations:
            iteration += 1

            try:
                response, provider, model = self._get_completion_with_fallback()
                self.active_provider = provider
                self.active_model = model
            except Exception as exc:
                error_msg = f"[Agent Error]: {str(exc)}"
                logger.error(error_msg)
                return error_msg

            choice = response.choices[0]
            message = choice.message

            # Check if the model requested tool calls
            if message.tool_calls:
                # Add assistant message with tool calls to conversation history
                self.messages.append(message.model_dump(exclude_unset=True))

                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    raw_args = tool_call.function.arguments

                    try:
                        args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        args = {}

                    print(f"  [{provider.upper()} Tool Call] -> {fn_name}({args})")

                    # Execute the tool
                    if fn_name in self.tool_registry:
                        try:
                            tool_result = str(self.tool_registry[fn_name](**args))
                        except Exception as e:
                            tool_result = json.dumps({"error": f"Tool execution error: {e}"})
                    else:
                        tool_result = execute_tool(fn_name, args)

                    # Append tool result to messages
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fn_name,
                        "content": tool_result,
                    })

                # Loop back so the model can process the tool results
                continue

            # Final assistant response
            content = message.content or ""
            self.messages.append({"role": "assistant", "content": content})
            return content

        return "Agent exceeded maximum tool call iterations."
