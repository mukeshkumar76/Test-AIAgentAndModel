"""Main entry point for running the AI Agent with OpenAI and Ollama support."""

import argparse
import sys
from config import settings
from agent import OpenAIAgent


def print_banner(active_provider: str, active_model: str, ollama_ready: bool) -> None:
    print("=" * 65)
    print("           AI Agent Framework (OpenAI + Local Ollama)")
    print(f"  Target Provider : {active_provider.upper()} ({active_model})")
    print(f"  Ollama Fallback : {'READY (gemma4)' if ollama_ready else 'NOT DETECTED'}")
    print("=" * 65)


def interactive_mode(agent: OpenAIAgent) -> None:
    """Run interactive REPL session with the AI Agent."""
    ollama_ready = settings.is_ollama_running()
    print_banner(agent.preferred_provider, agent.active_model, ollama_ready)
    print("Commands:")
    print("  'exit' / 'quit' / 'q'    -> Exit session")
    print("  'reset' / 'clear'        -> Clear conversation memory")
    print("  'save' [filename.json]   -> Save chat history to JSON")
    print("  'load' [filename.json]   -> Load chat history from JSON")
    print("-" * 65)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd in {"exit", "quit", "q"}:
                print("\nGoodbye!")
                break

            if cmd in {"reset", "clear"}:
                agent.reset()
                print("\n[Conversation memory has been reset.]")
                continue

            if cmd == "save":
                path = parts[1] if len(parts) > 1 else "chat_history.json"
                print(f"\n[{agent.save_history(path)}]")
                continue

            if cmd == "load":
                path = parts[1] if len(parts) > 1 else "chat_history.json"
                print(f"\n[{agent.load_history(path)}]")
                continue

            print("\nAgent is thinking...")
            response = agent.chat(user_input)
            print(f"\nAgent [{agent.active_provider.upper()} - {agent.active_model}]:\n{response}")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except Exception as exc:
            print(f"\n[Error]: {exc}")


def single_query_mode(agent: OpenAIAgent, prompt: str) -> None:
    """Execute a single query and print the result."""
    response = agent.chat(prompt)
    print(f"[{agent.active_provider.upper()} - {agent.active_model}]:\n{response}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI Agent (OpenAI + Ollama)")
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        help="Execute a single prompt non-interactively and print the response.",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Override the model (OpenAI model or Ollama model depending on provider).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["auto", "openai", "ollama"],
        default="auto",
        help="Choose preferred provider: 'auto' (try OpenAI then fallback to Ollama), 'openai', or 'ollama'.",
    )
    args = parser.parse_args()

    # Determine default provider behavior
    provider = args.provider
    openai_configured = settings.is_api_key_configured()
    ollama_running = settings.is_ollama_running()

    # If OpenAI key is not configured and auto mode is used, intelligently default to Ollama
    if provider == "auto" and not openai_configured:
        if ollama_running:
            print("[Info] OpenAI API key is not configured. Defaulting to local Ollama (gemma4).")
            provider = "ollama"
        else:
            print("=" * 65)
            print("  NOTICE: Neither OpenAI API Key is set, nor is Ollama running.")
            print("  Please configure OPENAI_API_KEY in .env or run `ollama serve`.")
            print("=" * 65)

    if args.model:
        if provider == "ollama":
            settings.ollama_model = args.model
        else:
            settings.openai_model = args.model

    agent = OpenAIAgent(preferred_provider=provider)

    if args.prompt:
        single_query_mode(agent, args.prompt)
    else:
        interactive_mode(agent)


if __name__ == "__main__":
    main()
