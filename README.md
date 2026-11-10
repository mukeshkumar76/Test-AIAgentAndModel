# AI Agent Framework (OpenAI + Local Ollama Fallback)

A modular, extensible AI Agent built with Python, OpenAI API, and an automatic fallback to local **Ollama** running **`gemma4`**.

---

## ✨ Features

- **Hybrid AI Execution (OpenAI + Local Ollama)**:
  - Uses OpenAI by default (`gpt-4o-mini`, `gpt-4o`, etc.).
  - **Auto Fallback**: If OpenAI is not configured, unreachable, or times out, it automatically switches to local **Ollama** with model **`gemma4`**.
  - Provider selection: Choose between `auto` (fallback enabled), `openai` (strict), or `ollama` (local offline).
- **Autonomous Tool Execution**:
  - Automatically loops through function/tool calling for both OpenAI and Ollama.
- **Built-in Tools**:
  - `calculate`: Safe mathematical expression evaluator.
  - `get_current_datetime`: Returns current system date and time.
  - `get_system_info`: Returns host OS, Python runtime, and machine details.
  - `read_file` & `write_file`: Workspace-scoped file I/O.
  - `list_files`: Workspace directory inspection.
- **Session Persistence**:
  - Save and resume chat sessions via JSON (`save [file.json]` and `load [file.json]`).

---

## 📁 Project Structure

```
Agent-Test/
├── .env                 # Environment configuration (API keys, models)
├── .env.example         # Template for environment settings
├── .gitignore           # Git ignore rules (.env, .venv, pycache)
├── requirements.txt     # Python dependencies
├── config.py            # Pydantic settings & Ollama health checks
├── tools.py             # Tools & OpenAI/Ollama function calling schemas
├── agent.py             # Agent engine with autonomous fallback loop
├── main.py              # CLI entrypoint (interactive REPL & single query)
└── README.md            # Documentation
```

---

## 🚀 Quickstart

### 1. Virtual Environment & Dependencies

```powershell
# Create virtual environment
python -m venv .venv

# Activate (PowerShell on Windows)
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Configuration (`.env`)

Configure your settings in `.env`:

```env
# OpenAI Settings
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
TEMPERATURE=0.7
MAX_TOKENS=2048

# Local Ollama Fallback Settings
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=gemma4
ENABLE_OLLAMA_FALLBACK=true
```

---

## 💻 Usage

### 1. Interactive Chat Session

Start the chat loop:
```powershell
python main.py
```
If your OpenAI key is configured, it will use OpenAI. If connection fails or no key is present, it will automatically route to local Ollama with `gemma4`.

#### Interactive In-Chat Commands:
- `save [filename.json]`: Save conversation history.
- `load [filename.json]`: Restore conversation history.
- `reset` or `clear`: Reset agent memory.
- `exit` or `quit`: Exit the session.

### 2. Force Local Ollama (`gemma4`) Directly

Run offline locally using your local Ollama instance:
```powershell
python main.py --provider ollama
```

### 3. Single Query Mode

Run a single command non-interactively:
```powershell
python main.py --prompt "What is 100 / 4 + 15? Use the calculate tool."
```
