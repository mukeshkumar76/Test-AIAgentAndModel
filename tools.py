"""Built-in tools for the OpenAI Agent."""

import ast
import json
import math
import operator
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

# Workspace boundary for safe file operations
WORKSPACE_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Safe Math Evaluator
# ---------------------------------------------------------------------------
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
}


def _eval_ast_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[op_type](_eval_ast_node(node.operand))
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            left = _eval_ast_node(node.left)
            right = _eval_ast_node(node.right)
            return _SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCTIONS:
            func = _SAFE_FUNCTIONS[node.func.id]
            args = [_eval_ast_node(arg) for arg in node.args]
            return func(*args)
        raise ValueError("Unsupported function call")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCTIONS:
            return _SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Unknown variable: {node.id}")
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        parsed = ast.parse(expression.strip(), mode="eval")
        result = _eval_ast_node(parsed)
        return json.dumps({"expression": expression, "result": result})
    except Exception as exc:
        return json.dumps({"error": f"Failed to evaluate expression: {exc}"})


def get_current_datetime() -> str:
    """Get the current date, time, and timezone information."""
    now = datetime.now()
    return json.dumps({
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "iso_format": now.isoformat(),
        "weekday": now.strftime("%A"),
    })


def get_system_info() -> str:
    """Get basic information about the operating system and Python runtime."""
    return json.dumps({
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.architecture()[0],
        "python_version": sys.version.split()[0],
    })


def _resolve_safe_path(rel_path: str) -> Path:
    """Ensure the target path stays within the workspace boundary."""
    target = (WORKSPACE_ROOT / rel_path).resolve()
    try:
        target.relative_to(WORKSPACE_ROOT)
    except ValueError:
        raise PermissionError(f"Access denied: Path '{rel_path}' is outside the workspace.")
    return target


def read_file(file_path: str) -> str:
    """Read contents of a text file inside the workspace."""
    try:
        resolved = _resolve_safe_path(file_path)
        if not resolved.exists():
            return json.dumps({"error": f"File '{file_path}' does not exist."})
        if not resolved.is_file():
            return json.dumps({"error": f"'{file_path}' is not a regular file."})
        content = resolved.read_text(encoding="utf-8")
        return json.dumps({"file_path": file_path, "content": content})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def write_file(file_path: str, content: str, append: bool = False) -> str:
    """Write or append text content to a file inside the workspace."""
    try:
        resolved = _resolve_safe_path(file_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(resolved, mode, encoding="utf-8") as f:
            f.write(content)
        action = "appended to" if append else "written to"
        return json.dumps({"status": "success", "message": f"Successfully {action} '{file_path}'."})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def list_files(directory: str = ".") -> str:
    """List files and folders in a workspace directory."""
    try:
        resolved = _resolve_safe_path(directory)
        if not resolved.exists() or not resolved.is_dir():
            return json.dumps({"error": f"Directory '{directory}' does not exist."})
        entries = []
        for item in sorted(resolved.iterdir()):
            if item.name.startswith(".venv") or item.name == "__pycache__":
                continue
            entries.append({
                "name": item.name,
                "is_dir": item.is_dir(),
                "size_bytes": item.stat().st_size if item.is_file() else None,
            })
        return json.dumps({"directory": directory, "entries": entries})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool Registry and OpenAI Schemas
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression safely (supports arithmetic operations +, -, *, /, **, %, and functions like sqrt, round, abs).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g., '14 * 25 + sqrt(144)'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Retrieve the current local date, time, and day of the week.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Retrieve system environment information such as operating system and Python version.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file located inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The relative path of the file to read (e.g. 'notes.txt', 'config.py').",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or append text content to a file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The relative path of the file to write to.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content to write.",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "Set to true to append content instead of overwriting.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and subdirectories in a directory within the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Relative directory path to list (defaults to '.' for root).",
                    }
                },
            },
        },
    },
]

TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "calculate": calculate,
    "get_current_datetime": get_current_datetime,
    "get_system_info": get_system_info,
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
}


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute a registered tool by name with the given arguments."""
    tool_fn = TOOL_REGISTRY.get(tool_name)
    if not tool_fn:
        return json.dumps({"error": f"Tool '{tool_name}' not found."})

    try:
        return str(tool_fn(**arguments))
    except Exception as exc:
        return json.dumps({"error": f"Error running tool '{tool_name}': {str(exc)}"})
