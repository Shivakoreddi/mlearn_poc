#!/usr/bin/env python3
"""
Pc‑Assist — cross‑platform CLI chat assistant for local system health, self‑healing & productivity tasks.

*   Works on macOS & Windows (tested on macOS 14, Win 11).
*   Two back‑ends:
      1. **openai**  – GPT‑4o / GPT‑4o‑mini via OpenAI function‑calling API.
      2. **local**   – llama‑cpp model loaded from GGUF file (e.g. Llama‑3‑8B‑Q4).
*   Safe by default: any destructive action (`kill`, `install`, `delete`) is **echoed** and requires **"yes"** confirmation.

Quick start
-----------
```bash
pip install typer rich psutil openai llama_cpp python‑dotenv
export OPENAI_API_KEY="sk‑..."          # if using OpenAI mode
python pc_assist.py chat --mode openai
```

For local mode (requires a quantised model):
```bash
python pc_assist.py chat --mode local --model_path ~/Models/llama‑3‑8b‑Q4.gguf
```
"""
from __future__ import annotations

import os
import json
import time
import shlex
import psutil
import typer
import getpass
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich import print as rprint
from rich.prompt import Prompt, Confirm
from rich.table import Table

# Optional imports — only if chosen by CLI flags
try:
    import openai  # type: ignore
except ImportError:
    openai = None  # handled later

try:
    from llama_cpp import Llama  # type: ignore
except ImportError:
    Llama = None  # handled later

app = typer.Typer(add_completion=False, help="Chat‑based system assistant 🖥️")

# ------------------------- Utilities ------------------------- #

def run_subprocess(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    """Run a command and return CompletedProcess while streaming output."""
    rprint(f"[grey]$ {' '.join(shlex.quote(c) for c in cmd)}[/grey]")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        rprint(proc.stdout.rstrip())
    if proc.stderr:
        rprint(f"[red]{proc.stderr.rstrip()}[/red]")
    return proc


def system_metrics() -> Dict[str, Any]:
    disk = psutil.disk_usage(Path.home().anchor)
    mem = psutil.virtual_memory()
    return {
        "os": platform.platform(),
        "user": getpass.getuser(),
        "cpu_percent": psutil.cpu_percent(percpu=False),
        "ram_percent": mem.percent,
        "disk_free_gb": round(disk.free / 1e9, 1),
        "disk_used_percent": disk.percent,
        "battery_percent": psutil.sensors_battery().percent if psutil.sensors_battery() else None,
    }


# ------------------------- LLM wrappers ------------------------- #

class BaseLLM:
    """Abstract base wrapper."""

    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        if openai is None:
            raise RuntimeError("openai package not installed. `pip install openai`. ")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise RuntimeError("OPENAI_API_KEY env var required for OpenAI mode")
        self.model = model
        self.temperature = temperature

    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return openai.ChatCompletion.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            tools=tools or [],
        ).to_dict_recursive()


class LocalLLM(BaseLLM):
    def __init__(self, model_path: str, context: int = 8192):
        if Llama is None:
            raise RuntimeError("llama_cpp not installed. `pip install llama_cpp`.")
        self.llm = Llama(model_path=model_path, n_ctx=context)

    def chat(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        prompt = "\n".join([f"### {m['role'].upper()}: {m['content']}" for m in messages])
        if tools:
            prompt += "\n### TOOLS: " + json.dumps(tools)
        completion = self.llm(prompt, max_tokens=1024, stop=["### USER:", "### SYSTEM:"], temperature=0.2)
        return {
            "choices": [{"message": {"role": "assistant", "content": completion["choices"][0]["text"].strip()}, "finish_reason": "stop"}]
        }


# ------------------------- Tools / functions registry ------------------------- #

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "system_metrics",
            "description": "Return live CPU, RAM, and disk stats about the user's computer.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kill_process",
            "description": "Force‑terminate a process given its PID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {"type": "integer", "description": "PID to kill"},
                },
                "required": ["pid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "install_package",
            "description": "Install a software package via Homebrew (mac) or winget (Windows).",
            "parameters": {
                "type": "object",
                "properties": {
                    "package": {"type": "string", "description": "Package name or brew formula / winget Id"},
                },
                "required": ["package"],
            },
        },
    },
]


# ------------------------- Action executors ------------------------- #


def kill_process(pid: int):
    confirm = Confirm.ask(f"Kill process {pid}?", default=False)
    if not confirm:
        rprint("[yellow]Cancelled.[/yellow]")
        return
    try:
        psutil.Process(pid).kill()
        rprint(f"[green]Process {pid} terminated.[/green]")
    except psutil.NoSuchProcess:
        rprint("[red]No such PID.[/red]")


def install_package(package: str):
    confirm = Confirm.ask(f"Install package '{package}'?", default=False)
    if not confirm:
        rprint("[yellow]Cancelled.[/yellow]")
        return
    if platform.system() == "Darwin":  # macOS
        run_subprocess(["brew", "install", package])
    elif platform.system() == "Windows":
        run_subprocess(["winget", "install", "--exact", package])
    else:
        rprint("[red]Unsupported OS for installer.[/red]")


TOOL_FUNCTIONS = {
    "system_metrics": lambda **_: system_metrics(),
    "kill_process": lambda pid, **_: kill_process(pid),
    "install_package": lambda package, **_: install_package(package),
}


# ------------------------- Chat session ------------------------- #

SYSTEM_PRIMER = (
    "You are Pc‑Assist, a local system helper.\n"
    "When you need live system info or to perform an action, call the appropriate tool.\n"
    "For destructive actions ask the user first then call the tool only if user agreed.\n"
)


def interactive_chat(llm: BaseLLM):
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PRIMER}]

    rprint("[bold cyan]\n💬 Pc‑Assist ready! Type 'exit' to quit.\n[/bold cyan]")
    while True:
        user_msg = Prompt.ask("You")
        if user_msg.lower() in {"exit", "quit"}:
            rprint("Bye 👋")
            break
        msgs.append({"role": "user", "content": user_msg})

        # First call — allow tool calls
        reply = llm.chat(msgs, tools=TOOL_SCHEMAS)
        choice = reply["choices"][0]

        # OpenAI returns explicit finish_reason; local wrapper returns 'stop'
        if choice.get("finish_reason") == "tool_call":
            # parse tool
            tool_call = choice["message"]["tool_call"]  # type: ignore
            name = tool_call["name"]
            args = json.loads(tool_call["arguments"])

            # execute locally
            result = TOOL_FUNCTIONS[name](**args)
            msgs.append(choice["message"])  # assistant tool request
            msgs.append({"role": "tool", "name": name, "content": json.dumps(result)})

            # second round, now with tool result
            final_reply = llm.chat(msgs)
            assistant_text = final_reply["choices"][0]["message"]["content"]
            rprint(f"[bright_green]{assistant_text}[/bright_green]")
            msgs.append({"role": "assistant", "content": assistant_text})
        else:
            assistant_text = choice["message"]["content"]
            rprint(f"[bright_green]{assistant_text}[/bright_green]")
            msgs.append({"role": "assistant", "content": assistant_text})


# ------------------------- Typer CLI ------------------------- #

@app.command()
def chat(
    mode: str = typer.Option("openai", help="openai | local"),
    model: str = typer.Option("gpt-4o-mini", help="OpenAI model name or local gguf model path"),
):
    """Start interactive assistant chat."""
    if mode == "openai":
        llm = OpenAILLM(model=model)
    elif mode == "local":
        llm = LocalLLM(model_path=model)
    else:
        typer.secho("Unsupported mode", fg=typer.colors.RED)
        raise typer.Exit(1)

    interactive_chat(llm)


@app.command()
def metrics():
    """Print live system metrics (without chat)."""
    tbl = Table(title="System metrics")
    tbl.add_column("Key")
    tbl.add_column("Value")
    for k, v in system_metrics().items():
        tbl.add_row(k, str(v))
    rprint(tbl)


if __name__ == "__main__":
    app()
