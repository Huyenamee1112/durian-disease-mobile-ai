"""Local stdio MCP server backed by the project's authenticated Kaggle CLI."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / ".venv/bin/kaggle"
OUTPUT_ROOT = ROOT / "ai/kaggle_outputs"
mcp = FastMCP("kaggle")
READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=True)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)


def kernel_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+(?:/[0-9]+)?", value):
        raise ValueError("Use owner/notebook-slug, optionally followed by /version.")
    return value


def under(path: str, base: Path) -> Path:
    candidate = (base / path).resolve()
    if not candidate.is_relative_to(base.resolve()):
        raise ValueError(f"Path must remain inside {base.relative_to(ROOT)}.")
    return candidate


def redact(text: str) -> str:
    for key, value in os.environ.items():
        if value and len(value) >= 8 and any(word in key.upper() for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")):
            text = text.replace(value, "[REDACTED]")
    return text


def run_cli(args: list[str], timeout: int = 90) -> dict:
    if not CLI.is_file():
        raise ValueError("Install Kaggle in the project's .venv first.")
    # File-backed output avoids keeping large training logs in memory.
    with tempfile.TemporaryFile() as stream:
        try:
            result = subprocess.run(
                [str(CLI), "kernels", *args], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT, timeout=timeout,
                stdin=subprocess.DEVNULL, check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "message": "Kaggle request timed out. Check status before retrying a submission."}
        size = stream.tell()
        stream.seek(max(0, size - 24000))
        output = redact(stream.read().decode("utf-8", errors="replace"))
    return {"ok": result.returncode == 0, "exit_code": result.returncode,
            "output": output, "truncated": size > 24000}


@mcp.tool(annotations=READ)
async def list_notebooks() -> dict:
    """List the authenticated user's notebooks. Does not start training."""
    return await asyncio.to_thread(run_cli, ["list", "--mine"])


@mcp.tool(annotations=READ)
async def notebook_status(notebook: str) -> dict:
    """Get the latest run status of owner/notebook-slug."""
    return await asyncio.to_thread(run_cli, ["status", kernel_id(notebook)])


@mcp.tool(annotations=READ)
async def notebook_logs(notebook: str) -> dict:
    """Read recent execution logs without following an endless stream."""
    return await asyncio.to_thread(run_cli, ["logs", kernel_id(notebook)])


@mcp.tool(annotations=READ)
async def notebook_files(notebook: str) -> dict:
    """List output files available from a notebook."""
    return await asyncio.to_thread(run_cli, ["files", kernel_id(notebook)])


@mcp.tool(annotations=WRITE)
async def submit_training(folder: str, run_seconds: int = 3600) -> dict:
    """Upload code and START a GPU/CPU run, consuming Kaggle quota.
    Call only when the user requested training. folder is relative to ai/kaggle.
    Metadata must explicitly mark the notebook private. No automatic retries.
    """
    if not 60 <= run_seconds <= 43200:
        raise ValueError("run_seconds must be between 60 and 43200.")
    path = under(folder, ROOT / "ai/kaggle")
    metadata = json.loads((path / "kernel-metadata.json").read_text())
    kernel_id(metadata["id"])
    if str(metadata.get("is_private", "")).lower() != "true":
        raise ValueError("Notebook metadata must explicitly set is_private to true.")
    code = under(metadata["code_file"], path)
    if not code.is_file() or code.suffix not in {".py", ".ipynb"}:
        raise ValueError("code_file must point to an existing .py or .ipynb in the notebook folder.")
    result = await asyncio.to_thread(run_cli, ["push", "-p", str(path), "-t", str(run_seconds)], 180)
    result["notebook"] = metadata["id"]
    return result


@mcp.tool(annotations=WRITE)
async def download_outputs(notebook: str, folder: str, file_pattern: str = "") -> dict:
    """Download outputs into ai/kaggle_outputs/<folder>. Does not start training.
    Use a fresh folder for each run; existing nonempty folders are refused.
    """
    notebook = kernel_id(notebook)
    destination = under(folder, OUTPUT_ROOT)
    if destination == OUTPUT_ROOT.resolve():
        raise ValueError("Choose a run-specific subfolder.")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Choose a new output folder to preserve existing results.")
    if len(file_pattern) > 300:
        raise ValueError("file_pattern is too long.")
    destination.mkdir(parents=True, exist_ok=True)
    args = ["output", notebook, "-p", str(destination)]
    if file_pattern:
        args += ["--file-pattern", file_pattern]
    result = await asyncio.to_thread(run_cli, args, 600)
    result["folder"] = str(destination.relative_to(ROOT))
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
