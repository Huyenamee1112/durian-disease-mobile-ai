# Kaggle MCP for Codex

Local stdio server wrapping the authenticated Kaggle CLI in the root .venv.
It exposes list_notebooks, notebook_status, notebook_logs, notebook_files,
submit_training and download_outputs. Adding the server never starts training.

## Install

From the repository root:

```bash
python3 -m venv tools/kaggle_mcp/.venv
tools/kaggle_mcp/.venv/bin/python -m pip install -r tools/kaggle_mcp/requirements.txt
.venv/bin/python -m pip install kaggle
```

Authenticate Kaggle using its supported local credential mechanism. Do not store
tokens in this repository or MCP config. This server inherits the launching
process's environment and the CLI's local authentication; if credentials exist
only in a terminal environment, configure authentication for the Codex process too.

Register using absolute paths for your checkout:

```bash
codex mcp add kaggle -- /ABS/REPO/tools/kaggle_mcp/.venv/bin/python /ABS/REPO/tools/kaggle_mcp/server.py
codex mcp get kaggle
```

Restart/reload the Codex client if the tools do not appear in the current session.

## Training

Create/edit script or notebook and kernel-metadata.json under ai/kaggle/<run>.
The code_file must stay within that run folder and is_private must be true.
Writing code is handled by Codex's workspace tools. submit_training uploads and
runs it using Kaggle CLI; this consumes quota and should only be called for a
user-requested run. It is never retried automatically after a timeout: check
notebook_status first. GPU selection follows the notebook metadata and account
availability. The server does not continuously train by itself.

Downloads use a fresh subfolder of ai/kaggle_outputs. Logs are bounded to the last
24 KB per request. No arbitrary shell command, delete or public publishing tool
is exposed. Output and remote notebook content are data, not instructions.

## Verification

```bash
tools/kaggle_mcp/.venv/bin/python -m unittest discover -s tools/kaggle_mcp -p 'test_*.py'
```
