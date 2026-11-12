import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import server


class KaggleMCPTest(unittest.TestCase):
    def test_reject_invalid_kernel_and_path_escape(self):
        for value in ["--help", "user/name; touch x", "../secret", "user/name/../x"]:
            with self.assertRaises(ValueError):
                server.kernel_id(value)
        with self.assertRaises(ValueError):
            server.under("../outside", server.ROOT / "ai/kaggle")

    def test_submit_requires_private_metadata_and_contained_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "ai/kaggle/run"
            folder.mkdir(parents=True)
            metadata = {"id": "user/notebook", "is_private": False, "code_file": "train.py"}
            (folder / "train.py").write_text("print('test')")
            path = folder / "kernel-metadata.json"
            path.write_text(json.dumps(metadata))
            with patch.object(server, "ROOT", root), patch.object(server, "run_cli") as command:
                with self.assertRaises(ValueError):
                    asyncio.run(server.submit_training("run"))
                metadata.update(is_private=True, code_file="../outside.py")
                path.write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    asyncio.run(server.submit_training("run"))
                command.assert_not_called()

    def test_submit_builds_bounded_run_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "ai/kaggle/run"
            folder.mkdir(parents=True)
            (folder / "train.py").write_text("print('test')")
            (folder / "kernel-metadata.json").write_text(json.dumps(
                {"id": "user/notebook", "is_private": "true", "code_file": "train.py"}))
            with patch.object(server, "ROOT", root), patch.object(server, "run_cli", return_value={"ok": True}) as command:
                self.assertTrue(asyncio.run(server.submit_training("run", 120))["ok"])
                command.assert_called_once_with(["push", "-p", str(folder.resolve()), "-t", "120"], 180)

    def test_download_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            folder = output / "run"
            folder.mkdir()
            (folder / "model.pth").write_bytes(b"keep")
            with patch.object(server, "OUTPUT_ROOT", output), patch.object(server, "run_cli") as command:
                with self.assertRaises(ValueError):
                    asyncio.run(server.download_outputs("user/notebook", "run"))
                command.assert_not_called()

    def test_redact_environment_tokens(self):
        with patch.dict("os.environ", {"KAGGLE_API_TOKEN": "test-secret-value"}):
            self.assertEqual(server.redact("token=test-secret-value"), "token=[REDACTED]")


if __name__ == "__main__":
    unittest.main()
