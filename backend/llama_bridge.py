"""
LazyMoE llama.cpp Bridge - Uses llama-server HTTP API
Model stays loaded in RAM permanently. No reload per query.
"""

import os
import time
import shutil
import subprocess
import threading
import logging
import urllib.request
import urllib.error
import json
from typing import Generator, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("lazy-moe.llama")

LLAMA_SERVER_PORT = 8080


@dataclass
class InferenceConfig:
    model_path: str
    n_ctx: int = 2048
    n_predict: int = 200
    n_threads: int = 4
    n_gpu_layers: int = 0
    temp: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    mlock: bool = False
    mmap: bool = True
    verbose: bool = False


@dataclass
class TokenEvent:
    token: str
    token_id: int = 0
    timestamp: float = field(default_factory=time.time)
    active_experts: list = field(default_factory=list)


class LlamaBridge:

    LLAMA_SERVER_NAMES = ["llama-server", "llama-server.exe"]
    LLAMA_CLI_NAMES = ["llama-cli", "llama-cli.exe"]

    def __init__(self, config: InferenceConfig,
                 on_expert_activate: Optional[Callable] = None):
        self.config = config
        self.on_expert_activate = on_expert_activate
        self._server_bin = self._find_binary(self.LLAMA_SERVER_NAMES)
        self._cli_bin = self._find_binary(self.LLAMA_CLI_NAMES)
        self._server_process = None
        self._server_ready = False
        self._server_lock = threading.Lock()

        # Start server in background if available
        if self._server_bin and os.path.exists(self.config.model_path):
            self._start_server()

    def _start_server(self):
        """Start llama-server in background. Model loads once, stays in RAM."""
        def run():
            cmd = [
                self._server_bin,
                "-m", self.config.model_path,
                "-c", str(self.config.n_ctx),
                "-t", str(self.config.n_threads),
                "--port", str(LLAMA_SERVER_PORT),
                "--host", "127.0.0.1",
                "--no-warmup",
                "--log-disable",
            ]
            logger.info(f"Starting llama-server on port {LLAMA_SERVER_PORT}...")
            self._server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        # Wait for server to be ready (up to 3 minutes for model load)
        logger.info("Waiting for llama-server to load model...")
        for i in range(180):
            time.sleep(1)
            if self._ping_server():
                self._server_ready = True
                logger.info(f"llama-server ready after {i+1}s")
                return

        logger.warning("llama-server did not start in time — will use cli fallback")

    def _ping_server(self) -> bool:
        try:
            req = urllib.request.urlopen(
                f"http://127.0.0.1:{LLAMA_SERVER_PORT}/health",
                timeout=1
            )
            return req.status == 200
        except Exception:
            return False

    def stream(self, prompt: str, system_prompt: str = "") -> Generator[TokenEvent, None, None]:
        # Try server first (model already loaded = fast)
        if self._server_ready and self._ping_server():
            yield from self._stream_from_server(prompt)
            return

        # Fallback: run cli directly (slow — reloads model each time)
        if self._cli_bin and os.path.exists(self.config.model_path):
            yield from self._stream_from_cli(prompt)
            return

        # Mock mode
        logger.warning("No llama backend available — mock mode")
        yield from self._mock_stream(prompt)

    def _stream_from_server(self, prompt: str) -> Generator[TokenEvent, None, None]:
        """Fast path: query running llama-server HTTP API."""
        logger.info("Querying llama-server...")
        try:
            payload = json.dumps({
                "prompt": prompt,
                "n_predict": self.config.n_predict,
                "temperature": self.config.temp,
                "top_p": self.config.top_p,
                "repeat_penalty": self.config.repeat_penalty,
                "stream": False,
            }).encode("utf-8")

            req = urllib.request.Request(
                f"http://127.0.0.1:{LLAMA_SERVER_PORT}/completion",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data.get("content", "").strip()

            if not text:
                logger.warning("Empty response from server")
                yield from self._mock_stream(prompt)
                return

            logger.info(f"Server response: {text[:80]}")

            words = text.split(" ")
            for i, word in enumerate(words):
                if not word:
                    continue
                token = ("" if i == 0 else " ") + word
                yield TokenEvent(token=token, token_id=i, active_experts=[])
                time.sleep(0.05)

        except Exception as e:
            logger.exception(f"Server query error: {e}")
            yield from self._mock_stream(prompt)

    def _stream_from_cli(self, prompt: str) -> Generator[TokenEvent, None, None]:
        """Slow path: spawn llama-cli, wait for it to load and generate."""
        logger.info("Running llama-cli (slow — reloads model)...")
        try:
            cmd = [
                self._cli_bin,
                "-m", self.config.model_path,
                "-p", prompt,
                "-n", str(self.config.n_predict),
                "-c", str(self.config.n_ctx),
                "-t", str(self.config.n_threads),
                "--temp", str(self.config.temp),
                "--no-warmup",
                "--log-disable",
                "--mmap",
                "-e",
            ]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            # Read with timeout
            try:
                stdout, _ = proc.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
                logger.error("CLI timed out")
                yield from self._mock_stream(prompt)
                return

            raw = stdout.decode("utf-8", errors="replace")
            logger.info(f"CLI raw ({len(raw)} chars): {repr(raw[:150])}")

            # Strip prompt from start, stats from end
            text = raw
            if text.startswith(prompt):
                text = text[len(prompt):]
            for stop in ["[ Prompt:", "[Prompt:", "llama_print"]:
                idx = text.find(stop)
                if idx != -1:
                    text = text[:idx]
            text = text.strip()

            if not text:
                logger.warning("Empty CLI output")
                yield from self._mock_stream(prompt)
                return

            logger.info(f"CLI extracted: {text[:80]}")
            for i, word in enumerate(text.split(" ")):
                if not word:
                    continue
                token = ("" if i == 0 else " ") + word
                yield TokenEvent(token=token, token_id=i, active_experts=[])
                time.sleep(0.05)

        except Exception as e:
            logger.exception(f"CLI error: {e}")
            yield from self._mock_stream(prompt)

    def stop(self):
        if self._server_process and self._server_process.poll() is None:
            self._server_process.terminate()

    def is_available(self) -> bool:
        has_binary = bool(self._server_bin or self._cli_bin)
        has_model = os.path.exists(self.config.model_path)
        return has_binary and has_model

    def _mock_stream(self, prompt: str) -> Generator[TokenEvent, None, None]:
        import random
        logger.warning("MOCK mode")
        responses = {
            "code":    "Here is a Python implementation using dynamic programming. We initialize a table and fill it bottom-up for O(n squared) time complexity.",
            "math":    "Solving step by step: isolate the variable, apply the quadratic formula. The roots are x equals 2 and x equals negative 3.",
            "default": "The capital of India is New Delhi. It is a major political and cultural hub located in the northern part of the country.",
        }
        key = "code" if any(w in prompt.lower() for w in ["code","python","function"]) else \
              "math" if any(w in prompt.lower() for w in ["solve","math","equation"]) else "default"
        text = responses[key]
        for i, word in enumerate(text.split()):
            token = ("" if i == 0 else " ") + word
            yield TokenEvent(token=token, token_id=i, active_experts=random.sample(range(8), 2))
            time.sleep(0.06)

    def _find_binary(self, names: list) -> Optional[str]:
        for name in names:
            path = shutil.which(name)
            if path:
                logger.info(f"Found binary: {path}")
                return path
        common_dirs = [
            os.path.join(os.path.expanduser("~"), "lazy-moe", "llama.cpp"),
        ]
        for d in common_dirs:
            for name in names:
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    logger.info(f"Found binary: {p}")
                    return p
        return None

    def _cleanup(self):
        pass
