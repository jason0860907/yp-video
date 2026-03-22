"""vLLM process lifecycle manager."""

import asyncio
import os
import subprocess
from pathlib import Path

import aiohttp


def _load_vllm_env() -> dict[str, str]:
    """Load config from vllm.env."""
    env_path = Path(__file__).parent.parent / "vllm.env"
    config = {}
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    config[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return config


class VLLMManager:
    """Manage vLLM server process lifecycle."""

    def __init__(self):
        self.config = _load_vllm_env()
        self.process: asyncio.subprocess.Process | None = None
        self._health_task: asyncio.Task | None = None
        self._status = "stopped"  # stopped, starting, running, error
        self._model = self.config.get("VLLM_MODEL", "")
        self._port = int(self.config.get("VLLM_PORT", "8000"))

    @property
    def status(self) -> str:
        return self._status

    @property
    def model(self) -> str:
        return self._model

    @property
    def port(self) -> int:
        return self._port

    @property
    def server_url(self) -> str:
        return f"http://localhost:{self._port}"

    def get_status_dict(self) -> dict:
        return {
            "status": self._status,
            "model": self._model,
            "port": self._port,
            "pid": self.process.pid if self.process else None,
        }

    async def check_health(self) -> bool:
        """Ping vLLM /v1/models endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.server_url}/v1/models", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def _detect_existing(self) -> bool:
        """Check if vLLM is already running (e.g., from tmux)."""
        if await self.check_health():
            self._status = "running"
            return True
        return False

    async def start(self) -> dict:
        """Start vLLM server process."""
        if self._status == "running":
            return {"ok": True, "message": "Already running"}

        # Check if already running externally
        if await self._detect_existing():
            from app.jobs import job_manager
            job_manager.vllm_using_gpu = True
            return {"ok": True, "message": "Detected existing vLLM server"}

        self._status = "starting"

        project_root = Path(__file__).parent.parent
        venv_python = project_root / ".venv" / "bin" / "python"

        cmd = [
            str(venv_python), "-m", "vllm.entrypoints.openai.api_server",
            "--model", self._model,
            "--port", str(self._port),
            "--max-model-len", self.config.get("VLLM_MAX_MODEL_LEN", "32768"),
            "--gpu-memory-utilization", self.config.get("VLLM_GPU_MEMORY_UTILIZATION", "0.95"),
            "--max-num-seqs", self.config.get("VLLM_MAX_NUM_SEQS", "16"),
            "--enable-prefix-caching",
            "--reasoning-parser", "qwen3",
            "--no-enable-log-requests",
            "--allowed-local-media-path", str(Path.home()),
        ]

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(project_root),
            )
        except Exception as e:
            self._status = "error"
            return {"ok": False, "message": str(e)}

        # Start health check loop
        self._health_task = asyncio.create_task(self._health_loop())

        from app.jobs import job_manager
        job_manager.vllm_using_gpu = True

        return {"ok": True, "message": "Starting vLLM server..."}

    async def _health_loop(self):
        """Periodically check vLLM health."""
        # Wait for startup
        for _ in range(120):  # up to 10 minutes
            await asyncio.sleep(5)
            if await self.check_health():
                self._status = "running"
                return
            if self.process and self.process.returncode is not None:
                self._status = "error"
                return
        self._status = "error"

    async def stop(self) -> dict:
        """Stop vLLM server."""
        # Try to kill our managed process
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10)
            except asyncio.TimeoutError:
                self.process.kill()
            self.process = None

        # Also try to kill tmux session
        try:
            subprocess.run(["tmux", "kill-session", "-t", "vllm"],
                           capture_output=True, timeout=5)
        except Exception:
            pass

        if self._health_task:
            self._health_task.cancel()
            self._health_task = None

        self._status = "stopped"

        from app.jobs import job_manager
        job_manager.vllm_using_gpu = False

        return {"ok": True, "message": "vLLM server stopped"}

    async def initial_check(self):
        """Run on startup to detect existing vLLM instances."""
        if await self._detect_existing():
            from app.jobs import job_manager
            job_manager.vllm_using_gpu = True


# Singleton
vllm_manager = VLLMManager()
