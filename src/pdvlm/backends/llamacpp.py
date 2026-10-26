"""llama.cpp backend: runs `llama-server` as a subprocess and talks to its
OpenAI-compatible endpoint. Used for every GGUF quantisation and for all
efficiency measurements.

Peak RAM: the server is launched under `/usr/bin/time` so the OS reports the
lifetime peak (load + warm-up + all requests) when it exits. On macOS we read
both "maximum resident set size" and "peak memory footprint" (the latter is
what Activity Monitor shows and includes Metal/unified-memory allocations).
On Linux we read GNU time's "Maximum resident set size".
"""

import base64
import json
import platform
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

import psutil
import requests
from PIL import Image

from pdvlm.backends.base import Backend, Prediction, to_png_bytes
from pdvlm.parse import p_yes_from_top_logprobs


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _parse_time_report(text: str) -> dict:
    out = {}
    if platform.system() == "Darwin":
        m = re.search(r"(\d+)\s+maximum resident set size", text)
        if m:
            out["peak_rss_mb"] = int(m.group(1)) / 2**20  # bytes on macOS
        m = re.search(r"(\d+)\s+peak memory footprint", text)
        if m:
            out["peak_footprint_mb"] = int(m.group(1)) / 2**20
    else:
        m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
        if m:
            out["peak_rss_mb"] = int(m.group(1)) / 1024  # kB on GNU time
    return out


class SystemMemorySampler:
    """Peak system-wide memory in use (total - available) relative to an idle baseline.

    Process RSS double-counts file-backed pages and misses GPU buffers on
    unified-memory devices; the system-level increase is what decides whether a
    model fits next to everything else on the device. Needs a quiet machine.
    """

    def __init__(self, interval_s: float = 0.1, baseline_s: float = 1.0):
        self.interval_s = interval_s
        samples = []
        t_end = time.perf_counter() + baseline_s
        while time.perf_counter() < t_end:
            samples.append(self._in_use())
            time.sleep(interval_s)
        self.baseline = sorted(samples)[len(samples) // 2]
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _in_use() -> int:
        vm = psutil.virtual_memory()
        return vm.total - vm.available

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._in_use())
            self._stop.wait(self.interval_s)

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        return {"sys_mem_delta_mb": (self.peak - self.baseline) / 2**20,
                "sys_mem_baseline_mb": self.baseline / 2**20}


class LlamaCppBackend(Backend):
    name = "llamacpp"

    def __init__(
        self,
        model: str | Path,
        mmproj: str | Path,
        n_gpu_layers: int = 0,
        threads: int | None = None,
        ctx_size: int = 4096,
        max_tokens: int = 16,
        mmproj_offload: bool | None = None,
        image_max_tokens: int | None = None,
        reasoning_off: bool = False,
        server_bin: str = "llama-server",
        extra_args: list[str] | None = None,
        log_path: str | Path | None = None,
        startup_timeout_s: float = 300,
    ):
        self.model, self.mmproj = Path(model), Path(mmproj)
        self.n_gpu_layers, self.threads, self.ctx_size = n_gpu_layers, threads, ctx_size
        self.max_tokens = max_tokens
        # Default: projector follows the LM (CPU-only run keeps everything on CPU).
        self.mmproj_offload = n_gpu_layers > 0 if mmproj_offload is None else mmproj_offload
        self.image_max_tokens = image_max_tokens
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log_path = Path(log_path or f"/tmp/llama-server-{self.port}.log")

        cmd = [
            server_bin, "-m", str(self.model), "--mmproj", str(self.mmproj),
            "--host", "127.0.0.1", "--port", str(self.port),
            "-ngl", str(n_gpu_layers), "-c", str(ctx_size),
            "-np", "1",  # one slot: one decision at a time, no extra KV allocation
            "-cram", "0",  # no host-side prompt cache: every request is a cold decision
            "--no-webui",
            "--mmproj-offload" if self.mmproj_offload else "--no-mmproj-offload",
        ]
        if n_gpu_layers == 0:
            cmd += ["-dev", "none"]  # truly CPU-only, no Metal/CUDA buffers at all
        if threads:
            cmd += ["-t", str(threads)]
        if image_max_tokens:
            cmd += ["--image-max-tokens", str(image_max_tokens)]
        if reasoning_off:
            cmd += ["-rea", "off"]
        self.extra_args = list(extra_args or [])  # e.g. ["--no-mmap", "--no-repack"]
        cmd += self.extra_args
        self.cmd = cmd

        self._mem = SystemMemorySampler()

        time_flag = "-l" if platform.system() == "Darwin" else "-v"
        self._log = open(self.log_path, "w")
        self._proc = subprocess.Popen(
            ["/usr/bin/time", time_flag, *cmd], stdout=self._log, stderr=subprocess.STDOUT
        )
        self.load_s = self._wait_ready(startup_timeout_s)

    def _wait_ready(self, timeout: float) -> float:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early; see {self.log_path}")
            try:
                if requests.get(f"{self.url}/health", timeout=1).status_code == 200:
                    return time.perf_counter() - t0
            except requests.ConnectionError:
                pass
            time.sleep(0.2)
        raise TimeoutError(f"llama-server not ready after {timeout}s; see {self.log_path}")

    def server_pid(self) -> int | None:
        kids = psutil.Process(self._proc.pid).children()
        return kids[0].pid if kids else None

    def info(self) -> dict:
        return {
            "backend": self.name,
            "model_file": self.model.name,
            "mmproj_file": self.mmproj.name,
            "model_size_mb": self.model.stat().st_size / 2**20,
            "mmproj_size_mb": self.mmproj.stat().st_size / 2**20,
            "n_gpu_layers": self.n_gpu_layers,
            "mmproj_offload": self.mmproj_offload,
            "threads": self.threads,
            "ctx_size": self.ctx_size,
            "image_max_tokens": self.image_max_tokens,
            "load_s": self.load_s,
            "extra_args": " ".join(self.extra_args),
            "server_cmd": " ".join(self.cmd),
        }

    def generate(self, images: list[Image.Image], prompt: str) -> Prediction:
        content = [
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + base64.b64encode(to_png_bytes(im)).decode()}}
            for im in images
        ]
        content.append({"type": "text", "text": prompt})
        body = {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,  # greedy: deterministic, repeatable benchmark
            "cache_prompt": False,
            "logprobs": True,
            "top_logprobs": 10,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        text, first_top, timings, ttft = [], None, {}, None
        t0 = time.perf_counter()
        with requests.post(f"{self.url}/v1/chat/completions", json=body, stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                timings = chunk.get("timings", timings)
                for ch in chunk.get("choices", []):
                    piece = ch.get("delta", {}).get("content")
                    if piece:
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        text.append(piece)
                    lp = (ch.get("logprobs") or {}).get("content") or []
                    if lp and first_top is None:
                        first_top = {t["token"]: t["logprob"] for t in lp[0].get("top_logprobs", [])}
        e2e = (time.perf_counter() - t0) * 1000
        return Prediction(
            raw_text="".join(text),
            answer="", parse_mode="",
            p_yes=p_yes_from_top_logprobs(first_top) if first_top else None,
            ttft_ms=ttft, e2e_ms=e2e,
            prompt_tokens=timings.get("prompt_n"),
            gen_tokens=timings.get("predicted_n"),
            prompt_ms=timings.get("prompt_ms"),
            decode_ms=timings.get("predicted_ms"),
            decode_tps=timings.get("predicted_per_second"),
        )

    def close(self) -> dict:
        if getattr(self, "_closed", None) is not None:
            return self._closed
        mem = self._mem.stop()  # before shutdown, so the peak reflects the running server
        if self._proc.poll() is None:
            pid = self.server_pid()
            if pid:
                psutil.Process(pid).terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._log.close()
        self._closed = {**_parse_time_report(self.log_path.read_text(errors="replace")), **mem}
        return self._closed
