"""Model registry: short name -> HF repo / GGUF files.

Spec format: "<model>:<quant>", e.g. "lfm-450m:Q8_0", "lfm-3b:F16", "lfm-3b:hf".
The vision projector (mmproj) is always kept at F16 so that only the language
model's quantisation varies between configurations.
"""

from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path("models")


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    gguf_repo: str | None
    gguf_pattern: str | None  # format with quant
    mmproj: str | None
    reasoning_off: bool = False  # Qwen3.5 thinks by default; we want a direct answer


REGISTRY = {
    "lfm-450m": ModelSpec("LiquidAI/LFM2.5-VL-450M", "LiquidAI/LFM2.5-VL-450M-GGUF",
                          "LFM2.5-VL-450M-{q}.gguf", "mmproj-LFM2.5-VL-450m-F16.gguf"),
    "lfm-1.6b": ModelSpec("LiquidAI/LFM2.5-VL-1.6B", "LiquidAI/LFM2.5-VL-1.6B-GGUF",
                          "LFM2.5-VL-1.6B-{q}.gguf", "mmproj-LFM2.5-VL-1.6B-F16.gguf"),
    "lfm-3b": ModelSpec("LiquidAI/LFM2.5-VL-3B", "LiquidAI/LFM2.5-VL-3B-GGUF",
                        "LFM2.5-VL-3B-{q}.gguf", "mmproj-LFM2.5-VL-3B-F16.gguf"),
    # Qwen publishes no official GGUF; these are Unsloth's community conversions
    # (no F16 LM file there, only BF16, Q8_0, Q4_K_M).
    "qwen3.5-2b": ModelSpec("Qwen/Qwen3.5-2B", "unsloth/Qwen3.5-2B-GGUF",
                            "Qwen3.5-2B-{q}.gguf", "mmproj-F16.gguf", reasoning_off=True),
    "qwen3.5-4b": ModelSpec("Qwen/Qwen3.5-4B", "unsloth/Qwen3.5-4B-GGUF",
                            "Qwen3.5-4B-{q}.gguf", "mmproj-F16.gguf", reasoning_off=True),
}


def resolve(spec: str) -> tuple[ModelSpec, str]:
    name, _, quant = spec.partition(":")
    return REGISTRY[name], quant or "hf"


def gguf_paths(spec: str) -> tuple[Path, Path]:
    m, q = resolve(spec)
    d = MODELS_DIR / m.gguf_repo.split("/")[1]
    return d / m.gguf_pattern.format(q=q), d / m.mmproj


def make_backend(spec: str, **kw):
    m, q = resolve(spec)
    if q == "hf":
        from pdvlm.backends.hf import HFBackend
        return HFBackend(m.hf_id, **kw)
    from pdvlm.backends.llamacpp import LlamaCppBackend
    model, mmproj = gguf_paths(spec)
    return LlamaCppBackend(model, mmproj, reasoning_off=m.reasoning_off, **kw)
