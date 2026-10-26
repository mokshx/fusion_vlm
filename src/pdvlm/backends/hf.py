"""Hugging Face transformers backend: the unquantised (bf16/fp16) reference.
Used for accuracy runs on Kaggle/Colab GPUs; latency numbers from here are
GPU-server numbers and are NOT the paper's edge-efficiency results."""

import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from pdvlm.backends.base import Backend, Prediction


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFBackend(Backend):
    name = "hf"

    def __init__(self, model_id: str, dtype: str = "bfloat16", max_tokens: int = 16, device: str | None = None):
        self.model_id, self.max_tokens = model_id, max_tokens
        self.device = device or _device()
        # T4 (Kaggle/Colab free tier) has no bf16 support -> fall back to fp16.
        if dtype == "bfloat16" and self.device == "cuda" and not torch.cuda.is_bf16_supported():
            dtype = "float16"
        self.dtype = dtype
        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=getattr(torch, dtype)
        ).to(self.device).eval()
        self.load_s = time.perf_counter() - t0
        tok = self.processor.tokenizer
        self._yes_ids = {i for w in ("yes", "Yes", " yes", " Yes", "YES") for i in tok.encode(w, add_special_tokens=False)[:1]}
        self._no_ids = {i for w in ("no", "No", " no", " No", "NO") for i in tok.encode(w, add_special_tokens=False)[:1]}

    def info(self) -> dict:
        return {"backend": self.name, "model_id": self.model_id, "dtype": self.dtype,
                "device": self.device, "load_s": self.load_s,
                "gpu": torch.cuda.get_device_name() if self.device == "cuda" else None}

    @torch.inference_mode()
    def generate(self, images: list[Image.Image], prompt: str) -> Prediction:
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(self.device)
        n_prompt = inputs["input_ids"].shape[1]
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=self.max_tokens,
            output_scores=True, return_dict_in_generate=True,
        )
        if self.device == "cuda":
            torch.cuda.synchronize()
        e2e = (time.perf_counter() - t0) * 1000
        gen = out.sequences[0, n_prompt:]
        probs = out.scores[0][0].float().softmax(-1)
        py, pn = probs[list(self._yes_ids)].sum().item(), probs[list(self._no_ids)].sum().item()
        return Prediction(
            raw_text=self.processor.batch_decode(gen[None], skip_special_tokens=True)[0],
            answer="", parse_mode="",
            p_yes=py / (py + pn) if py + pn > 0 else None,
            e2e_ms=e2e, prompt_tokens=n_prompt, gen_tokens=len(gen),
        )

    def close(self) -> dict:
        if self.device == "cuda":
            return {"peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20}
        return {}
