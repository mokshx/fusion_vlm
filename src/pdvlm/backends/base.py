"""Common interface every model/runtime wrapper implements."""

import io
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

from PIL import Image

from pdvlm.parse import parse_yes_no


@dataclass
class Prediction:
    raw_text: str
    answer: str  # yes | no | unparsed
    parse_mode: str  # strict | lenient | none
    p_yes: float | None  # from first-token logprobs, None if unavailable
    ttft_ms: float | None = None  # client-side time to first token
    e2e_ms: float | None = None  # client-side request -> last token
    prompt_tokens: int | None = None  # includes image tokens
    gen_tokens: int | None = None
    prompt_ms: float | None = None  # server-side prefill incl. image encode
    decode_ms: float | None = None
    decode_tps: float | None = None
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        row = asdict(self)
        row.update({f"x_{k}": v for k, v in row.pop("extra").items()})
        return row


def resize_longest(img: Image.Image, longest: int | None) -> Image.Image:
    """Downscale so the longest side is `longest` px; None means native.

    Never upscales: a 256 px setting on a 640x480 frame is a real reduction in
    visual detail, whereas upscaling would add tokens without adding information.
    """
    img = img.convert("RGB")
    if longest is None or max(img.size) <= longest:
        return img
    s = longest / max(img.size)
    return img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.LANCZOS)


def to_png_bytes(img: Image.Image) -> bytes:
    # PNG (lossless) so resolution is the only variable, not JPEG artefacts.
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class Backend(ABC):
    name: str

    @abstractmethod
    def generate(self, images: list[Image.Image], prompt: str) -> Prediction: ...

    def predict(self, images: list[Image.Image], prompt: str) -> Prediction:
        pred = self.generate(images, prompt)
        pred.answer, pred.parse_mode = parse_yes_no(pred.raw_text)
        return pred

    def info(self) -> dict:
        """Static config recorded alongside every result row."""
        return {"backend": self.name}

    def close(self) -> dict:
        """Release resources; may return final measurements (e.g. peak RAM)."""
        return {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
