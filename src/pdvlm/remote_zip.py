"""Read individual members of a remote zip via HTTP range requests.

UR Fall stores each sequence as a zip of PNG frames (60-120 MB) and Le2i ships
one 9 GB zip. We only need the central directory (at the end of the file) and
the bytes of chosen members, so this avoids downloading whole archives from
slow servers.

`url` may be a callable that returns a fresh URL, for pre-signed links that
expire (Le2i on dat@UBFC): on HTTP 403 the URL is refreshed and the request
retried.
"""

import io
import time
import zipfile
from collections import OrderedDict
from collections.abc import Callable

import requests

MAX_REQUEST_BLOCKS = 64  # cap one range request at 4 MiB so a failed request is cheap to retry


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str | Callable[[], str], block: int = 64 * 1024,
                 retries: int = 8, max_cached_blocks: int = 512):
        self._url_factory = url if callable(url) else None
        self.url = url() if callable(url) else url
        self.block, self.retries, self.max_cached_blocks = block, retries, max_cached_blocks
        self.s = requests.Session()
        # 1-byte ranged GET instead of HEAD: pre-signed S3 URLs are often GET-only.
        r = self._request("GET", {"Range": "bytes=0-0"})
        if r.status_code != 206 or "Content-Range" not in r.headers:
            raise OSError(f"server does not support range requests: {self.url}")
        self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        self.pos = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()  # bounded LRU: block index -> bytes
        self.bytes_fetched = 0

    def _request(self, method: str, headers: dict) -> requests.Response:
        for attempt in range(self.retries):
            try:
                r = self.s.request(method, self.url, headers=headers, timeout=120, allow_redirects=True)
                if r.status_code == 403 and self._url_factory:
                    self.url = self._url_factory()  # pre-signed URL expired
                    continue
                r.raise_for_status()
                return r
            except requests.RequestException:
                if attempt == self.retries - 1:
                    raise
                time.sleep(min(60, 2 ** attempt))
        raise OSError(f"giving up on {method} after {self.retries} attempts")

    def _block(self, i: int, span: int = 1) -> bytes:
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        span = max(1, min(span, MAX_REQUEST_BLOCKS))
        lo, hi = i * self.block, min(self.size, (i + span) * self.block) - 1
        data = self._request("GET", {"Range": f"bytes={lo}-{hi}"}).content
        if len(data) != hi - lo + 1:
            raise OSError(f"short range read {lo}-{hi}: got {len(data)}")
        for k in range(span):
            piece = data[k * self.block:(k + 1) * self.block]
            if piece:
                self._cache[i + k] = piece
        self.bytes_fetched += len(data)
        while len(self._cache) > self.max_cached_blocks:
            self._cache.popitem(last=False)
        return self._cache[i]

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            i, off = divmod(self.pos, self.block)
            span = -(-(off + n) // self.block)  # one request for the whole read (capped)
            chunk = self._block(i, span)[off:off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_remote_zip(url: str | Callable[[], str]) -> tuple[zipfile.ZipFile, HttpRangeFile]:
    f = HttpRangeFile(url)
    return zipfile.ZipFile(f), f


def extract_member(z: zipfile.ZipFile, name: str, dest, chunk: int = 4 * 2**20) -> int:
    """Stream one member to `dest` in chunks (each chunk is a separate, retried
    range request). Writes to a .part file and renames on success."""
    from pathlib import Path
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    n = 0
    with z.open(name) as src, open(tmp, "wb") as out:
        while True:
            buf = src.read(chunk)
            if not buf:
                break
            out.write(buf)
            n += len(buf)
    tmp.rename(dest)
    return n
