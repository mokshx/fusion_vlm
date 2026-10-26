"""Read individual members of a remote zip via HTTP range requests.

UR Fall stores each sequence as a zip of PNG frames (60-120 MB). To pick a few
frames we only need the central directory (at the end of the file) and the
bytes of the chosen members, so this avoids downloading whole archives from a
slow server.
"""

import io
import time
import zipfile

import requests


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str, block: int = 64 * 1024, retries: int = 8):
        self.url, self.block, self.retries = url, block, retries
        self.s = requests.Session()
        r = self._request("HEAD", {})
        if r.headers.get("Accept-Ranges") != "bytes":
            raise OSError(f"server does not support range requests: {url}")
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self._cache: dict[int, bytes] = {}  # block index -> bytes
        self.bytes_fetched = 0

    def _request(self, method: str, headers: dict) -> requests.Response:
        for attempt in range(self.retries):
            try:
                r = self.s.request(method, self.url, headers=headers, timeout=60, allow_redirects=True)
                r.raise_for_status()
                return r
            except requests.RequestException:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        raise AssertionError

    def _block(self, i: int, span: int = 1) -> bytes:
        if i not in self._cache:
            lo, hi = i * self.block, min(self.size, (i + span) * self.block) - 1
            data = self._request("GET", {"Range": f"bytes={lo}-{hi}"}).content
            if len(data) != hi - lo + 1:
                raise OSError(f"short range read {lo}-{hi}: got {len(data)}")
            for k in range(span):
                self._cache[i + k] = data[k * self.block:(k + 1) * self.block]
            self.bytes_fetched += len(data)
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
            # One request for the whole read rather than one per block.
            span = -(-(off + n) // self.block)
            chunk = self._block(i, span)[off:off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_remote_zip(url: str) -> tuple[zipfile.ZipFile, HttpRangeFile]:
    f = HttpRangeFile(url)
    return zipfile.ZipFile(f), f
