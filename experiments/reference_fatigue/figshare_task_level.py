"""Selective, checksum-verified access to the public RF knee-extension archive."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import struct
import time
from urllib.error import HTTPError, URLError
import zipfile

import numpy as np
from scipy import ndimage
from urllib.request import Request, urlopen


DATASET_DOI = "10.6084/m9.figshare.15273906.v1"
ARTICLE_URL = "https://journals.physiology.org/doi/10.1152/jn.00453.2021"
FIGSHARE_URL = (
    "https://figshare.com/articles/dataset/"
    "Less_common_synaptic_input_between_muscles_from_the_same_group_"
    "allows_for_more_flexible_coordination_strategies_during_a_fatiguing_task/15273906"
)
ARCHIVE_URL = "https://ndownloader.figshare.com/files/30987409"
ARCHIVE_SIZE = 10_607_794_441
ARCHIVE_MD5 = "96c92019747dea13c668c7de6e453d75"

_RF_MEMBER = re.compile(
    r"^Knee extension/(S\d+)/([^/]+)_(30|TLim)_RF_edited_final\.mat$",
    re.IGNORECASE,
)


def select_paired_rf_members(infos):
    """Return only subjects with both RF 30 and TLim recordings."""
    candidates: dict[str, dict[str, zipfile.ZipInfo]] = {}
    for info in infos:
        match = _RF_MEMBER.match(info.filename)
        if not match or info.is_dir():
            continue
        subject, _, condition = match.groups()
        condition_key = "30" if condition.lower() == "30" else "tlim"
        subject_rows = candidates.setdefault(subject, {})
        if condition_key in subject_rows:
            raise ValueError(f"duplicate RF condition member for {subject}: {condition_key}")
        subject_rows[condition_key] = info
    return {
        subject: dict(sorted(candidates[subject].items()))
        for subject in sorted(candidates)
        if set(candidates[subject]) == {"30", "tlim"}
    }


def detect_force_bouts(
    reference_signal,
    sampling_rate_hz: float,
    *,
    minimum_bout_s: float = 10.0,
    merge_gap_s: float = 0.25,
    smooth_s: float = 0.10,
    threshold_fraction: float = 0.20,
):
    """Find active contraction intervals from the aligned, uncalibrated force trace.

    The trace is used only to segment contractions and exclude rest/post-task MVC
    portions. It is never used as a model feature or to assign window labels.
    Returned intervals are start-inclusive/end-exclusive sample indices.
    """
    values = np.asarray(reference_signal, dtype=float)
    fs = float(sampling_rate_hz)
    if values.ndim != 1 or values.size < 8 or not np.isfinite(values).all():
        raise ValueError("reference_signal must be a finite one-dimensional trace")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    if not 0 < threshold_fraction < 1:
        raise ValueError("threshold_fraction must be between zero and one")
    smooth_n = max(1, int(round(smooth_s * fs)))
    if smooth_n % 2 == 0:
        smooth_n += 1
    smoothed = ndimage.median_filter(values, size=smooth_n, mode="nearest")
    lower, upper = np.percentile(smoothed, [5.0, 95.0])
    if not np.isfinite(lower + upper) or upper <= lower:
        raise ValueError("reference_signal has insufficient dynamic range")
    threshold = float(lower + threshold_fraction * (upper - lower))
    active = smoothed > threshold
    merge_n = max(1, int(round(merge_gap_s * fs)))
    if merge_n > 1:
        active = ndimage.binary_closing(active, structure=np.ones(merge_n, dtype=bool))
    changes = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    minimum_n = int(round(minimum_bout_s * fs))
    bouts = [(int(start), int(end)) for start, end in zip(starts, ends) if end - start >= minimum_n]
    return bouts, threshold


class HttpRangeFile(io.RawIOBase):
    """Seekable HTTP Range reader used to extract members without the 10.6 GB ZIP."""

    def __init__(self, url: str, size: int, *, block_size: int = 8 * 1024 * 1024, cache_blocks: int = 3, max_retries: int = 5):
        super().__init__()
        if size < 1 or block_size < 1 or cache_blocks < 1 or max_retries < 1:
            raise ValueError("size, block_size, and cache_blocks must be positive")
        self.url = url
        self.size = int(size)
        self.block_size = int(block_size)
        self.cache_blocks = int(cache_blocks)
        self.max_retries = int(max_retries)
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.range_requests = 0
        self.bytes_received = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError("invalid seek mode")
        if target < 0:
            raise ValueError("cannot seek before the beginning of the remote file")
        self.position = int(target)
        return self.position

    def _load_block(self, block_number: int) -> bytes:
        if block_number in self.cache:
            self.cache.move_to_end(block_number)
            return self.cache[block_number]
        start = block_number * self.block_size
        end = min(self.size - 1, start + self.block_size - 1)
        request = Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        last_error = None
        for attempt in range(self.max_retries):
            try:
                with urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise OSError(f"expected HTTP 206 for byte range, received {response.status}")
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {start}-{end}/{self.size}"):
                        raise OSError(f"unexpected Content-Range header: {content_range!r}")
                    data = response.read(end - start + 1)
                break
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                last_error = error
                retryable = not isinstance(error, HTTPError) or error.code in {403, 408, 425, 429, 500, 502, 503, 504}
                if not retryable or attempt + 1 >= self.max_retries:
                    raise
                time.sleep(min(4.0, 0.25 * (2**attempt)))
        else:
            raise OSError(f"range retrieval failed after retries: {last_error}")
        if len(data) != end - start + 1:
            raise OSError(f"short HTTP range: expected {end - start + 1}, received {len(data)}")
        self.range_requests += 1
        self.bytes_received += len(data)
        self.cache[block_number] = data
        while len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, size=-1):
        if self.closed:
            raise ValueError("I/O operation on closed remote file")
        if size is None or size < 0:
            size = max(0, self.size - self.position)
        size = min(int(size), max(0, self.size - self.position))
        result = bytearray()
        while size:
            block_number = self.position // self.block_size
            block_offset = self.position % self.block_size
            block = self._load_block(block_number)
            count = min(size, len(block) - block_offset)
            result.extend(block[block_offset : block_offset + count])
            self.position += count
            size -= count
        return bytes(result)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _crc32_file(path: Path) -> int:
    import zlib

    crc = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            crc = zlib.crc32(block, crc)
    return crc & 0xFFFFFFFF


def _extract_one_subject(subject, members, output_dir, url, archive_size):
    remote = HttpRangeFile(url, archive_size)
    receipts = []
    with zipfile.ZipFile(remote) as archive:
        lookup = {info.filename: info for info in archive.infolist()}
        for condition, selected_info in members.items():
            info = lookup[selected_info.filename]
            destination = output_dir / subject / Path(info.filename).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file():
                size_ok = destination.stat().st_size == info.file_size
                crc_ok = _crc32_file(destination) == info.CRC
                if not (size_ok and crc_ok):
                    raise ValueError(f"existing extracted member failed size/CRC validation: {destination}")
                sha256 = _hash_file(destination)
                reused = True
            else:
                temporary = destination.with_suffix(destination.suffix + ".part")
                digest = hashlib.sha256()
                written = 0
                with archive.open(info, "r") as source, temporary.open("wb") as target:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        target.write(block)
                        digest.update(block)
                        written += len(block)
                if written != info.file_size or _crc32_file(temporary) != info.CRC:
                    temporary.unlink(missing_ok=True)
                    raise ValueError(f"extracted member failed size/CRC validation: {info.filename}")
                os.replace(temporary, destination)
                sha256 = digest.hexdigest()
                reused = False
            receipts.append({
                "subject_id": subject,
                "condition": condition,
                "archive_member": info.filename,
                "compressed_bytes": int(info.compress_size),
                "uncompressed_bytes": int(info.file_size),
                "zip_crc32": f"{info.CRC:08x}",
                "member_crc32_verified": True,
                "sha256": sha256,
                "local_path": str(destination.resolve()),
                "reused_local_verified_copy": reused,
            })
    return {"subject_id": subject, "members": receipts, "range_requests": remote.range_requests, "bytes_received": remote.bytes_received}


def extract_paired_rf_members(output_dir, *, url=ARCHIVE_URL, archive_size=ARCHIVE_SIZE, workers=2):
    """Range-extract paired RF files only and write source/member receipts."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_reader = HttpRangeFile(url, archive_size)
    with zipfile.ZipFile(catalog_reader) as catalog:
        pairs = select_paired_rf_members(catalog.infolist())
    if len(pairs) < 4:
        raise ValueError("fewer than four complete participant RF pairs were found")
    summaries = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(_extract_one_subject, subject, members, output_dir, url, archive_size): subject
            for subject, members in pairs.items()
        }
        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary)
            print(f"extracted participant {summary['subject_id']} ({len(summary['members'])} RF trials)", flush=True)
    member_rows = [row for item in summaries for row in item["members"]]
    receipt = {
        "dataset_title": "Less common synaptic input between muscles from the same group allows for more flexible coordination strategies during a fatiguing task",
        "dataset_doi": DATASET_DOI,
        "dataset_url": FIGSHARE_URL,
        "article_url": ARTICLE_URL,
        "license": "CC BY 4.0",
        "figshare_file_id": 30987409,
        "archive_url": url,
        "archive_bytes": int(archive_size),
        "archive_md5_official": ARCHIVE_MD5,
        "full_archive_md5_verified": False,
        "download_method": "HTTP Range retrieval of the complete archive central directory and only paired RF MAT members; original 10.6 GB ZIP not downloaded",
        "complete_pair_count": len(pairs),
        "extracted_member_count": len(member_rows),
        "member_crc32_verified": all(row["member_crc32_verified"] for row in member_rows),
        "range_requests": int(catalog_reader.range_requests + sum(item["range_requests"] for item in summaries)),
        "bytes_received": int(catalog_reader.bytes_received + sum(item["bytes_received"] for item in summaries)),
        "members": sorted(member_rows, key=lambda row: (row["subject_id"], row["condition"])),
    }
    receipt_path = output_dir / "source_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt
