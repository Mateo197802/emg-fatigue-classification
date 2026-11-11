"""Fetch public EMG dataset metadata and explicitly selected files.

Files are kept in a private run directory outside the repository. Large signal
archives are opt-in, resumable, publisher-checksum verified, and never extracted
unless the caller requests extraction of an archive below the safety limit.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import requests


CERQUEIRA_ID = "14182446"
SWIFTIES_ID = "15172815"
FIGSHARE_ID = "15273906"
LARGE_ARCHIVE_BYTES = 1_000_000_000
RANGE_CHUNK_BYTES = 4 * 1024 * 1024
SMALL_AUTO_EXTRACT_LIMIT_BYTES = 100_000_000
USER_AGENT = "emg-fatigue-classification-research/1.0 (reproducible public dataset audit)"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _http_json(url: str) -> dict[str, Any]:
    response = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}, timeout=(15, 45))
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object from {url}")
    return value


def _safe_basename(value: str) -> str:
    name = Path(str(value)).name
    if name in {"", ".", ".."} or name != str(value).replace("\\", "/").split("/")[-1]:
        raise ValueError(f"unsafe remote filename: {value!r}")
    return name


def _checksum_digest(file_path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with file_path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_publisher_checksum(raw: str | None) -> tuple[str, str] | None:
    if not raw:
        return None
    if ":" in raw:
        algorithm, digest = raw.split(":", 1)
    else:
        algorithm, digest = "md5", raw
    algorithm = algorithm.lower()
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"unsupported publisher checksum algorithm: {algorithm}")
    return algorithm, digest.lower()


def _disk_preflight(destination: Path, size_bytes: int, *, extract: bool) -> dict[str, int]:
    destination.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(destination).free)
    # Reserve room for a resumed download and, when requested, a second copy of
    # the archive while extraction writes selected contents.
    required = int(size_bytes * (2.05 if extract else 1.10))
    if free < required:
        raise OSError(f"insufficient free space: have {free} bytes, require {required} bytes")
    return {"free_bytes_before_download": free, "minimum_free_bytes_required": required}


def _download_by_verified_ranges(*, url: str, partial: Path, expected_size: int, headers: dict[str, str]) -> dict[str, Any]:
    """Download an archive in bounded HTTP ranges and resume only at verified offsets."""
    offset = partial.stat().st_size if partial.exists() else 0
    resumed = offset > 0
    if offset > expected_size:
        raise RuntimeError(f"partial download exceeds expected size; preserving it: {partial}")
    last_progress = offset
    status: int | None = None

    def fetch_range(start: int, end: int) -> tuple[bytes, int]:
        expected_length = end - start + 1
        for attempt in range(9):
            try:
                range_headers = {**headers, "Range": f"bytes={start}-{end}"}
                with requests.get(url, headers=range_headers, stream=True, timeout=(20, 120)) as response:
                    response_status = int(response.status_code)
                    if response_status != 206:
                        raise RuntimeError(f"server did not honor byte range {start}-{end}: HTTP {response_status}")
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+)", content_range.strip(), flags=re.IGNORECASE)
                    if not match:
                        raise RuntimeError(f"missing or malformed Content-Range for requested bytes {start}-{end}: {content_range!r}")
                    returned_start, returned_end, returned_total = map(int, match.groups())
                    if (returned_start, returned_end, returned_total) != (start, end, expected_size):
                        raise RuntimeError(
                            f"server returned a mismatched byte range: requested {start}-{end}/{expected_size}, "
                            f"received {returned_start}-{returned_end}/{returned_total}"
                        )
                    body = bytearray()
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            body.extend(block)
                            if len(body) > expected_length:
                                raise RuntimeError("server returned more bytes than requested")
                    if len(body) == expected_length:
                        return bytes(body), response_status
                    last_error: Exception = RuntimeError(
                        f"short ranged response for bytes {start}-{end}: received {len(body)}, expected {expected_length}"
                    )
            except requests.RequestException as exc:
                last_error = exc
            if attempt < 8:
                time.sleep(min(0.5 * (2 ** attempt), 8.0))
        raise RuntimeError(f"ranged download repeatedly interrupted for bytes {start}-{end}: {last_error}") from last_error

    workers = 16
    while offset < expected_size:
        ranges: list[tuple[int, int]] = []
        batch_end = offset
        for _ in range(workers):
            if batch_end >= expected_size:
                break
            end = min(batch_end + RANGE_CHUNK_BYTES - 1, expected_size - 1)
            ranges.append((batch_end, end))
            batch_end = end + 1
        with ThreadPoolExecutor(max_workers=min(workers, len(ranges))) as executor:
            futures = [executor.submit(fetch_range, start, end) for start, end in ranges]
            results = [future.result() for future in futures]
        actual_size = partial.stat().st_size if partial.exists() else 0
        if actual_size != offset:
            raise RuntimeError(f"partial changed during ranged download: expected offset {offset}, found {actual_size}")
        with partial.open("ab" if offset else "wb") as stream:
            for (start, end), (body, response_status) in zip(ranges, results, strict=True):
                if start != stream.tell() or len(body) != end - start + 1:
                    raise RuntimeError("verified range batch is not contiguous; partial retained")
                stream.write(body)
                status = response_status
        offset = partial.stat().st_size
        if offset - last_progress >= 64 * 1024 * 1024 or offset == expected_size:
            print(f"Downloading {partial.name}: {offset:,}/{expected_size:,} bytes", flush=True)
            last_progress = offset
    return {
        "http_status": status,
        "resumed": resumed,
        "range_chunk_bytes": RANGE_CHUNK_BYTES,
        "parallel_range_workers": workers,
    }


def _download_file(
    *,
    url: str,
    target: Path,
    expected_size: int,
    publisher_checksum: str | None,
    extract_zip: bool,
) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    preflight = _disk_preflight(target.parent, expected_size, extract=extract_zip)
    expected = _parse_publisher_checksum(publisher_checksum)
    if target.exists():
        if target.stat().st_size != expected_size:
            raise FileExistsError(f"existing file has unexpected size; preserving it: {target}")
        if expected and _checksum_digest(target, expected[0]) != expected[1]:
            raise FileExistsError(f"existing file fails publisher checksum; preserving it: {target}")
        final_sha256 = _checksum_digest(target, "sha256")
        result: dict[str, Any] = {
            "status": "already_present_verified",
            "path": str(target.resolve()),
            "size_bytes": target.stat().st_size,
            "publisher_checksum": publisher_checksum,
            "sha256": final_sha256,
            **preflight,
        }
    else:
        partial = target.with_name(target.name + ".part")
        headers = {"User-Agent": USER_AGENT}
        if expected_size >= 16 * 1024 * 1024:
            transfer = _download_by_verified_ranges(url=url, partial=partial, expected_size=expected_size, headers=headers)
            status = transfer["http_status"]
            resumed = bool(transfer["resumed"])
        else:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with requests.get(url, headers=headers, stream=True, timeout=(15, 120)) as response:
                    response.raise_for_status()
                    status = int(response.status_code)
                    if offset and status == 206:
                        mode = "ab"
                        resumed = True
                    else:
                        mode = "wb"
                        resumed = False
                        offset = 0
                    with partial.open(mode) as stream:
                        last_progress = offset
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if block:
                                stream.write(block)
                                offset += len(block)
                                if offset - last_progress >= 64 * 1024 * 1024:
                                    print(f"Downloading {target.name}: {offset:,}/{expected_size:,} bytes", flush=True)
                                    last_progress = offset
            except (requests.RequestException, TimeoutError) as exc:
                raise RuntimeError(f"download interrupted; resumable partial remains at {partial}: {exc}") from exc
        actual_size = partial.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(f"download size mismatch for {target.name}: expected {expected_size}, got {actual_size}; partial retained")
        if expected and _checksum_digest(partial, expected[0]) != expected[1]:
            raise RuntimeError(f"publisher checksum mismatch for {target.name}; partial retained")
        final_sha256 = _checksum_digest(partial, "sha256")
        partial.replace(target)
        result = {
            "status": "downloaded_and_verified",
            "path": str(target.resolve()),
            "size_bytes": target.stat().st_size,
            "publisher_checksum": publisher_checksum,
            "sha256": final_sha256,
            "http_status": status,
            "resumed": resumed,
            **preflight,
        }

    if target.suffix.lower() == ".zip":
        result["archive_integrity"] = _validate_zip(target)
        if extract_zip:
            if target.stat().st_size > SMALL_AUTO_EXTRACT_LIMIT_BYTES:
                raise ValueError("automatic extraction is limited to archives under 100 MB")
            aliases = {
                "self_perceived_fatigue_index": "labels",
                "Subject_Data": "subject_data",
                "Knee extension": "knee",
            }
            alias = aliases.get(target.stem, target.stem[:10].lower())
            result["extraction"] = _extract_zip_safely(target, target.parent.parent / "u" / alias)
    return result


def _validate_zip(path: Path) -> dict[str, Any]:
    if not zipfile.is_zipfile(path):
        raise zipfile.BadZipFile(f"not a valid ZIP archive: {path}")
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"CRC error in {path.name}: {bad_member}")
        return {
            "status": "crc_passed",
            "member_count": len(archive.infolist()),
            "uncompressed_bytes": sum(info.file_size for info in archive.infolist()),
        }


def _safe_zip_member(info: zipfile.ZipInfo) -> PurePosixPath:
    path = PurePosixPath(info.filename)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member path: {info.filename!r}")
    if path.parts[0].endswith(":"):
        raise ValueError(f"archive member uses a drive path: {info.filename!r}")
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    if unix_mode and (unix_mode & 0o170000) == 0o120000:
        raise ValueError(f"archive symlinks are not extracted: {info.filename!r}")
    return path


def _extract_zip_safely(archive_path: Path, destination: Path, *, extract_nested: bool = True) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    nested: list[dict[str, Any]] = []
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            member = _safe_zip_member(info)
            target = destination.joinpath(*member.parts).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"archive member escapes extraction directory: {info.filename!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
            extracted.append({"member": info.filename, "size_bytes": target.stat().st_size, "sha256": _checksum_digest(target, "sha256")})
            if extract_nested and target.suffix.lower() == ".zip" and target.stat().st_size <= SMALL_AUTO_EXTRACT_LIMIT_BYTES:
                nested.append(_validate_zip(target))
                nested_key = hashlib.sha256(info.filename.encode("utf-8")).hexdigest()[:8]
                nested_destination = destination.parent / f"n{nested_key}"
                nested.append(_extract_zip_safely(target, nested_destination, extract_nested=False))
    return {
        "status": "extracted_selected_archive_contents",
        "destination": str(destination.resolve()),
        "member_count": len(extracted),
        "members": extracted,
        "nested_archives": nested,
    }


def _zenodo_file_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["key"]): {
            "name": str(item["key"]),
            "size_bytes": int(item["size"]),
            "publisher_checksum": item.get("checksum"),
            "download_url": item.get("links", {}).get("self") or item.get("links", {}).get("content"),
        }
        for item in record.get("files", [])
    }


def _fetch_zenodo(record_id: str, expected_prefix: str) -> dict[str, Any]:
    record = _http_json(f"https://zenodo.org/api/records/{record_id}")
    metadata = record.get("metadata", {})
    if not str(metadata.get("title", "")).lower().startswith(expected_prefix.lower()):
        raise ValueError(f"Zenodo record {record_id} resolved to an unexpected title: {metadata.get('title')!r}")
    access_right = str(metadata.get("access_right", "unknown"))
    if access_right not in {"open", "embargoed"}:
        raise PermissionError(f"Zenodo record {record_id} is not open access: {access_right}")
    return {"raw": record, "metadata": metadata, "files": _zenodo_file_map(record)}


def _dataset_metadata(record: dict[str, Any], *, kind: str, record_url: str) -> dict[str, Any]:
    metadata = record["metadata"]
    creators = metadata.get("creators", [])
    license_value = metadata.get("license") or {}
    if isinstance(license_value, dict):
        license_id = license_value.get("id") or license_value.get("title")
    else:
        license_id = license_value
    return {
        "dataset_key": kind,
        "title": metadata.get("title"),
        "doi": metadata.get("doi"),
        "version": metadata.get("version"),
        "authors": creators,
        "license": license_id,
        "publication_date": metadata.get("publication_date"),
        "publisher": "Zenodo",
        "official_record_url": record_url,
        "metadata_fetched_at_utc": _utc_now(),
        "available_files": record["files"],
        "download_status": "metadata_verified_files_not_selected_here",
    }


def _download_selected_zenodo(
    *,
    record: dict[str, Any],
    dataset_dir: Path,
    selected_names: Iterable[str],
    extract_small_zips: bool,
) -> dict[str, Any]:
    files = record["files"]
    results: dict[str, Any] = {}
    missing: list[str] = []
    for name in selected_names:
        info = files.get(name)
        if info is None:
            missing.append(name)
            continue
        target = dataset_dir / "originals" / _safe_basename(name)
        should_extract = extract_small_zips and target.suffix.lower() == ".zip" and int(info["size_bytes"]) <= SMALL_AUTO_EXTRACT_LIMIT_BYTES
        results[name] = _download_file(
            url=str(info["download_url"]),
            target=target,
            expected_size=int(info["size_bytes"]),
            publisher_checksum=info.get("publisher_checksum"),
            extract_zip=should_extract,
        )
    if missing:
        raise FileNotFoundError(f"selected files missing from official record: {missing}")
    return results


def _fetch_figshare(article_id: str) -> dict[str, Any]:
    article = _http_json(f"https://api.figshare.com/v2/articles/{article_id}")
    expected_title = "Less common synaptic input between muscles from the same group allows for more flexible coordination strategies during a fatiguing task"
    if str(article.get("title", "")).casefold() != expected_title.casefold():
        raise ValueError(f"Figshare article {article_id} resolved to an unexpected title: {article.get('title')!r}")
    return article


def _figshare_metadata(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_key": "figshare_quadriceps_knee_extension",
        "title": article.get("title"),
        "doi": article.get("doi"),
        "version": article.get("version"),
        "authors": article.get("authors"),
        "license": article.get("license"),
        "publication_date": article.get("published_date"),
        "publisher": "Figshare",
        "official_record_url": article.get("url_public_html"),
        "metadata_fetched_at_utc": _utc_now(),
        "available_files": article.get("files", []),
        "download_status": "metadata_verified_files_not_selected_here",
    }


def _download_figshare_knee(article: dict[str, Any], dataset_dir: Path, extract: bool) -> dict[str, Any]:
    matches = [item for item in article.get("files", []) if str(item.get("name", "")).casefold() == "knee extension.zip"]
    if len(matches) != 1:
        raise ValueError(f"expected one knee extension archive, found {len(matches)}")
    item = matches[0]
    target = dataset_dir / "originals" / _safe_basename(str(item["name"]))
    return _download_file(
        url=str(item["download_url"]),
        target=target,
        expected_size=int(item["size"]),
        publisher_checksum=str(item.get("computed_md5") or "") or None,
        extract_zip=bool(extract),
    )


def _oxford_access_record() -> dict[str, Any]:
    return {
        "dataset_key": "oxford_motor_fatigue",
        "title": "Investigating motor fatigue using force, EMG, and kinematic data",
        "doi": "10.60964/rnd-rtyw-9w81",
        "publisher": "University of Oxford MRC BNDU Cambium",
        "official_record_url": "https://data.mrc.ox.ac.uk/data-set/investigating-motor-fatigue-using-force-emg-and-kinematic-data",
        "access_status": "ACCESS_PENDING",
        "access_basis": "official dataset page requires registration and login before download or access request; no authentication was supplied",
        "metadata_fetched_at_utc": _utc_now(),
    }


def _load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"registry_version": 1, "datasets": {}}


def run_download(
    run_dir: Path,
    *,
    datasets: set[str],
    download_cerqueira_signals: bool = False,
    download_swifties_sensors: bool = False,
    download_figshare_knee: bool = False,
    extract_figshare_knee: bool = False,
    extract_small_zips: bool = True,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve() / "private" / "external_supervision"
    root.mkdir(parents=True, exist_ok=True)
    registry_path = root / "dataset_registry.json"
    registry = _load_registry(registry_path)
    results: dict[str, Any] = {}

    if "cerqueira" in datasets:
        dataset_dir = root / "cerqueira_v2"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        record = _fetch_zenodo(CERQUEIRA_ID, "A Dataset of sEMG and Self-Perceived Fatigue Levels")
        metadata = _dataset_metadata(record, kind="cerqueira_self_perceived_fatigue", record_url=f"https://zenodo.org/records/{CERQUEIRA_ID}")
        _write_json(dataset_dir / "official_metadata.json", metadata)
        selected = ["Metadata.xlsx", "protocol.xlsx", "self_perceived_fatigue_index.zip", "code.ipynb"]
        if download_cerqueira_signals:
            selected.append("sEMG_data.zip")
        files = _download_selected_zenodo(record=record, dataset_dir=dataset_dir, selected_names=selected, extract_small_zips=extract_small_zips)
        metadata["selected_file_receipts"] = files
        metadata["download_status"] = "metadata_and_labels_downloaded" if not download_cerqueira_signals else "including_signal_archive_downloaded"
        _write_json(dataset_dir / "official_metadata.json", metadata)
        registry["datasets"]["cerqueira_v2"] = metadata
        results["cerqueira_v2"] = {"directory": str(dataset_dir), "files": files, "title": metadata["title"], "doi": metadata["doi"]}

    if "swifties" in datasets:
        dataset_dir = root / "swifties_v5"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        record = _fetch_zenodo(SWIFTIES_ID, "SWIFTIES ")
        metadata = _dataset_metadata(record, kind="swifties_movement_and_fatigue_tasks", record_url=f"https://zenodo.org/records/{SWIFTIES_ID}")
        metadata["version"] = metadata.get("version") or "v5 (publisher landing page)"
        _write_json(dataset_dir / "official_metadata.json", metadata)
        selected = ["README.txt", "description.txt", "Subject_Data.zip", "SensorDataSummary.xlsx"]
        if download_swifties_sensors:
            selected.append("Sensor_Data.zip")
        files = _download_selected_zenodo(record=record, dataset_dir=dataset_dir, selected_names=selected, extract_small_zips=extract_small_zips)
        metadata["selected_file_receipts"] = files
        metadata["download_status"] = "metadata_only_sensor_archive_skipped" if not download_swifties_sensors else "including_sensor_archive_downloaded"
        _write_json(dataset_dir / "official_metadata.json", metadata)
        registry["datasets"]["swifties_v5"] = metadata
        results["swifties_v5"] = {"directory": str(dataset_dir), "files": files, "title": metadata["title"], "doi": metadata["doi"]}

    if "figshare" in datasets:
        dataset_dir = root / "figshare_quadriceps"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        article = _fetch_figshare(FIGSHARE_ID)
        metadata = _figshare_metadata(article)
        _write_json(dataset_dir / "official_metadata.json", metadata)
        knee = next((item for item in article.get("files", []) if str(item.get("name", "")).casefold() == "knee extension.zip"), None)
        results["figshare_knee_metadata"] = {
            "directory": str(dataset_dir), "title": metadata["title"], "doi": metadata["doi"],
            "license": metadata["license"],
            "knee_archive": None if knee is None else {key: knee.get(key) for key in ("name", "size", "computed_md5", "download_url")},
        }
        if download_figshare_knee:
            files = _download_figshare_knee(article, dataset_dir, extract_figshare_knee)
            metadata["selected_file_receipts"] = {str(knee["name"]): files}
            metadata["download_status"] = "knee_extension_archive_downloaded"
            registry["datasets"]["figshare_quadriceps"] = metadata
            _write_json(dataset_dir / "official_metadata.json", metadata)
            results["figshare_knee_metadata"]["download"] = files
        else:
            metadata["download_status"] = "record_and_archive_metadata_verified_archive_not_downloaded"
            registry["datasets"]["figshare_quadriceps"] = metadata
            _write_json(dataset_dir / "official_metadata.json", metadata)

    if "oxford" in datasets:
        oxford = _oxford_access_record()
        registry["datasets"]["oxford_motor_fatigue"] = oxford
        results["oxford_motor_fatigue"] = oxford

    registry["updated_at_utc"] = _utc_now()
    _write_json(registry_path, registry)
    return {"registry_path": str(registry_path), "external_root": str(root), "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Existing private experiment run root")
    parser.add_argument("--datasets", default="cerqueira,swifties", help="Comma-separated: cerqueira,swifties,figshare,oxford")
    parser.add_argument("--download-cerqueira-signals", action="store_true", help="Also fetch the 3.3 GB original signal archive")
    parser.add_argument("--download-swifties-sensors", action="store_true", help="Also fetch the 23.6 GB sensor archive after disk preflight")
    parser.add_argument("--download-figshare-knee", action="store_true", help="Fetch the approximately 10.6 GB knee extension archive")
    parser.add_argument("--extract-figshare-knee", action="store_true", help="Extract the knee archive; requires roughly twice its compressed size free")
    parser.add_argument("--no-extract-small-zips", action="store_true", help="Keep small ZIP files unopened for a metadata-only pass")
    args = parser.parse_args()
    allowed = {"cerqueira", "swifties", "figshare", "oxford"}
    datasets = {item.strip().lower() for item in args.datasets.split(",") if item.strip()}
    if not datasets or not datasets.issubset(allowed):
        parser.error(f"--datasets must be a comma-separated subset of {sorted(allowed)}")
    if args.extract_figshare_knee and not args.download_figshare_knee:
        parser.error("--extract-figshare-knee requires --download-figshare-knee")
    result = run_download(
        args.run_dir,
        datasets=datasets,
        download_cerqueira_signals=args.download_cerqueira_signals,
        download_swifties_sensors=args.download_swifties_sensors,
        download_figshare_knee=args.download_figshare_knee,
        extract_figshare_knee=args.extract_figshare_knee,
        extract_small_zips=not args.no_extract_small_zips,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
