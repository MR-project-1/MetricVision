"""Download single KITTI odometry sequences or EuRoC scenes without fetching the full archives.

KITTI ships every sequence in one 23 GB zip and EuRoC ships each environment as one 6-13 GB zip.
Both servers support HTTP range requests, so this script reads the zip directory remotely and
extracts only the requested sequence (about 2.4 GB for KITTI 00, 1.6 GB for EuRoC MH_01).

Usage (remotezip is only needed here, so it is not a project dependency):
    uv run --with remotezip python scripts/download_datasets.py kitti --sequence 00 --out ~/datasets/kitti
    uv run --with remotezip python scripts/download_datasets.py euroc --sequence MH_01_easy --out ~/datasets/euroc
    ... --dry-run   # Only list what would be downloaded.

Resulting layouts match configs/kitti.yaml (``<out>/sequences/00``, ``<out>/poses/00.txt``) and
configs/euroc.yaml (``<out>/MH_01_easy/mav0``). Re-running resumes: existing files are skipped.
"""

from __future__ import annotations  # Postponed evaluation of type hints.

import argparse  # Command-line arguments.
import struct  # Zip local-header parsing.
import time  # Retry back-off.
import urllib.request  # Streaming range downloads.
import zipfile  # Unpacking the inner EuRoC zip.
import zlib  # Streaming inflate of the inner EuRoC zip.
from concurrent.futures import ThreadPoolExecutor  # Parallel KITTI image fetches.
from pathlib import Path  # Paths.
from threading import local  # One remote zip handle per thread.

from remotezip import RemoteZip  # Remote zip directory access via range requests.

KITTI_GRAY = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip"  # All grayscale sequences.
KITTI_POSES = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip"  # Ground truth (00-10).
EUROC_BITSTREAMS = {  # ETH Research Collection files (https://doi.org/10.3929/ethz-b-000690084).
    "MH": "https://www.research-collection.ethz.ch/server/api/core/bitstreams/7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/content",  # machine_hall.zip
    "V1": "https://www.research-collection.ethz.ch/server/api/core/bitstreams/02ecda9a-298f-498b-970c-b7c44334d880/content",  # vicon_room1.zip
    "V2": "https://www.research-collection.ethz.ch/server/api/core/bitstreams/ea12bc01-3677-4b4c-853d-87c7870b8c44/content",  # vicon_room2.zip
}
HEADERS = {"User-Agent": "Mozilla/5.0"}  # The ETH server rejects the default Python agent.
_thread = local()  # Per-thread RemoteZip handles.


def _retry(action, attempts: int = 30):  # noqa: ANN001, ANN202
    """Run ``action`` with retries (flaky DNS and connection resets are common on long downloads).

    Args:
        action: Zero-argument callable.
        attempts: Maximum attempts.

    Returns:
        Whatever ``action`` returns.
    """
    for attempt in range(attempts):  # Bounded retries.
        try:
            return action()  # Success.
        except Exception:  # noqa: BLE001  # Network errors of many kinds.
            if attempt == attempts - 1:  # Out of attempts.
                raise  # Surface the last error.
            time.sleep(min(2 + attempt, 15))  # Back off.
    return None  # Unreachable.


def kitti(sequence: str, out: Path, dry_run: bool) -> None:
    """Fetch one KITTI odometry sequence (both grayscale cameras, calib, times) and its poses if public.

    Args:
        sequence: Two-digit sequence id, e.g. ``00``.
        out: Destination root (gets ``sequences/<seq>`` and ``poses/<seq>.txt``).
        dry_run: Only print what would be fetched.
    """
    with RemoteZip(KITTI_GRAY) as archive:  # Read the central directory only.
        members = [i for i in archive.infolist() if f"/sequences/{sequence}/" in i.filename and not i.filename.endswith("/")]  # This sequence.
    print(f"KITTI {sequence}: {len(members)} files, {sum(i.file_size for i in members) / 1e9:.2f} GB")  # Summary.
    if dry_run:  # Listing only.
        return
    if int(sequence) <= 10:  # Ground truth is public for 00-10 only.
        with RemoteZip(KITTI_POSES) as archive:  # Small archive.
            (out / "poses").mkdir(parents=True, exist_ok=True)  # Folder.
            (out / "poses" / f"{sequence}.txt").write_bytes(archive.read(f"dataset/poses/{sequence}.txt"))  # GT file.

    def fetch(name: str) -> None:
        """Download one member unless it already exists."""
        target = out / name.removeprefix("dataset/")  # Strip the archive's top folder.
        if target.exists() and target.stat().st_size > 0:  # Resume support.
            return
        target.parent.mkdir(parents=True, exist_ok=True)  # Folder.

        def read() -> bytes:
            """Read the member with a per-thread connection, reconnecting on failure."""
            if not hasattr(_thread, "zip"):  # Lazily open per thread.
                _thread.zip = RemoteZip(KITTI_GRAY)  # Own connection.
            try:
                return _thread.zip.read(name)  # Range request.
            except Exception:
                del _thread.zip  # Fresh connection next attempt.
                raise

        partial = target.with_suffix(target.suffix + ".part")  # Atomic write.
        partial.write_bytes(_retry(read))  # Download.
        partial.rename(target)  # Publish.

    with ThreadPoolExecutor(max_workers=8) as pool:  # Parallel small files.
        for done, _ in enumerate(pool.map(fetch, [m.filename for m in members]), start=1):  # In order.
            if done % 1000 == 0 or done == len(members):  # Progress.
                print(f"  {done}/{len(members)}")


def euroc(sequence: str, out: Path, dry_run: bool) -> None:
    """Fetch one EuRoC scene in ASL format and unpack it to ``<out>/<sequence>/mav0``.

    Args:
        sequence: Scene name, e.g. ``MH_01_easy`` or ``V1_02_medium``.
        out: Destination root.
        dry_run: Only print what would be fetched.

    Raises:
        ValueError: If the scene name is unknown.
    """
    url = EUROC_BITSTREAMS.get(sequence[:2])  # Environment archive.
    if url is None:  # Unknown prefix.
        raise ValueError(f"EuRoC scene must start with MH, V1 or V2, got {sequence}")
    with RemoteZip(url, headers=HEADERS) as archive:  # Central directory only.
        info = next((i for i in archive.infolist() if i.filename.endswith(f"{sequence}/{sequence}.zip")), None)  # ASL zip.
    if info is None:  # Not in this archive.
        raise ValueError(f"{sequence} not found in the EuRoC archive")
    print(f"EuRoC {sequence}: {info.compress_size / 1e9:.2f} GB")  # Summary.
    if dry_run or (out / sequence / "mav0").is_dir():  # Listing only, or already unpacked.
        return
    out.mkdir(parents=True, exist_ok=True)  # Folder.
    header = urllib.request.urlopen(
        urllib.request.Request(url, headers={**HEADERS, "Range": f"bytes={info.header_offset}-{info.header_offset + 29}"})
    ).read()  # Local header.
    name_len, extra_len = struct.unpack("<HH", header[26:30])  # Variable header part.
    start = info.header_offset + 30 + name_len + extra_len  # First compressed byte.
    inner = out / f"{sequence}.zip"  # Inner ASL zip on disk.
    request = urllib.request.Request(url, headers={**HEADERS, "Range": f"bytes={start}-{start + info.compress_size - 1}"})  # Member bytes.
    inflater = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None  # Raw deflate stream.
    with urllib.request.urlopen(request, timeout=60) as response, inner.open("wb") as handle:  # Stream to disk.
        while chunk := response.read(1 << 22):  # 4 MB chunks keep memory flat.
            handle.write(inflater.decompress(chunk) if inflater else chunk)  # Inflate on the fly.
        if inflater:  # Flush the tail.
            handle.write(inflater.flush())
    with zipfile.ZipFile(inner) as asl:  # Unpack mav0/.
        asl.extractall(out / sequence)  # -> <out>/<sequence>/mav0.
    inner.unlink()  # Free the space.
    print(f"  unpacked to {out / sequence}")


def main() -> None:
    """Parse arguments and download."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # CLI.
    parser.add_argument("dataset", choices=["kitti", "euroc"])  # Which benchmark.
    parser.add_argument("--sequence", required=True, help="KITTI id (00-21) or EuRoC scene (e.g. MH_01_easy).")  # Which sequence.
    parser.add_argument("--out", type=Path, required=True, help="Destination root folder.")  # Where.
    parser.add_argument("--dry-run", action="store_true", help="Only list sizes.")  # Listing.
    args = parser.parse_args()  # Parse.
    out = args.out.expanduser()  # Allow ~.
    if args.dataset == "kitti":  # KITTI.
        kitti(f"{int(args.sequence):02d}", out, args.dry_run)
    else:  # EuRoC.
        euroc(args.sequence, out, args.dry_run)


if __name__ == "__main__":  # Script entry.
    main()  # Run.
