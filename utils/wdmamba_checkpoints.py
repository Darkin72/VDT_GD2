"""Download the WDMamba archive without treating CLI exit 0 as success."""

from pathlib import Path
import codecs
import posixpath
import re
import queue
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile


SHARE_URL = "https://pan.baidu.com/s/1HIs-nHXEaLxwBb1279PVbw?pwd=98j9"
SHARE_CODE = "98j9"
ARCHIVE_NAME = "WDMamba_ckpts.zip"


def run_pcs(pcs, *args, secrets=(), timeout=900, heartbeat_interval=15):
    """Stream BaiduPCS-Go output and keep it for success/error checks.

    BaiduPCS-Go uses carriage returns for progress bars and can stay silent while
    negotiating a download.  Reading chunks instead of waiting for ``communicate``
    makes both cases visible in a notebook.
    """
    command = [str(pcs), *map(str, args)]
    def redact(text):
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[hidden]")
        return text
    print("$", redact(" ".join(command)), flush=True)
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, bufsize=0,
    )
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    started = time.monotonic()
    last_activity = started
    messages = queue.Queue()
    chunks = []
    pending = ""
    # Keep enough text to redact a secret even when it spans pipe reads.
    keep = max((len(secret) - 1 for secret in secrets if secret), default=0)

    def read_output():
        try:
            while True:
                data = process.stdout.read(8192)
                if not data:
                    break
                messages.put(data)
        except Exception as exc:
            messages.put(exc)
        finally:
            messages.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        while True:
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"BaiduPCS-Go {args[0]} exceeded {timeout}s; see live log above.")
            try:
                data = messages.get(timeout=0.2)
            except queue.Empty:
                now = time.monotonic()
                if now - last_activity >= heartbeat_interval:
                    elapsed = int(now - started)
                    print(
                        f"\n[BaiduPCS-Go {args[0]}: {elapsed}s elapsed; waiting for more log output]",
                        flush=True,
                    )
                    last_activity = now
                continue
            if data is None:
                break
            if isinstance(data, Exception):
                raise data
            last_activity = time.monotonic()
            pending = redact(pending + decoder.decode(data))
            emit_count = max(0, len(pending) - keep)
            if emit_count:
                text, pending = pending[:emit_count], pending[emit_count:]
                chunks.append(text)
                sys.stdout.write(text)
                sys.stdout.flush()
        tail = redact(pending + decoder.decode(b"", final=True))
        if tail:
            chunks.append(tail)
            sys.stdout.write(tail)
        process.wait(timeout=max(0.1, timeout - (time.monotonic() - started)))
    finally:
        # Interrupting a notebook cell must also stop the downloader it started.
        if process.poll() is None:
            process.kill()
        process.wait()
        reader.join(timeout=5)
        process.stdout.close()
    sys.stdout.write("\n")
    sys.stdout.flush()
    output = "".join(chunks)
    if process.returncode:
        raise RuntimeError(f"BaiduPCS-Go {args[0]} exited with {process.returncode}. See log above.")
    if not output:
        print("(BaiduPCS-Go produced no output; heartbeat confirms whether it was still running.)", flush=True)
    return output


def download_baidu(pcs, destination):
    """Download only files from the official share using an authenticated account."""
    who = run_pcs(pcs, "who")
    uid = re.search(r"\buid\s*:\s*(\d+)", who, re.IGNORECASE)
    if uid is None or int(uid.group(1)) == 0:
        raise RuntimeError(
            "BaiduPCS-Go has no authenticated Baidu account. The share code 98j9 is not a login. "
            "Set BAIDU_LOGIN=True to log in privately, or use WDMAMBA_SOURCE='local'/'upload' "
            "with the official WDMamba_ckpts.zip downloaded in your browser."
        )
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    print(
        "Starting Baidu transfer/download. The official archive is about 188 MB; "
        "progress output and a heartbeat will appear below.",
        flush=True,
    )
    run_pcs(pcs, "config", "set", "-savedir", destination)
    # --download downloads only the transferred share. Flags must precede positional arguments.
    output = run_pcs(pcs, "transfer", "--download", SHARE_URL, SHARE_CODE)
    if "文件重复" in output:
        # Transfer saves the share in BaiduPCS-Go's remote working directory.
        # A previous run may have saved it there without completing the download.
        print("Archive already exists in Baidu Cloud; downloading that file directly.", flush=True)
        workdir_output = run_pcs(pcs, "pwd")
        workdirs = [line.strip() for line in workdir_output.splitlines() if line.strip().startswith("/")]
        if len(workdirs) != 1:
            raise RuntimeError("Cannot determine Baidu remote working directory. See the pwd log above.")
        remote_archive = posixpath.join(workdirs[0], ARCHIVE_NAME)
        run_pcs(pcs, "download", remote_archive, "--saveto", destination, "--ow")
    elif "失败" in output or "分享链接转存到网盘成功" not in output:
        raise RuntimeError(
            "Baidu did not confirm a successful transfer. See BaiduPCS-Go log above. "
            "Use WDMAMBA_SOURCE='local'/'upload' if Baidu requests browser verification."
        )
    archives = sorted(destination.rglob(ARCHIVE_NAME))
    if not archives or any(not zipfile.is_zipfile(path) for path in archives):
        raise RuntimeError(
            "BaiduPCS-Go did not save a complete WDMamba_ckpts.zip. "
            "See its download log; use a local/Drive copy if Baidu requests verification."
        )
    return destination


def download_url(url, destination):
    """Accept a direct download URL, not a Baidu or Google Drive share page."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    try:
        urllib.request.urlretrieve(url, partial)
        with partial.open("rb") as handle:
            header = handle.read(512).lstrip().lower()
        if not header or header.startswith((b"<!doctype html", b"<html")):
            raise RuntimeError("The URL returned HTML/empty content; provide a direct file URL.")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination

