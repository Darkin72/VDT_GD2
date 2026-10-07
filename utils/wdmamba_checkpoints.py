"""Download the WDMamba archive without treating CLI exit 0 as success."""

from pathlib import Path
import re
import subprocess
import urllib.request
import zipfile


SHARE_URL = "https://pan.baidu.com/s/1HIs-nHXEaLxwBb1279PVbw?pwd=98j9"
SHARE_CODE = "98j9"
ARCHIVE_NAME = "WDMamba_ckpts.zip"


def run_pcs(pcs, *args, secrets=()):
    """BaiduPCS-Go reports some application failures with a zero exit status."""
    command = [str(pcs), *map(str, args)]
    def redact(text):
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[hidden]")
        return text
    print("$", redact(" ".join(command)), flush=True)
    result = subprocess.run(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", timeout=900,
    )
    output = redact(result.stdout)
    print(output or "(BaiduPCS-Go produced no output)", flush=True)
    if result.returncode:
        raise RuntimeError(f"BaiduPCS-Go {args[0]} exited with {result.returncode}. See log above.")
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
    run_pcs(pcs, "config", "set", "-savedir", destination)
    # --download downloads only the transferred share. Flags must precede positional arguments.
    output = run_pcs(pcs, "transfer", "--download", SHARE_URL, SHARE_CODE)
    if "失败" in output or "分享链接转存到网盘成功" not in output:
        raise RuntimeError(
            "Baidu did not confirm a successful transfer. See BaiduPCS-Go log above. "
            "If the file already exists in your account, download that archive in the browser "
            "and use WDMAMBA_SOURCE='local'/'upload'."
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

