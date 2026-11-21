#!/usr/bin/env python3
"""Fail closed on tracked files outside the reviewed public source boundary.

This checks the current Git index and working copies; it does not certify prose
as de-identified or scrub existing history. Initial publication must use a fresh
history, and future commits still need human review.
"""

from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT_FILES = {"README.md", "NOTICE.md", "pyproject.toml", ".gitignore"}
PREFIXES = {
    "src": {".py"},
    "tests": {".py"},
    "tools": {".py"},
    "docs": {".md"},
    "configs": {".json"},
    "examples": {".py", ".json"},
    ".github": {".yml", ".yaml"},
}
PATTERNS = {
    "machine-specific filesystem path": r"(?:/Users/|/home/|/data2/|/corral(?:-repl)?/|/scratch/)",
    "patient-style identifier": r"\b\d{2}_\d{5,8}\b",
    "TCGA specimen identifier": r"\bTCGA-[A-Z0-9]{2}-[A-Z0-9]{4}(?:-[A-Z0-9-]+)?\b",
    "DICOM identifier": r"\b1\.3\.6\.1\.4\.1\.\d+(?:\.\d+){3,}",
    "GitHub credential": r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})",
    "private key": r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----",
    "cloud access credential": r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "embedded password/token assignment": r"""(?i)(?:api_key|access_token|password)\s*[:=]\s*["'][^"'\s]{12,}["']""",
}


def inspect_text(path, blob):
    errors = []
    if len(blob) > 300_000 or b"\x00" in blob:
        return [f"{path}: binary or unexpectedly large file"]
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        return [f"{path}: non-UTF-8 content"]
    # The scanner's literal patterns describe prohibited strings; don't flag
    # those pattern definitions as an actual occurrence of such a string.
    if path != "tools/check_public_tree.py":
        for name, pattern in PATTERNS.items():
            if re.search(pattern, text):
                errors.append(f"{path}: {name}")
    return errors


def check(root):
    root = Path(root)
    proc = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    paths = [p.decode() for p in proc.stdout.split(b"\0") if p]
    if not paths:
        raise ValueError("No tracked/staged files; stage the reviewed source files first")
    errors, bytes_checked = [], 0
    for path in paths:
        p = PurePosixPath(path)
        allowed = path in ROOT_FILES or (p.parts[0] in PREFIXES and p.suffix in PREFIXES[p.parts[0]])
        if (
            not allowed
            or ".." in p.parts
            or any(x in p.parts for x in ("data", "outputs", "private", "__pycache__"))
        ):
            errors.append(f"{path}: outside publication allowlist")
            continue
        stage = subprocess.run(
            ["git", "ls-files", "-s", "--", path], cwd=root, capture_output=True, text=True, check=True
        ).stdout
        if not stage.startswith(("100644 ", "100755 ")) or (root / path).is_symlink():
            errors.append(f"{path}: symlink, submodule, or unsupported file mode")
            continue
        indexed = subprocess.run(
            ["git", "show", f":{path}"], cwd=root, capture_output=True, check=True
        ).stdout
        errors.extend(inspect_text(path, indexed))
        if (root / path).is_file():
            working = (root / path).read_bytes()
            if working != indexed:
                errors.extend(inspect_text(path, working))
        bytes_checked += len(indexed)
    return paths, bytes_checked, errors


def main():
    root = Path(__file__).resolve().parents[1]
    paths, size, errors = check(root)
    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        raise SystemExit(1)
    print(f"Public-tree check passed: {len(paths)} tracked text files, {size:,} bytes; no data artifacts.")


if __name__ == "__main__":
    main()
