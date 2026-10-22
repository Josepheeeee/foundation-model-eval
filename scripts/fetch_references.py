"""Fetch complete upstream source trees at immutable commits; never reset dirty work."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def git(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def fetch(name: str, spec: dict):
    parent = (ROOT / "external").resolve()
    target = (parent / name).resolve()
    if target.parent != parent or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("reference path outside external")
    if not re.fullmatch(r"[0-9a-f]{40}", spec["commit"]):
        raise ValueError("immutable commit required")
    if target.exists():
        if git("status", "--porcelain", cwd=target):
            raise RuntimeError(f"{name}: dirty checkout; preserve and resolve manually")
        if git("remote", "get-url", "origin", cwd=target).rstrip("/") != spec["url"].rstrip("/"):
            raise RuntimeError(f"{name}: unexpected remote")
    else:
        target.mkdir(parents=True)
        git("init", str(target))
        git("remote", "add", "origin", spec["url"], cwd=target)
    git("fetch", "--depth", "1", "origin", spec["commit"], cwd=target)
    git("checkout", "--detach", spec["commit"], cwd=target)
    actual = git("rev-parse", "HEAD", cwd=target)
    if actual != spec["commit"]:
        raise RuntimeError("reference verification failed")
    print(f"{name}: {actual}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="default: all locked references")
    parser.add_argument("--only", action="append", default=[], help="fetch one named reference")
    args = parser.parse_args()
    specs = yaml.safe_load((ROOT / "references.lock.yaml").read_text(encoding="utf-8"))["repositories"]
    names = args.names or args.only or list(specs)
    for name in names:
        if name not in specs:
            parser.error(f"unknown reference: {name}")
        fetch(name, specs[name])


if __name__ == "__main__":
    main()
