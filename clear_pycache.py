#!/usr/bin/env python3
"""Clear Python cache files and empty directories.

Removes directories named __pycache__, files ending with .pyc or .pyo,
and then removes empty directories. Skips `.git` directories.

Usage:
  python clear_pycache.py [--dry-run] [--yes]

Options:
  --dry-run   Print actions without deleting anything.
  --yes       Do not prompt for confirmation.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable


def iter_walk(root: Path) -> Iterable[tuple[str, list[str], list[str]]]:
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        yield dirpath, dirnames, filenames


def is_in_git(path: str) -> bool:
    parts = Path(path).parts
    return ".git" in parts


def clear_pycache(root: Path, dry_run: bool = False) -> tuple[int, int, int]:
    removed_dirs = 0
    removed_files = 0
    removed_empty_dirs = 0

    for dirpath, dirnames, filenames in iter_walk(root):
        if is_in_git(dirpath):
            continue

        # remove __pycache__ dirs if present
        for d in list(dirnames):
            if d == "__pycache__":
                target = Path(dirpath) / d
                if dry_run:
                    print(f"Would remove directory: {target}")
                else:
                    try:
                        shutil.rmtree(target)
                        print(f"Removed directory: {target}")
                    except Exception as e:
                        print(f"Failed to remove {target}: {e}")
                removed_dirs += 1

        # remove compiled files
        for f in list(filenames):
            if f.endswith((".pyc", ".pyo")):
                target = Path(dirpath) / f
                if dry_run:
                    print(f"Would remove file: {target}")
                else:
                    try:
                        target.unlink()
                        print(f"Removed file: {target}")
                    except Exception as e:
                        print(f"Failed to remove {target}: {e}")
                removed_files += 1

        # remove empty directories (after other removals)
        p = Path(dirpath)
        if p == root:
            continue
        if is_in_git(str(p)):
            continue
        try:
            if not any(p.iterdir()):
                if dry_run:
                    print(f"Would remove empty directory: {p}")
                else:
                    p.rmdir()
                    print(f"Removed empty directory: {p}")
                removed_empty_dirs += 1
        except Exception:
            # skip directories we can't access or that changed
            pass

    return removed_dirs, removed_files, removed_empty_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear Python __pycache__ and compiled files")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without performing them")
    parser.add_argument("--yes", action="store_true", help="Do not prompt for confirmation")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    if not args.yes and not args.dry_run:
        resp = input(f"This will remove __pycache__ dirs, .pyc/.pyo files and empty dirs under {root}. Continue? [y/N] ")
        if resp.lower() not in ("y", "yes"):
            print("Aborted.")
            return

    removed_dirs, removed_files, removed_empty_dirs = clear_pycache(root, dry_run=args.dry_run)

    print(f"Done. Directories removed: {removed_dirs}, files removed: {removed_files}, empty dirs removed: {removed_empty_dirs}")


if __name__ == "__main__":
    main()
