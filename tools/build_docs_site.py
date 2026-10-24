#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Wireless System Research Group, McMaster University

"""Stage the HTML guides for GitHub Pages, retaining readable repository links."""

from __future__ import annotations

import argparse
from html import escape, unescape
from pathlib import Path
import re
import shutil
from urllib.parse import quote, unquote, urlsplit, urlunsplit

REPOSITORY_URL = "https://github.com/wisermaclab/hermes"
LINK_ATTRIBUTE = re.compile(
    r"(?P<prefix><(?:a|link|img|script)\b[^>]*?\b(?:href|src)\s*=\s*)"
    r"(?P<quote>[\"'])(?P<url>[^\"']*)(?P=quote)",
    re.IGNORECASE,
)
SITE_ASSETS = {".css", ".js", ".json", ".xml", ".svg", ".png", ".woff", ".woff2"}


def build_site(root: Path, output: Path) -> int:
    """Copy HTML and linked assets; send source and Markdown links to GitHub."""
    root = root.resolve()
    output = output.resolve()
    if output == root or root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError("Site output must be outside the source checkout")
    pages = sorted((root / "docs").rglob("*.html"))
    if not pages or not (root / "docs/tutorial/index.html").is_file():
        raise ValueError("The documentation source has no tutorial entry page")
    output.mkdir(parents=True, exist_ok=False)

    def rewrite_url(source: Path, value: str) -> str:
        parsed = urlsplit(unescape(value))
        if parsed.scheme or parsed.netloc or not parsed.path:
            return value
        target = (source.parent / unquote(parsed.path)).resolve()
        relative = target.relative_to(root)
        if not target.exists():
            raise ValueError(f"Missing documentation link: {source.relative_to(root)}: {value}")
        if target.suffix == ".html" and target in pages:
            return value
        if target.is_file() and target.suffix in SITE_ASSETS:
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, destination)
            return value
        kind = "tree" if target.is_dir() else "blob"
        url = f"{REPOSITORY_URL}/{kind}/main/{quote(relative.as_posix(), safe='/')}"
        return escape(urlunsplit((*urlsplit(url)[:3], parsed.query, parsed.fragment)), quote=True)

    for source in pages:

        def replace_link(match: re.Match[str], page: Path = source) -> str:
            value = rewrite_url(page, match["url"])
            return f"{match['prefix']}{match['quote']}{value}{match['quote']}"

        destination = output / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            LINK_ATTRIBUTE.sub(replace_link, source.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="refresh" content="0;url=docs/tutorial/">'
        "<title>HERMES documentation</title></head><body>"
        '<p><a href="docs/tutorial/">Open the HERMES tutorial</a></p></body></html>\n',
        encoding="utf-8",
    )
    (output / ".nojekyll").touch()
    return len(pages)


def main() -> int:
    """Build the public documentation site in a new output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New site output directory")
    args = parser.parse_args()
    try:
        count = build_site(Path(__file__).resolve().parents[1], args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Staged {count} HTML pages in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
