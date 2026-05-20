"""Markdown image embedding helpers for portable report files."""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path


IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)]+)\)")


def image_to_data_uri(image_path: Path) -> str:
    """Return a base64 data URI for a local image path."""
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None:
        mime = "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _split_markdown_target(target: str) -> tuple[str, str]:
    """Split a Markdown image target into path and optional title suffix."""
    target = target.strip()
    if target.startswith("<"):
        end = target.find(">")
        if end != -1:
            return target[1:end], target[end + 1 :].strip()
    for quote in ('"', "'"):
        marker = f" {quote}"
        if marker in target and target.endswith(quote):
            path, title = target.split(marker, 1)
            return path.strip(), f'{quote}{title}'
    return target, ""


def embed_markdown_images(report_path: Path | str, output_path: Path | str | None = None) -> Path:
    """Replace local markdown image links in *report_path* with data URIs.

    Remote URLs and already embedded data URIs are left unchanged. By default the
    source markdown file is rewritten in place.
    """
    report_path = Path(report_path)
    output_path = Path(output_path) if output_path is not None else report_path
    report_dir = report_path.parent
    text = report_path.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        alt = match.group("alt")
        target, suffix = _split_markdown_target(match.group("target"))
        if target.startswith(("http://", "https://", "data:", "#")):
            return match.group(0)
        image_path = (report_dir / target).resolve()
        if not image_path.exists() or not image_path.is_file():
            return match.group(0)
        title = f" {suffix}" if suffix else ""
        return f"![{alt}]({image_to_data_uri(image_path)}{title})"

    output_path.write_text(IMAGE_PATTERN.sub(replace, text), encoding="utf-8")
    return output_path
