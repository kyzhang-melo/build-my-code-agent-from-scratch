"""Shared stdlib-only file tool semantics for local and container backends."""

from __future__ import annotations

import fnmatch
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

MAX_LINE_CHARS = 2000
MAX_READ_BYTES = 40_000
MAX_READ_FILE_BYTES = 5 * 1024 * 1024
EXCLUDE_DIRS = {".git", ".svn", ".hg", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".sessions", ".transcripts", ".ssh"}
BROAD_GLOB_PATTERNS = {"**", "**/", "**/*", "**/**"}
SENSITIVE_NAMES = {".env", ".netrc", ".npmrc", ".pypirc", "credentials", "id_dsa", "id_ecdsa", "id_ed25519", "id_rsa"}
SENSITIVE_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".tiff", ".pdf",
    ".zip", ".gz", ".bz2", ".xz", ".tar", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".class", ".jar",
    ".pyc", ".pyo", ".wasm", ".node",
    ".mp3", ".wav", ".flac", ".ogg", ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".sqlite", ".sqlite3", ".db", ".woff", ".woff2", ".ttf", ".otf",
}


def _sensitive(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    parts = {part.lower() for part in relative.parts}
    name = path.name.lower()
    return name in SENSITIVE_NAMES or name.startswith(".env.") or path.suffix.lower() in SENSITIVE_SUFFIXES or bool(parts & {".git", ".ssh", ".sessions", ".transcripts"})


def _looks_binary(path: Path) -> bool:
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    with path.open("rb") as handle:
        return b"\0" in handle.read(4096)


@dataclass(frozen=True)
class FileEngine:
    root: Path
    restrict_to_root: bool = False
    max_line_chars: int = MAX_LINE_CHARS
    max_read_bytes: int = MAX_READ_BYTES
    max_read_file_bytes: int = MAX_READ_FILE_BYTES

    def _resolve(self, value: str, *, allow_external_absolute: bool = False) -> Path:
        root = self.root.resolve()
        raw = Path(value).expanduser()
        target = (root / raw).resolve()
        if not target.is_relative_to(root) and (
            self.restrict_to_root or not (allow_external_absolute and raw.is_absolute())
        ):
            raise ValueError(f"Path escapes workspace: {value}")
        return target

    def read(self, path: str, offset: int = 1, limit: int = 1000) -> str:
        try:
            target = self._resolve(path, allow_external_absolute=True)
            if _sensitive(target, self.root.resolve()):
                return f"Error: Access to sensitive path is blocked: {path}"
            if not target.exists():
                return f"Error: File not found: {path}"
            if not target.is_file():
                return f"Error: Not a file: {path}"
            if _looks_binary(target):
                return f"Error: '{path}' appears to be a binary or non-text file. Use appropriate tools to inspect it."
            count_to_eof = target.stat().st_size <= self.max_read_file_bytes
            rendered, truncated = [], []
            rendered_bytes = total = 0
            last = offset - 1
            max_lines = max_bytes = False
            reached_eof = True
            with target.open(encoding="utf-8", errors="replace") as handle:
                for number, raw_line in enumerate(handle, 1):
                    total = number
                    if number < offset:
                        continue
                    line_cap = len(rendered) >= limit
                    byte_cap = rendered_bytes >= self.max_read_bytes
                    if line_cap or byte_cap:
                        max_lines |= line_cap
                        max_bytes |= byte_cap
                        if count_to_eof:
                            continue
                        reached_eof = False
                        break
                    content = raw_line.rstrip("\n")
                    if len(content) > self.max_line_chars:
                        content = content[:self.max_line_chars] + " [...line truncated]"
                        truncated.append(number)
                    line = f"{number}\t{content}"
                    rendered.append(line)
                    rendered_bytes += len(line)
                    last = number
            if not rendered:
                if total == 0:
                    return "<system-reminder>File exists but is empty.</system-reminder>"
                return f"<system-reminder>File has {total} lines; offset {offset} is past the end of the file.</system-reminder>"
            total_part = f" Total lines: {total}." if reached_eof else f" Total lines: {total}+ (not fully counted)."
            notes = []
            if max_lines: notes.append(f"Stopped at the {limit}-line limit")
            elif max_bytes: notes.append(f"Stopped at the {self.max_read_bytes}-byte limit")
            elif reached_eof and len(rendered) < limit: notes.append("End of file")
            if truncated: notes.append(f"lines {truncated} truncated to {self.max_line_chars} chars")
            if not count_to_eof: notes.append("large file -- use grep or a targeted offset/limit to narrow the read")
            note_part = (" " + "; ".join(notes) + ".") if notes else ""
            more = max_lines or max_bytes or not reached_eof
            hint = f" Use offset={last + 1} to continue." if more else ""
            footer = f"[Read {len(rendered)} lines (lines {offset}-{last}).{total_part}{note_part}{hint}]"
            return "\n".join(rendered) + "\n\n" + footer
        except Exception as exc:
            return f"Error: {exc}"

    def write(self, path: str, content: str, mode: str = "overwrite") -> str:
        try:
            if mode not in {"overwrite", "append"}:
                return "Error: mode must be 'overwrite' or 'append'"
            target = self._resolve(path)
            if _sensitive(target, self.root.resolve()):
                return f"Error: Access to sensitive path is blocked: {path}"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w" if mode == "overwrite" else "a", encoding="utf-8", newline="") as handle:
                handle.write(content)
            size = len(content.encode("utf-8"))
            verb = "Appended" if mode == "append" else "Wrote"
            return f"{verb} {size} bytes to {path} (current size: {target.stat().st_size} bytes)"
        except Exception as exc:
            return f"Error: {exc}"

    def edit(self, path: str, edits: list[dict]) -> str:
        try:
            if not edits:
                return "Error: edits must contain at least one replacement"
            target = self._resolve(path)
            if _sensitive(target, self.root.resolve()):
                return f"Error: Access to sensitive path is blocked: {path}"
            if not target.exists():
                return f"Error: File not found: {path}. Use write_file to create it first."
            if not target.is_file():
                return f"Error: Not a file: {path}"
            raw = target.read_bytes().decode("utf-8", errors="replace")
            bom = "\ufeff" if raw.startswith("\ufeff") else ""
            body = raw[len(bom):]
            ending = "\r\n" if "\r\n" in body[:body.find("\n") + 1] else "\n"
            normalized = body.replace("\r\n", "\n").replace("\r", "\n")
            edited = _apply_edits(normalized, edits, path)
            final = bom + (edited.replace("\n", "\r\n") if ending == "\r\n" else edited)
            target.write_text(final, encoding="utf-8", newline="")
            return f"Edited {path}: applied {len(edits)} replacement(s)"
        except Exception as exc:
            return f"Error: {exc}"

    def glob(self, pattern: str, directory: str | None = None, include_dirs: bool = True, limit: int = 1000) -> str:
        try:
            base = self._resolve(directory or ".", allow_external_absolute=True)
            if not base.exists(): return f"Error: Directory not found: {directory or '.'}"
            if not base.is_dir(): return f"Error: Not a directory: {directory or '.'}"
            if _sensitive(base, self.root.resolve()): return f"Error: Access to sensitive directory is blocked: {directory or '.'}"
            if pattern in BROAD_GLOB_PATTERNS:
                entries = [f"{item.name}/" if item.is_dir() else item.name for item in sorted(base.iterdir()) if item.name not in EXCLUDE_DIRS and not _sensitive(item.resolve(), self.root.resolve())]
                body = "\n".join(entries) if entries else "(empty)"
                return f"Error: pattern `{pattern}` matches everything and is too broad. Use a more specific recursive pattern such as `**/*.py`, `**/*.md`, `src/**/*.ts`, or `tests/**/*_test.py`. If you need a shallow directory overview, use pattern `*` with a specific directory. Top-level entries of `{base}` (directories marked with `/`):\n{body}"
            if pattern.startswith("**/") and "/" not in pattern[3:]:
                matches = []
                for current, dirs, files in os.walk(base):
                    dirs[:] = [name for name in dirs if name not in EXCLUDE_DIRS and not _sensitive((Path(current) / name).resolve(), self.root.resolve())]
                    for name in files + (dirs if include_dirs else []):
                        if fnmatch.fnmatch(name, pattern[3:]): matches.append(Path(current) / name)
            else:
                matches = list(base.glob(pattern))
                if not include_dirs: matches = [item for item in matches if item.is_file()]
            matches = [item for item in matches if not any(part in EXCLUDE_DIRS for part in item.relative_to(base).parts) and not _sensitive(item.resolve(), self.root.resolve())]
            matches.sort()
            if not matches: return f"No matches found for pattern `{pattern}`."
            header = f"Found {len(matches)} matches for pattern `{pattern}`."
            if len(matches) > limit: header += f" Showing first {limit}."
            return "\n".join([header, *(str(item.relative_to(base)) for item in matches[:limit])])
        except Exception as exc:
            return f"Error: {exc}"


@dataclass(frozen=True)
class _Match:
    index: int
    length: int
    replacement: str
    edit_index: int


_TRANSLATION = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2010": "-", "\u2011": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " ", "\u2002": " ", "\u2003": " ", "\u2009": " ", "\u202f": " ", "\u3000": " "})


def _fuzzy(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "\n".join(line.rstrip() for line in normalized.split("\n")).translate(_TRANSLATION)


def _occurrences(content: str, needle: str) -> list[int]:
    positions, start = [], 0
    while True:
        found = content.find(needle, start)
        if found < 0: return positions
        positions.append(found)
        start = found + len(needle)


def _split_lines_with_endings(content: str) -> list[str]:
    return re.findall(r"[^\n]*\n|[^\n]+", content)


def _line_spans(content: str) -> list[tuple[int, int]]:
    offset = 0
    spans = []
    for line in _split_lines_with_endings(content):
        spans.append((offset, offset + len(line)))
        offset += len(line)
    return spans


def _match_line_range(
    spans: list[tuple[int, int]], match: _Match,
) -> tuple[int, int]:
    match_end = match.index + match.length
    start_line = next(
        (index for index, (start, end) in enumerate(spans) if start <= match.index < end),
        -1,
    )
    if start_line == -1:
        raise ValueError("Replacement range is outside the file content")
    end_line = start_line
    while end_line < len(spans) and spans[end_line][1] < match_end:
        end_line += 1
    if end_line >= len(spans):
        raise ValueError("Replacement range is outside the file content")
    return start_line, end_line + 1


def _replace_matches(
    content: str, matches: list[_Match], *, offset: int = 0,
) -> str:
    result = content
    for match in reversed(matches):
        index = match.index - offset
        result = result[:index] + match.replacement + result[index + match.length:]
    return result


def _replace_fuzzy_matches_preserving_lines(
    original_content: str,
    fuzzy_content: str,
    matches: list[_Match],
) -> str:
    """Rewrite touched lines in fuzzy space and copy untouched lines verbatim."""
    original_lines = _split_lines_with_endings(original_content)
    spans = _line_spans(fuzzy_content)
    if len(original_lines) != len(spans):
        raise ValueError("Fuzzy normalization changed the file's line structure")

    groups: list[dict] = []
    for match in sorted(matches, key=lambda item: item.index):
        start_line, end_line = _match_line_range(spans, match)
        current = groups[-1] if groups else None
        if current is not None and start_line < current["end_line"]:
            current["end_line"] = max(current["end_line"], end_line)
            current["matches"].append(match)
        else:
            groups.append({
                "start_line": start_line,
                "end_line": end_line,
                "matches": [match],
            })

    result = []
    original_line_index = 0
    for group in groups:
        start_line = group["start_line"]
        end_line = group["end_line"]
        result.extend(original_lines[original_line_index:start_line])
        group_start = spans[start_line][0]
        group_end = spans[end_line - 1][1]
        result.append(_replace_matches(
            fuzzy_content[group_start:group_end],
            group["matches"],
            offset=group_start,
        ))
        original_line_index = end_line
    result.extend(original_lines[original_line_index:])
    return "".join(result)


def _apply_edits(content: str, edits: list[dict], path: str) -> str:
    normalized = [{"old_text": item["old_text"].replace("\r\n", "\n").replace("\r", "\n"), "new_text": item["new_text"].replace("\r\n", "\n").replace("\r", "\n")} for item in edits]
    use_fuzzy = any(item["old_text"] not in content and _fuzzy(item["old_text"]) in _fuzzy(content) for item in normalized)
    match_content = _fuzzy(content) if use_fuzzy else content
    matches = []
    for index, item in enumerate(normalized):
        needle = _fuzzy(item["old_text"]) if use_fuzzy else item["old_text"]
        positions = _occurrences(match_content, needle)
        if not positions: raise ValueError(f"Could not find edits[{index}].old_text in {path}. Re-read the file and copy the target text, including its whitespace and newlines.")
        if len(positions) > 1: raise ValueError(f"Found {len(positions)} occurrences of edits[{index}].old_text in {path}. Add surrounding context so the target is unique.")
        matches.append(_Match(positions[0], len(needle), item["new_text"], index))
    matches.sort(key=lambda item: item.index)
    for left, right in zip(matches, matches[1:]):
        if left.index + left.length > right.index: raise ValueError(f"edits[{left.edit_index}] and edits[{right.edit_index}] overlap in {path}. Merge them into one edit or target disjoint regions.")
    result = (
        _replace_fuzzy_matches_preserving_lines(content, match_content, matches)
        if use_fuzzy
        else _replace_matches(match_content, matches)
    )
    if result == content: raise ValueError(f"No changes made to {path}. The replacements produced identical content.")
    return result
