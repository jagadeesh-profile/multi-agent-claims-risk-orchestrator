"""Guards against committing API-key-shaped secrets."""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEMINI_KEY_PREFIX = "AI" + "za"
SKIPPED_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "logs",
    "models",
    "pytest-cache-files-2ozu0sm5",
    "pytest-cache-files-rbiykqc8",
    "tmp",
    "venv",
}
SKIPPED_FILES = {".env"}


def _project_text_files() -> list[Path]:
    text_files: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        relative_parts = set(path.relative_to(PROJECT_ROOT).parts)
        if path.is_dir() or relative_parts & SKIPPED_DIRS or path.name in SKIPPED_FILES:
            continue
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        text_files.append(path)
    return text_files


def test_project_files_do_not_contain_gemini_api_keys() -> None:
    offenders: list[str] = []
    for path in _project_text_files():
        text = path.read_text(encoding="utf-8")
        if GEMINI_KEY_PREFIX in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []
