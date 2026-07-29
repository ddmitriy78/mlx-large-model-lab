from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TEXT_SUFFIXES = {
    "",
    ".css",
    ".html",
    ".json",
    ".jsonc",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PUBLIC_TEXT_PATHS = (
    ROOT / ".github",
    ROOT / "configs",
    ROOT / "docs",
    ROOT / "prompts",
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "tests",
    ROOT / ".gitignore",
    ROOT / "CONTRIBUTING.md",
    ROOT / "PLANS.md",
    ROOT / "README.md",
    ROOT / "SECURITY.md",
    ROOT / "opencode.jsonc",
    ROOT / "pyproject.toml",
)
POSIX_HOME_ROOTS = ("/" + "Users" + "/", "/" + "home" + "/")
CONTRIBUTOR_HOME_PATTERN = re.compile(
    rf"(?:{re.escape(POSIX_HOME_ROOTS[0])}|"
    rf"{re.escape(POSIX_HOME_ROOTS[1])})"
    r"[^/\\\s\"'<>]+(?:[/\\]|$)"
    r"|[A-Za-z]:\\Users\\[^/\\\s\"'<>]+(?:[/\\]|$)"
)


def public_text_files() -> list[Path]:
    files: list[Path] = []
    for path in PUBLIC_TEXT_PATHS:
        candidates = path.rglob("*") if path.is_dir() else (path,)
        files.extend(
            candidate
            for candidate in candidates
            if candidate.is_file()
            and candidate.suffix.lower() in PUBLIC_TEXT_SUFFIXES
        )
    return files


class PublicHygieneTests(unittest.TestCase):
    def test_public_text_has_no_contributor_home_paths(self) -> None:
        matches: list[str] = []
        for path in public_text_files():
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if CONTRIBUTOR_HOME_PATTERN.search(line):
                    matches.append(f"{path.relative_to(ROOT)}:{line_number}")

        self.assertEqual(matches, [], f"Contributor home paths found: {matches}")

    def test_opencode_model_ids_are_portable_aliases(self) -> None:
        text = (ROOT / "opencode.jsonc").read_text(encoding="utf-8")
        model_ids = re.findall(r'"id"\s*:\s*"([^"]+)"', text)

        self.assertGreater(len(model_ids), 1)
        self.assertTrue(all("/" not in model_id for model_id in model_ids))
        self.assertTrue(all("\\" not in model_id for model_id in model_ids))


if __name__ == "__main__":
    unittest.main()
