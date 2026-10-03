from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class IdentityLoadError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class IdentitySection:
    name: str
    content: str


@dataclass(frozen=True, slots=True)
class IdentityBundle:
    sections: tuple[IdentitySection, ...]

    @property
    def rendered(self) -> str:
        return "\n\n".join(
            f"## Identity: {section.name}\n\n{section.content}"
            for section in self.sections
        )

    @property
    def file_count(self) -> int:
        return len(self.sections)


class IdentityLoader:
    """Loads trusted, version-controlled identity files deterministically."""

    def __init__(self, identity_dir: Path) -> None:
        self.identity_dir = identity_dir

    def load(self) -> IdentityBundle:
        if not self.identity_dir.is_dir():
            raise IdentityLoadError(f"Identity directory not found: {self.identity_dir}")

        files = sorted(self.identity_dir.glob("*.md"), key=lambda path: path.name)
        if not files:
            raise IdentityLoadError("No identity Markdown files found")

        sections: list[IdentitySection] = []
        for path in files:
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                raise IdentityLoadError(f"Identity file is empty: {path.name}")
            sections.append(IdentitySection(name=path.stem, content=content))
        return IdentityBundle(sections=tuple(sections))
