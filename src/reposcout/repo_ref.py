"""Parse and validate a GitHub repository reference ("owner/name" or a github.com URL)."""

from __future__ import annotations

import re
from dataclasses import dataclass

# GitHub rules: owner 1-39 chars of letters, digits, hyphens (no leading hyphen);
# repository names allow letters, digits, '.', '_', '-' (but not '.' or '..').
_OWNER = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})"
_NAME = r"[A-Za-z0-9._-]{1,100}"
_PATTERN = re.compile(
    rf"^\s*(?:(?:https?://)?(?:www\.)?github\.com/)?(?P<owner>{_OWNER})/(?P<name>{_NAME}?)"
    r"(?:\.git)?(?:/(?:tree|blob)/[^\s]*)?/?\s*$"
)


class InvalidRepo(ValueError):
    """The text is not a valid GitHub repository reference."""


@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def html_url(self) -> str:
        return f"https://github.com/{self.full_name}"

    def __str__(self) -> str:
        return self.full_name

    def same_as(self, owner: str | None, name: str | None) -> bool:
        return (owner or "").lower() == self.owner.lower() and (
            name or ""
        ).lower() == self.name.lower()


def parse_repo(text: str) -> RepoRef:
    """Accepts `owner/name`, `https://github.com/owner/name(.git)` and `.../tree/<branch>` URLs."""
    match = _PATTERN.match(text or "")
    if not match:
        raise InvalidRepo(
            f"Not a GitHub repository: {text!r}. Use 'owner/name' or a github.com URL."
        )
    name = match.group("name")
    if name.endswith(".git"):
        name = name[:-4]
    if name in {".", ".."} or not name:
        raise InvalidRepo(f"Not a GitHub repository: {text!r}")
    return RepoRef(owner=match.group("owner"), name=name)
