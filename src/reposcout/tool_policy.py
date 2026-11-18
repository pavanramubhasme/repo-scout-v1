"""Everything between "the LLM asked for a tool call" and "the MCP server executes it".

Small models make predictable mistakes (wrong enum case, numbers as strings, invented
parameters), and any model can be steered by text it reads in a repository (indirect
prompt injection). This layer handles both in code, deterministically:

1. Scope by construction: `owner`/`repo` are removed from the schemas the LLM sees and
   filled in here from the request. The model cannot even express "read another repo".
2. Search queries may not carry `repo:`/`org:`/`user:` qualifiers. Passing owner/repo is
   NOT a scope boundary on its own: the GitHub MCP server (v2.2.0, prepareSearchArgs in
   pkg/github/search_utils.go) only adds `repo:owner/name` when the query has no `repo:`
   qualifier yet, so an injected `repo:victim/private` would replace the scope.
3. Only tools marked read-only by their server are registered at all.
4. Argument repair: enum case, "42" -> 42, "true" -> True, unknown parameters dropped.
5. Defaults that keep results small: 10 items per page, compact field lists.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool

from .repo_ref import RepoRef

SCOPE_PARAMS = ("owner", "repo")
# Parameters the LLM never sees: the scope (injected) and knobs that only add schema noise.
HIDDEN_PARAMS = {"owner", "repo", "fields", "field_filters", "after"}
_SCOPE_QUALIFIER = re.compile(r"(?i)(?<![\w-])(?:repo|org|user|owner)\s*:")

DEFAULT_ARGS: dict[str, dict[str, Any]] = {
    "list_issues": {
        "perPage": 10,
        "fields": ["number", "title", "state", "labels", "comments", "created_at", "updated_at"],
    },
    "search_issues": {
        "perPage": 10,
        "fields": [
            "number",
            "title",
            "state",
            "html_url",
            "labels",
            "comments",
            "created_at",
            "closed_at",
            "pull_request",
        ],
    },
    "list_pull_requests": {
        "perPage": 10,
        "fields": [
            "number",
            "title",
            "state",
            "merged",
            "html_url",
            "user",
            "created_at",
            "merged_at",
            "closed_at",
        ],
    },
    "list_commits": {"perPage": 10, "fields": ["sha", "html_url", "commit"]},
    "list_releases": {
        "perPage": 5,
        "fields": ["tag_name", "name", "html_url", "published_at", "prerelease"],
    },
    "issue_read": {"perPage": 10},
    "pull_request_read": {"perPage": 30},
}


class PolicyViolation(Exception):
    """The call is refused; the message is returned to the model so it can correct itself."""


@dataclass
class ToolSpec:
    """One MCP tool as the agent knows it."""

    name: str
    server: str
    tool: BaseTool
    schema: dict[str, Any]
    llm_schema: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.llm_schema = slim_schema(self.schema)

    @property
    def openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.tool.description,
                "parameters": self.llm_schema,
            },
        }


def is_read_only(tool: BaseTool) -> bool:
    """MCP servers mark read-only tools with the `readOnlyHint` annotation."""
    return bool((tool.metadata or {}).get("readOnlyHint"))


def slim_schema(schema: dict[str, Any]) -> dict[str, Any]:
    slim = copy.deepcopy(schema) if schema else {"type": "object", "properties": {}}
    props = slim.setdefault("properties", {})
    for name in HIDDEN_PARAMS:
        props.pop(name, None)
    if "required" in slim:
        slim["required"] = [r for r in slim["required"] if r not in HIDDEN_PARAMS]
    slim.pop("$schema", None)
    slim.pop("title", None)
    return slim


def _repair(value: Any, prop: dict[str, Any]) -> Any:
    enum = prop.get("enum")
    if enum and isinstance(value, str) and value not in enum:
        for option in enum:
            if isinstance(option, str) and option.lower() == value.strip().lower():
                return option
    kind = prop.get("type")
    if (
        kind in ("number", "integer")
        and isinstance(value, str)
        and re.fullmatch(r"\s*#?-?\d+\s*", value)
    ):
        return int(value.strip().lstrip("#"))
    if kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    if kind == "array" and isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


def prepare_args(
    spec: ToolSpec, raw_args: dict[str, Any] | None, repo: RepoRef
) -> tuple[dict[str, Any], list[str]]:
    """Return the arguments to send to the MCP server, plus notes on what was changed."""
    props: dict[str, Any] = spec.schema.get("properties", {})
    notes: list[str] = []
    args: dict[str, Any] = {}
    for key, value in (raw_args or {}).items():
        if value is None:
            continue
        if key in SCOPE_PARAMS:
            if not repo.same_as(raw_args.get("owner", repo.owner), raw_args.get("repo", repo.name)):
                notes.append(f"ignored model-chosen {key}={value!r}")
            continue
        if key not in props or key in HIDDEN_PARAMS:
            notes.append(f"dropped unknown argument {key!r}")
            continue
        fixed = _repair(value, props[key])
        if fixed != value:
            notes.append(f"repaired {key}: {value!r} -> {fixed!r}")
        args[key] = fixed

    for key in ("query", "q"):
        if key in args and _SCOPE_QUALIFIER.search(str(args[key])):
            raise PolicyViolation(
                f"Search qualifiers such as repo:, org: or user: are not allowed. "
                f"The search is already limited to {repo}. Search with plain words."
            )

    for key, value in DEFAULT_ARGS.get(spec.name, {}).items():
        if key in props and key not in args:
            args[key] = copy.deepcopy(value)

    if "owner" in props:
        args["owner"] = repo.owner
    if "repo" in props:
        args["repo"] = repo.name

    missing = [r for r in spec.schema.get("required", []) if r not in args]
    if missing:
        raise PolicyViolation(
            f"Missing required argument(s) for {spec.name}: {', '.join(missing)}."
        )
    return args, notes


def call_signature(name: str, args: dict[str, Any]) -> str:
    """Stable key to spot a repeated identical call (a common small-model loop)."""
    items = sorted((k, repr(v)) for k, v in args.items())
    return f"{name}:{items}"
