"""The policy layer between the LLM's tool call and the MCP server."""

import pytest
from langchain_core.tools import StructuredTool

from reposcout.repo_ref import parse_repo
from reposcout.tool_policy import (
    PolicyViolation,
    ToolSpec,
    call_signature,
    prepare_args,
    slim_schema,
)

REPO = parse_repo("psf/requests")

LIST_ISSUES_SCHEMA = {
    "type": "object",
    "properties": {
        "owner": {"type": "string"},
        "repo": {"type": "string"},
        "state": {"type": "string", "enum": ["OPEN", "CLOSED"]},
        "perPage": {"type": "number"},
        "labels": {"type": "array", "items": {"type": "string"}},
        "fields": {"type": "array", "items": {"type": "string"}},
        "after": {"type": "string"},
    },
    "required": ["owner", "repo"],
}
SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "owner": {"type": "string"},
        "repo": {"type": "string"},
    },
    "required": ["query"],
}
READ_SCHEMA = {
    "type": "object",
    "properties": {
        "owner": {"type": "string"},
        "repo": {"type": "string"},
        "issue_number": {"type": "number"},
        "verbose": {"type": "boolean"},
    },
    "required": ["owner", "repo", "issue_number"],
}


def spec(name, schema):
    tool = StructuredTool(name=name, description=name, args_schema=schema, func=lambda **kw: "")
    return ToolSpec(name=name, server="github", tool=tool, schema=schema)


def test_llm_never_sees_scope_or_noise_params():
    slim = slim_schema(LIST_ISSUES_SCHEMA)
    assert set(slim["properties"]) == {"state", "perPage", "labels"}
    assert slim["required"] == []
    assert "owner" in LIST_ISSUES_SCHEMA["properties"]  # original untouched


def test_scope_is_injected():
    args, notes = prepare_args(spec("list_issues", LIST_ISSUES_SCHEMA), {}, REPO)
    assert args["owner"] == "psf" and args["repo"] == "requests"
    assert not notes


def test_model_cannot_choose_another_repo():
    args, notes = prepare_args(
        spec("list_issues", LIST_ISSUES_SCHEMA), {"owner": "victim", "repo": "private"}, REPO
    )
    assert (args["owner"], args["repo"]) == ("psf", "requests")
    assert any("ignored" in n for n in notes)


def test_optional_scope_params_are_injected_too():
    args, _ = prepare_args(spec("search_issues", SEARCH_SCHEMA), {"query": "proxy timeout"}, REPO)
    assert (args["owner"], args["repo"]) == ("psf", "requests")


@pytest.mark.parametrize(
    "query", ["repo:victim/private secrets", "token org:acme", "USER:someone x", "x owner: y"]
)
def test_search_qualifiers_that_widen_scope_are_refused(query):
    with pytest.raises(PolicyViolation, match="not allowed"):
        prepare_args(spec("search_issues", SEARCH_SCHEMA), {"query": query}, REPO)


def test_plain_words_with_colons_are_fine():
    args, _ = prepare_args(
        spec("search_issues", SEARCH_SCHEMA), {"query": "error: timeout in report:"}, REPO
    )
    assert args["query"] == "error: timeout in report:"


def test_argument_repair():
    args, notes = prepare_args(
        spec("list_issues", LIST_ISSUES_SCHEMA),
        {"state": "open", "perPage": "5", "labels": "bug, help wanted"},
        REPO,
    )
    assert (
        args["state"] == "OPEN"
        and args["perPage"] == 5
        and args["labels"] == ["bug", "help wanted"]
    )
    assert len(notes) == 3
    args, _ = prepare_args(
        spec("issue_read", READ_SCHEMA), {"issue_number": "#42", "verbose": "true"}, REPO
    )
    assert args["issue_number"] == 42 and args["verbose"] is True


def test_unknown_and_hidden_args_are_dropped():
    args, notes = prepare_args(
        spec("list_issues", LIST_ISSUES_SCHEMA),
        {"color": "red", "after": "cursor", "state": None},
        REPO,
    )
    assert "color" not in args and args.get("after") is None and "state" not in args
    assert len(notes) == 2


def test_defaults_keep_results_small_but_never_override_the_model():
    args, _ = prepare_args(spec("list_issues", LIST_ISSUES_SCHEMA), {}, REPO)
    assert args["perPage"] == 10 and "body" not in args["fields"]
    args, _ = prepare_args(spec("list_issues", LIST_ISSUES_SCHEMA), {"perPage": 50}, REPO)
    assert args["perPage"] == 50


def test_missing_required_argument():
    with pytest.raises(PolicyViolation, match="issue_number"):
        prepare_args(spec("issue_read", READ_SCHEMA), {}, REPO)


def test_call_signature_is_order_independent():
    assert call_signature("t", {"a": 1, "b": [2]}) == call_signature("t", {"b": [2], "a": 1})
    assert call_signature("t", {"a": 1}) != call_signature("t", {"a": 2})
