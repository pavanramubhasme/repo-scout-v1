from reposcout.citations import dedupe, sources_for_call
from reposcout.repo_ref import parse_repo

REPO = parse_repo("psf/requests")
BASE = "https://github.com/psf/requests"


def test_read_file_links_to_exact_lines():
    out = "[psf/requests @ main abc1234] src/requests/adapters.py lines 75-85 of 748\n   75| x"
    [source] = sources_for_call("read_file", {"path": "src/requests/adapters.py"}, REPO, out)
    assert source["url"] == f"{BASE}/blob/HEAD/src/requests/adapters.py#L75-L85"
    assert source["label"] == "src/requests/adapters.py:L75-L85"


def test_grep_links_to_first_files():
    out = "[psf/requests @ main abc] 3 matches in 2 files\nsrc/a.py:10: x\nsrc/a.py:12: y\nsrc/b.py:3: z"
    sources = sources_for_call("grep_code", {"query": "x"}, REPO, out)
    assert [s["label"] for s in sources] == ["src/a.py:L10", "src/b.py:L3"]


def test_github_objects():
    assert (
        sources_for_call("issue_read", {"issue_number": 7}, REPO, "")[0]["url"]
        == f"{BASE}/issues/7"
    )
    assert (
        sources_for_call("pull_request_read", {"pullNumber": 9}, REPO, "")[0]["url"]
        == f"{BASE}/pull/9"
    )
    assert (
        sources_for_call("get_commit", {"sha": "abcdef123456"}, REPO, "")[0]["label"]
        == "commit abcdef1"
    )
    assert sources_for_call("get_latest_release", {}, REPO, "")[0]["url"] == f"{BASE}/releases"
    assert (
        "q=proxy%20error"
        in sources_for_call("search_issues", {"query": "proxy error"}, REPO, "")[0]["url"]
    )


def test_dedupe_keeps_order():
    a = {"label": "a", "url": "u1", "kind": "code"}
    b = {"label": "b", "url": "u2", "kind": "code"}
    assert dedupe([a, b, dict(a)]) == [a, b]
