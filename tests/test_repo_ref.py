import pytest

from reposcout.repo_ref import InvalidRepo, parse_repo


@pytest.mark.parametrize(
    "text, expected",
    [
        ("psf/requests", "psf/requests"),
        ("  psf/requests  ", "psf/requests"),
        ("https://github.com/psf/requests", "psf/requests"),
        ("https://github.com/psf/requests/", "psf/requests"),
        ("https://github.com/psf/requests.git", "psf/requests"),
        ("github.com/psf/requests", "psf/requests"),
        ("https://www.github.com/psf/requests/tree/main/src", "psf/requests"),
        ("https://github.com/psf/requests/blob/main/README.md", "psf/requests"),
        ("my-org/my.repo_name-2", "my-org/my.repo_name-2"),
    ],
)
def test_valid(text, expected):
    assert parse_repo(text).full_name == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "requests",
        "psf/",
        "/requests",
        "psf/requests/extra/path",
        "https://gitlab.com/a/b",
        "-bad/repo",
        "a/..",
        "a/b c",
        "psf/requests; rm -rf /",
    ],
)
def test_invalid(text):
    with pytest.raises(InvalidRepo):
        parse_repo(text)


def test_case_insensitive_comparison():
    ref = parse_repo("PSF/Requests")
    assert ref.same_as("psf", "requests")
    assert not ref.same_as("evil", "requests")
    assert ref.html_url == "https://github.com/PSF/Requests"
