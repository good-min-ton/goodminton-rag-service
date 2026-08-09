"""CORS origin configuration.

The frontend is on Vercel and the API is reached through a public tunnel, so the
two are always cross-origin. These pin how the comma-separated setting turns
into Starlette's arguments, including the wildcard form Vercel preview
deployments need.
"""

import re

import pytest
from starlette.middleware.cors import CORSMiddleware

from app.core.config import Settings


def _kwargs(value: str) -> dict:
    return Settings(_env_file=None, cors_origins=value).cors_middleware_kwargs


def _allows(value: str, origin: str) -> bool:
    """Ask the real middleware, not a reimplementation of its matching rules."""
    mw = CORSMiddleware(app=None, **_kwargs(value), allow_credentials=False)
    return mw.is_allowed_origin(origin)


def test_default_allows_every_origin():
    assert _kwargs("*") == {"allow_origins": ["*"]}
    assert _allows("*", "https://anything.example.com") is True


def test_blank_setting_falls_back_to_open():
    """An empty value must not lock everyone out - a deploy that forgot to set it
    should behave like the default, not go dark."""
    assert _kwargs("") == {"allow_origins": ["*"]}
    assert _kwargs("  ,  ") == {"allow_origins": ["*"]}


def test_exact_origins_are_listed_verbatim():
    kwargs = _kwargs("https://shop.vercel.app,https://goodminton.vn")

    assert kwargs == {
        "allow_origins": ["https://shop.vercel.app", "https://goodminton.vn"]
    }


def test_whitespace_around_entries_is_ignored():
    assert _kwargs(" https://a.com , https://b.com ") == {
        "allow_origins": ["https://a.com", "https://b.com"]
    }


def test_listed_origin_is_allowed_and_others_are_not():
    value = "https://shop.vercel.app"

    assert _allows(value, "https://shop.vercel.app") is True
    assert _allows(value, "https://evil.com") is False


def test_wildcard_covers_vercel_previews():
    """Preview deployments get a generated subdomain per branch, so listing them
    one by one is not possible."""
    value = "https://shop.vercel.app,https://*.vercel.app"

    assert _allows(value, "https://shop.vercel.app") is True
    assert (
        _allows(value, "https://goodminton-git-feat-picker-lezh1n.vercel.app") is True
    )
    assert _allows(value, "https://evil.com") is False


def test_wildcard_stays_within_one_label():
    """The security-relevant case: https://*.vercel.app must not hand access to
    a subdomain an attacker can register under some other domain."""
    value = "https://*.vercel.app"

    assert _allows(value, "https://ok.vercel.app") is True
    assert _allows(value, "https://evil.attacker.vercel.app") is False
    assert _allows(value, "https://vercel.app.attacker.com") is False


def test_wildcard_cannot_be_extended_by_a_longer_origin():
    """Starlette matches the regex with fullmatch, so a prefix match is not
    enough. Pinned because a switch to search() would silently open this up."""
    value = "https://*.vercel.app"

    assert _allows(value, "https://ok.vercel.app.evil.com") is False
    assert _allows(value, "evil://ok.vercel.app") is False


def test_star_anywhere_in_the_list_wins():
    """A narrower entry sitting next to `*` would read as a restriction that is
    not actually in force, so `*` collapses the whole list."""
    assert _kwargs("https://shop.vercel.app,*") == {"allow_origins": ["*"]}


def test_regex_is_only_built_when_a_wildcard_is_present():
    assert "allow_origin_regex" not in _kwargs("https://a.com")
    assert "allow_origin_regex" in _kwargs("https://*.a.com")


def test_dots_in_a_wildcard_entry_stay_literal():
    """Unescaped, the dot in .vercel.app would match any character."""
    pattern = _kwargs("https://*.vercel.app")["allow_origin_regex"]

    assert re.fullmatch(pattern, "https://ok.vercel.app")
    assert not re.fullmatch(pattern, "https://okXvercel.app")


@pytest.mark.parametrize("origin", ["", "null"])
def test_opaque_origins_are_not_allowed_when_restricted(origin):
    """A sandboxed iframe or a file:// page sends Origin: null."""
    assert _allows("https://shop.vercel.app", origin) is False
