"""Cleaning helpers for YouTube comment text."""
import hashlib
import html
import re

_BR = re.compile(r"<br\s*/?>", re.I)
_A_HREF = re.compile(r'<a\s+[^>]*href="([^"]+)"[^>]*>', re.I)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s\-]?)?[6-9](?:[\s\-]?\d){9}(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def unescape(raw) -> str:
    """The scam export is HTML-escaped (&#39;, &amp;) and can hold <a> and <br> tags."""
    return html.unescape(str(raw))


def extract_links(raw) -> list[str]:
    """Links from <a href> tags and from plain text, in order, without repeats."""
    text = unescape(raw)
    found = _A_HREF.findall(text) + _URL.findall(_TAG.sub(" ", text))
    seen, links = set(), []
    for link in found:
        key = link.rstrip("/.,)").lower()
        if key not in seen:
            seen.add(key)
            links.append(link)
    return links


def clean_text(raw) -> str:
    """Visible text only: unescape, turn <br> into spaces, drop tags, collapse whitespace."""
    text = _BR.sub(" ", unescape(raw))
    text = _TAG.sub(" ", text)
    return _WS.sub(" ", text).strip()


def hash_author(author, salt: str = "yt-case-study") -> str:
    """Stable short id so repeat authors can be counted without keeping the handle."""
    return hashlib.sha256(f"{salt}|{author}".encode("utf-8")).hexdigest()[:10]


def mask_contact(text: str) -> str:
    """Replace phone numbers, emails and links so examples can be shown safely."""
    text = _URL.sub("[link]", text)
    text = _EMAIL.sub("[email]", text)
    return _PHONE.sub("[phone]", text)
