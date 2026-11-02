"""Transparent rules for spam and scam promotion in comments.

The weights are my own judgement and have NOT been validated against labelled data. They are meant to be read,
argued with and tuned once a hand-labelled sample exists (scripts/evaluate_labels.py).
"""
import re
from dataclasses import dataclass, field

from .clean import clean_text, extract_links

SHORTENERS = ("bit.ly", "tinyurl", "t.me", "wa.me", "linktr.ee", "cutt.ly", "rb.gy", "goo.gl", "is.gd")
YOUTUBE_HOSTS = ("youtube.com", "youtu.be")

_CONTACT_APP = re.compile(r"\b(telegram|whatsapp|whats app|insta(?:gram)?)\b", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s\-]?)?[6-9](?:[\s\-]?\d){9}(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_MONEY = re.compile(r"(?:₹|\brs\.?\s?\d|\binr\b|\blakhs?\b|\bper (?:month|day|week)\b|\b\d+\s?k\b\s*(?:per|/))", re.I)
_PROMO = re.compile(
    r"\b(earn|income|profit|investment|guaranteed|referral|discount|enrol+|register|work from home|part[- ]time|"
    r"dm me|contact me|message me|check (?:out )?my|my channel|subscribe|click|link in|visit)\b", re.I)
_SCAM_TALK = re.compile(
    r"\b(scam|scammer|scammers|fraud|cheat|cheated|cheating|fake|loot|fooled|conned|ponzi|refund|consumer court|"
    r"legal action|complaint)\b", re.I)
_SOLICIT = re.compile(
    r"\b(dm|dm me|message me|msg me|contact me|connect with me|follow me|join (?:my|our|us)|text me|"
    r"call me|reach out|get in touch)\b", re.I)  # "whatsapp me" left out: in Hinglish "me" means "in"
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
_HINGLISH = {"hai", "nahi", "nhi", "kya", "bhai", "hain", "aur", "bahut", "yaar", "paisa", "kaam", "mat", "sab",
             "kuch", "log", "tha", "thi", "hoga", "karo", "kar", "raha", "rahe"}
_INDIC = re.compile(r"[ऀ-෿]")  # Devanagari through Malayalam


@dataclass
class Signals:
    score: int = 0
    tier: str = "low"
    reasons: list[str] = field(default_factory=list)
    n_links: int = 0
    scam_talk: bool = False
    language: str = "english_or_other"


def _external_links(raw) -> list[str]:
    return [l for l in extract_links(raw) if not any(h in l.lower() for h in YOUTUBE_HOSTS)]


def language_hint(text: str) -> str:
    """Rough guess only: indic script, likely romanised Hindi, or English/other."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(1 for c in letters if _INDIC.match(c)) / len(letters) >= 0.3:
        return "indic_script"
    words = re.findall(r"[a-z]+", text.lower())
    if len({w for w in words if w in _HINGLISH}) >= 2:
        return "hinglish_likely"
    return "english_or_other"


def score_comment(raw) -> Signals:
    text = clean_text(raw)
    s = Signals(language=language_hint(text))
    links = _external_links(raw)
    s.n_links = len(links)
    if links:
        s.score += 2
        s.reasons.append("external link")
        if any(any(x in l.lower() for x in SHORTENERS) for l in links):
            s.score += 1
            s.reasons.append("shortener or messaging link")
    if _PHONE.search(text):
        s.score += 3
        s.reasons.append("phone number")
    if _EMAIL.search(text):
        s.score += 2
        s.reasons.append("email address")
    if _CONTACT_APP.search(text):
        # naming an app is weak evidence: victims describe scammers asking them to use WhatsApp too
        s.score += 1
        s.reasons.append("contact app named")
    if _SOLICIT.search(text):
        s.score += 2
        s.reasons.append("asks reader to get in touch")
    promo = len(_PROMO.findall(text))
    if promo >= 2:
        s.score += 2
        s.reasons.append("promotional wording")
    elif promo == 1:
        s.score += 1
        s.reasons.append("promotional wording (weak)")
    if _MONEY.search(text):
        s.score += 1
        s.reasons.append("money amount")
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 20 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        s.score += 1
        s.reasons.append("mostly capitals")
    if len(_EMOJI.findall(text)) >= 5:
        s.score += 1
        s.reasons.append("many emojis")
    s.scam_talk = bool(_SCAM_TALK.search(text))
    if s.scam_talk and not (links or _PHONE.search(text) or _EMAIL.search(text)):
        # people describing or warning about a scam rarely advertise: no link, phone or email, so discount the score
        s.score = max(0, s.score - 2)
        s.reasons.append("scam-talk discount")
    s.tier = "high" if s.score >= 4 else "medium" if s.score >= 2 else "low"
    return s
