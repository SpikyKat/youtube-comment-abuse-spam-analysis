from ytmod.clean import clean_text, extract_links, hash_author, mask_contact
from ytmod.rules import language_hint, score_comment


def test_clean_text_unescapes_and_drops_tags():
    raw = "India&#39;s best<br>course <a href=\"https://x.example/a\">click here</a> &amp; save"
    assert clean_text(raw) == "India's best course click here & save"


def test_extract_links_reads_href_and_plain_text_once():
    raw = '<a href="https://bit.ly/abc">https://bit.ly/abc</a> and www.example.com'
    links = extract_links(raw)
    assert len(links) == 2
    assert any("bit.ly" in l for l in links)


def test_hash_author_is_stable_and_hides_handle():
    a = hash_author("@someone")
    assert a == hash_author("@someone")
    assert "someone" not in a and len(a) == 10
    assert a != hash_author("@someone_else")


def test_mask_contact_hides_phone_email_and_link():
    masked = mask_contact("call 98765 43210 or mail me@x.com or see https://t.me/abc")
    assert "9876" not in masked and "@x.com" not in masked and "t.me" not in masked


def test_benign_comment_is_low_risk():
    s = score_comment("This video explains the problem very clearly, thanks for sharing")
    assert s.tier == "low" and s.score == 0


def test_phone_and_contact_app_make_high_risk():
    s = score_comment("Earn 50k per month from home, message me on WhatsApp 98765 43210")
    assert s.tier == "high"
    assert "phone number" in s.reasons and "contact app named" in s.reasons


def test_victim_naming_whatsapp_is_not_promotion():
    s = score_comment("They asked me to message them on WhatsApp, took the fee and blocked me. Total fraud")
    assert s.tier == "low"


def test_dm_offer_with_discount_is_flagged():
    s = score_comment("Internship course at 80% discount, DM me for details")
    assert s.tier in ("medium", "high")


def test_scam_talk_is_not_treated_as_promotion():
    s = score_comment("This is a scam, they took my money and refused a refund")
    assert s.scam_talk is True
    assert s.tier == "low"


def test_external_link_counts_but_youtube_link_does_not():
    assert score_comment("see https://youtu.be/abc123").n_links == 0
    assert score_comment("see https://bit.ly/abc").n_links == 1


def test_language_hint():
    assert language_hint("bhai ye sab fraud hai nahi lena") == "hinglish_likely"
    assert language_hint("यह सब धोखा है") == "indic_script"
    assert language_hint("this looks fine to me") == "english_or_other"
