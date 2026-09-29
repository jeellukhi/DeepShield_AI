from __future__ import annotations
import re

# Expanded from 13 → 85 suspicious keywords covering scam/bot/fake patterns
SUSPICIOUS_KEYWORDS = [
    # Scam / financial fraud
    "official", "support", "helpdesk", "free", "giveaway", "crypto",
    "bitcoin", "investment", "dm for", "click link", "urgent", "verify now",
    "loan", "profit", "earn money", "make money", "passive income", "get rich",
    "financial freedom", "100% profit", "guaranteed return", "double your",
    "trading signals", "forex", "binary options", "nft drop", "airdrop",
    "whitelist", "presale", "token sale", "binance", "coinbase",
    # Fake celebrity / impersonation
    "official account", "verified account", "real account", "not a bot",
    "im not fake", "legit account", "this is real", "authentic",
    # Spam / engagement bait
    "follow back", "followback", "f4f", "l4l", "like for like",
    "follow for follow", "shoutout", "promo", "dm me", "link in bio",
    "check bio", "see profile", "visit link", "swipe up",
    # Fake influencer
    "collab", "sponsored", "ambassador", "brand deal", "paid partnership",
    # Bot / automation signals
    "auto", "bot", "automated", "software", "tool", "hack",
    "followers for sale", "buy followers", "increase followers",
    # Urgency / FOMO
    "limited time", "act now", "dont miss", "hurry", "last chance",
    "expires soon", "24 hours", "today only",
    # Adult / inappropriate spam
    "onlyfans", "18+", "nsfw", "adult content",
    # Generic scam phrases
    "click here", "sign up now", "register now", "join now",
    "contact me", "whatsapp me", "telegram", "signal me",
    "cash app", "paypal me", "venmo",
]


def build_profile_text(username: str, bio: str) -> str:
    """
    Build an enriched text representation of a profile for TF-IDF.
    Adds structured token signals on top of raw text.
    """
    uname = str(username or "").lower().strip()
    bio_text = str(bio or "").lower().strip()
    combined = f"{uname} {bio_text}".strip()

    tokens: list[str] = [
        f"username:{uname}",
        f"bio:{bio_text}",
    ]

    # --- Username length signals ---
    uname_len = len(uname)
    if uname_len <= 5:
        tokens.append("__uname_len_very_short")
    elif uname_len <= 9:
        tokens.append("__uname_len_short")
    elif uname_len <= 15:
        tokens.append("__uname_len_medium")
    else:
        tokens.append("__uname_len_long")

    # --- Bio length signals ---
    bio_len = len(bio_text)
    if bio_len == 0:
        tokens.append("__bio_empty")
    elif bio_len < 15:
        tokens.append("__bio_very_short")
    elif bio_len < 50:
        tokens.append("__bio_short")
    elif bio_len < 120:
        tokens.append("__bio_medium")
    else:
        tokens.append("__bio_long")

    # --- Username pattern signals ---
    if re.search(r"\d{4,}", uname):
        tokens.append("__uname_long_digits")
    if re.search(r"\d{6,}", uname):
        tokens.append("__uname_very_long_digits")
    digit_ratio = sum(ch.isdigit() for ch in uname) / max(len(uname), 1)
    if digit_ratio >= 0.5:
        tokens.append("__uname_mostly_digits")
    elif digit_ratio >= 0.3:
        tokens.append("__uname_high_digit_ratio")
    if len(set(uname.replace("_", "").replace(".", ""))) <= 4 and uname:
        tokens.append("__uname_repetitive")
    if uname.count("_") >= 3:
        tokens.append("__uname_many_underscores")
    if re.search(r"(official|support|help|admin|real|verify)", uname):
        tokens.append("__uname_authority_word")
    if re.search(r"\d{3,}$", uname):
        tokens.append("__uname_ends_digits")

    # --- Bio content signals ---
    if "@" in bio_text:
        tokens.append("__bio_contains_email")
    if "http://" in bio_text or "https://" in bio_text or "www." in bio_text:
        tokens.append("__bio_contains_url")
    if "📞" in bio_text or "📱" in bio_text or "☎" in bio_text:
        tokens.append("__bio_phone_emoji")
    if "💰" in bio_text or "💵" in bio_text or "💸" in bio_text or "🤑" in bio_text:
        tokens.append("__bio_money_emoji")
    if "🔗" in bio_text or "👇" in bio_text or "⬇" in bio_text:
        tokens.append("__bio_link_emoji")
    if bio_text.count("!") >= 3:
        tokens.append("__bio_many_exclamations")
    if bio_text.isupper() and bio_len > 10:
        tokens.append("__bio_all_caps")
    # Count emojis (rough proxy)
    emoji_count = sum(1 for ch in bio_text if ord(ch) > 127000)
    if emoji_count >= 5:
        tokens.append("__bio_many_emojis")

    # --- Suspicious keyword signals ---
    kw_count = 0
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in combined:
            safe_name = re.sub(r"[^a-z0-9]", "_", keyword)
            tokens.append(f"__kw_{safe_name}")
            kw_count += 1
    if kw_count >= 2:
        tokens.append("__multiple_suspicious_keywords")
    if kw_count >= 4:
        tokens.append("__many_suspicious_keywords")

    return " ".join(token for token in tokens if token)
