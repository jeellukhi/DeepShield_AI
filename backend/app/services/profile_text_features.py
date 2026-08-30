from __future__ import annotations

import re

SUSPICIOUS_KEYWORDS = [
    "official",
    "support",
    "helpdesk",
    "free",
    "giveaway",
    "crypto",
    "bitcoin",
    "investment",
    "dm for",
    "click link",
    "urgent",
    "verify now",
    "loan",
]


def build_profile_text(username: str, bio: str) -> str:
    uname = str(username or "").lower().strip()
    bio_text = str(bio or "").lower().strip()
    combined = f"{uname} {bio_text}".strip()
    tokens: list[str] = [f"username:{uname}", f"bio:{bio_text}"]

    uname_len = len(uname)
    if uname_len <= 6:
        tokens.append("__uname_len_short")
    elif uname_len <= 12:
        tokens.append("__uname_len_medium")
    else:
        tokens.append("__uname_len_long")

    bio_len = len(bio_text)
    if bio_len < 20:
        tokens.append("__bio_short")
    elif bio_len < 80:
        tokens.append("__bio_medium")
    else:
        tokens.append("__bio_long")

    if re.search(r"\d{4,}", uname):
        tokens.append("__uname_long_digits")
    if uname and (sum(ch.isdigit() for ch in uname) / max(len(uname), 1)) >= 0.35:
        tokens.append("__uname_high_digit_ratio")
    if len(set(uname)) <= 4 and uname:
        tokens.append("__uname_repetitive")
    if "@" in bio_text:
        tokens.append("__bio_contains_email")
    if "http://" in bio_text or "https://" in bio_text or "www." in bio_text:
        tokens.append("__bio_contains_url")

    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in combined:
            safe_name = keyword.replace(" ", "_")
            tokens.append(f"__kw_{safe_name}")

    return " ".join(token for token in tokens if token)
