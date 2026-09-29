from __future__ import annotations
import math
import re


def extract_numeric_features(
    followers: int | float,
    following: int | float,
    posts: int | float,
    engagement_rate: float,
    image_score: float = 50.0,
) -> list[float]:
    """
    Extract engineered numeric features from social profile stats.
    Returns a fixed-length float vector used by the numeric classifier.
    """
    followers = max(0.0, float(followers))
    following = max(0.0, float(following))
    posts = max(0.0, float(posts))
    engagement_rate = max(0.0, float(engagement_rate))
    image_score = max(0.0, min(100.0, float(image_score)))

    # --- Ratio features ---
    ff_ratio = followers / max(following, 1.0)           # follower/following ratio
    fp_ratio = followers / max(posts, 1.0)               # followers per post
    pf_ratio = posts / max(followers, 1.0) * 1000        # posts per 1k followers

    # --- Log-scaled raw values (avoids huge number skew) ---
    log_followers = math.log1p(followers)
    log_following = math.log1p(following)
    log_posts = math.log1p(posts)

    # --- Engagement anomaly ---
    # Real accounts: engagement 1%–10%. Bots: <0.3% or >30%
    engagement_low_flag = 1.0 if engagement_rate < 0.3 else 0.0
    engagement_extreme_flag = 1.0 if engagement_rate > 25.0 else 0.0
    engagement_normal_flag = 1.0 if 1.0 <= engagement_rate <= 15.0 else 0.0
    engagement_log = math.log1p(engagement_rate)

    # --- Suspicious pattern flags ---
    # Very few followers but many following = bot follow-farm
    follow_farm_flag = 1.0 if (followers < 100 and following > 500) else 0.0
    # Brand new / ghost account
    ghost_account_flag = 1.0 if (posts == 0 and followers < 50) else 0.0
    # Zero posts
    no_posts_flag = 1.0 if posts == 0 else 0.0
    # Very few posts
    few_posts_flag = 1.0 if 0 < posts < 5 else 0.0
    # Extremely low follower count
    tiny_account_flag = 1.0 if followers < 20 else 0.0
    # Extremely high following relative to followers
    low_ff_flag = 1.0 if ff_ratio < 0.05 else 0.0
    medium_ff_flag = 1.0 if 0.05 <= ff_ratio < 0.5 else 0.0
    high_ff_flag = 1.0 if ff_ratio >= 2.0 else 0.0

    # --- Image score signal ---
    image_suspicious = 1.0 if image_score < 40.0 else 0.0
    image_score_norm = image_score / 100.0

    # --- Composite risk score (linear heuristic, used as a feature) ---
    risk = 0.0
    if follow_farm_flag: risk += 0.35
    if ghost_account_flag: risk += 0.20
    if engagement_low_flag: risk += 0.20
    if engagement_extreme_flag: risk += 0.10
    if no_posts_flag: risk += 0.15
    if tiny_account_flag: risk += 0.10
    if low_ff_flag: risk += 0.15
    risk = min(1.0, risk)

    return [
        # Raw log-scaled values
        log_followers,
        log_following,
        log_posts,
        engagement_log,
        # Ratios
        min(ff_ratio, 100.0),       # cap at 100 to avoid inf
        min(fp_ratio, 10000.0),
        min(pf_ratio, 1000.0),
        # Engagement flags
        engagement_low_flag,
        engagement_extreme_flag,
        engagement_normal_flag,
        # Account pattern flags
        follow_farm_flag,
        ghost_account_flag,
        no_posts_flag,
        few_posts_flag,
        tiny_account_flag,
        low_ff_flag,
        medium_ff_flag,
        high_ff_flag,
        # Image score
        image_score_norm,
        image_suspicious,
        # Composite risk
        risk,
    ]


# Feature names (for debugging/explainability)
NUMERIC_FEATURE_NAMES = [
    "log_followers", "log_following", "log_posts", "engagement_log",
    "ff_ratio", "fp_ratio", "pf_ratio",
    "engagement_low_flag", "engagement_extreme_flag", "engagement_normal_flag",
    "follow_farm_flag", "ghost_account_flag", "no_posts_flag", "few_posts_flag",
    "tiny_account_flag", "low_ff_flag", "medium_ff_flag", "high_ff_flag",
    "image_score_norm", "image_suspicious",
    "composite_risk",
]
