def calculate_trust_score(image_score: float, profile_score: float, engagement_score: float) -> float:
    weighted = (0.45 * image_score) + (0.35 * profile_score) + (0.20 * engagement_score)
    return max(0.0, min(100.0, round(weighted, 2)))


def classify_risk(score: float) -> str:
    if score >= 75:
        return "Low"
    if score >= 45:
        return "Medium"
    return "High"
