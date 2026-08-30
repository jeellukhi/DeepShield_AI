def build_recommendation(trust_score: float, risk_level: str, component_scores: dict) -> dict:
    image_score = float(component_scores.get("image_score", 0))
    profile_score = float(component_scores.get("profile_score", 0))
    engagement_score = float(component_scores.get("engagement_score", 0))

    actions: list[str] = []
    if image_score < 45:
        actions.append("Image or video signal is suspicious. Verify source media before trusting the profile.")
    if profile_score < 50:
        actions.append("Username/Bio pattern appears risky. Avoid sharing personal or financial information.")
    if engagement_score < 50:
        actions.append("Engagement behavior looks abnormal. Cross-check account age and interaction quality.")
    if risk_level == "High":
        actions.append("Treat this account as high risk. Report and block if scam behavior is observed.")

    if trust_score >= 80 and risk_level == "Low":
        label = "Safe"
    elif trust_score >= 60:
        label = "Caution"
    elif trust_score >= 40:
        label = "Alert"
    else:
        label = "High Alert"

    if not actions and label == "Safe":
        actions.append("No major risks detected. Continue normal monitoring.")

    return {"label": label, "actions": actions}
