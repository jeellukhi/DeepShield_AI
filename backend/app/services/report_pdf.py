from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def _line(pdf: canvas.Canvas, text: str, y: float, left: float = 50) -> float:
    pdf.drawString(left, y, text)
    return y - 18


def build_report_pdf(report: dict) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    _, height = A4

    y = height - 50
    y = _line(pdf, "DeepShield AI - Verification Report", y)
    y = _line(pdf, "=" * 42, y + 2)

    created = report.get("created_at")
    if isinstance(created, datetime):
        created_text = created.strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        created_text = str(created or "N/A")

    report_id = str(report.get("_id", "N/A"))
    input_profile = report.get("input_profile", {})
    component_scores = report.get("component_scores", {})
    recommendation_details = report.get("recommendation_details", {})

    y = _line(pdf, f"Report ID: {report_id}", y)
    y = _line(pdf, f"Generated At: {created_text}", y)
    y = _line(pdf, f"Analysis Type: {report.get('analysis_type', 'profile')}", y)
    y = _line(pdf, f"Username: {input_profile.get('username', 'N/A')}", y)
    y = _line(pdf, f"Trust Score: {report.get('trust_score', 'N/A')}%", y)
    y = _line(pdf, f"Risk Level: {report.get('risk_level', 'N/A')}", y)
    y = _line(pdf, f"Recommendation: {report.get('recommendation', 'N/A')}", y)

    y -= 8
    y = _line(pdf, "Component Scores:", y)
    y = _line(pdf, f"  Image Score: {component_scores.get('image_score', 'N/A')}", y)
    y = _line(pdf, f"  Profile Score: {component_scores.get('profile_score', 'N/A')}", y)
    y = _line(pdf, f"  Engagement Score: {component_scores.get('engagement_score', 'N/A')}", y)

    actions = recommendation_details.get("actions", [])
    if actions:
        y -= 8
        y = _line(pdf, "Recommended Actions:", y)
        for idx, action in enumerate(actions, start=1):
            if y < 60:
                pdf.showPage()
                y = height - 50
            y = _line(pdf, f"  {idx}. {action}", y)

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()
