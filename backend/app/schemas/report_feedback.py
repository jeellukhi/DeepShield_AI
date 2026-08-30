from typing import Literal

from pydantic import BaseModel, Field


class ReportFeedbackRequest(BaseModel):
    feedback_label: Literal["confirmed_fake", "confirmed_real", "unsure"]
    feedback_note: str = Field(default="", max_length=300)
