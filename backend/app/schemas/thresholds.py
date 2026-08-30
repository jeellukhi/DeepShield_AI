from pydantic import BaseModel, Field


class ThresholdsUpdateRequest(BaseModel):
    image_fake_probability_threshold: float | None = Field(default=None, ge=20, le=80)
    video_fake_probability_threshold: float | None = Field(default=None, ge=20, le=80)
    profile_fake_probability_threshold: float | None = Field(default=None, ge=20, le=80)

