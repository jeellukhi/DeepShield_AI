from pydantic import BaseModel, Field


class ProfileAnalyzeRequest(BaseModel):
    username: str = Field(min_length=2, max_length=50)
    bio: str = Field(default="", max_length=300)
    followers: int = Field(ge=0)
    following: int = Field(ge=0)
    posts: int = Field(ge=0)
    engagement_rate: float = Field(ge=0.0, le=100.0)
    image_score: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Future model output for image authenticity score.",
    )
