from pydantic import BaseModel, Field

class ResponseOutput(BaseModel):
    predicted_digit : int = Field(..., description="Predicted digit")
    confidence : float = Field(..., description="Confidence score")
    probability : dict[str, float] = Field(
        ...,
    )


