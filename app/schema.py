from pydantic import BaseModel, Field, field_validator
from typing import List

# Supported content types
SUPPORTED_TYPES = ["qna", "user guide"]

# =========================
# Request models
# =========================

class AddContentRequest(BaseModel):
    content: str = Field(..., description="The text content to be added.")
    category: str = Field(..., description="Category under which the content should be indexed (e.g., FAQ, User Guide).")
    type: str = Field(..., description="Type of content. Supported values: 'qna', 'user guide'. Must be lowercase.")
    language: str = Field(..., description="Language of the content (e.g., 'en', 'fr').")

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        value_lower = value.lower()
        if value_lower not in SUPPORTED_TYPES:
            raise ValueError(f"type must be one of {SUPPORTED_TYPES}")
        return value_lower


class DeleteContentRequest(BaseModel):
    ids: List[int] = Field(..., description="List of content IDs that should be deleted.")


class InvokeRequest(BaseModel):
    content: str = Field(..., description="The content/query to process or search for.")
    category: str = Field(..., description="Category to use for indexing or searching (e.g., FAQ, User Guide).")
    type: str = Field(..., description="Type of the content. Supported values: 'qna', 'user guide'. Must be lowercase.")
    language: str = Field(..., description="Language of the content/query (e.g., 'en', 'fr').")

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        value_lower = value.lower()
        if value_lower not in SUPPORTED_TYPES:
            raise ValueError(f"type must be one of {SUPPORTED_TYPES}")
        return value_lower
