from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

# Supported content types
SUPPORTED_TYPES = ["qna", "user_guide", "blog"]

# =========================
# Request models
# =========================

class AddContentRequest(BaseModel):
    content: str = Field(..., description="The text content to be added.")
    category: str = Field(..., description="Category under which the content should be indexed (e.g., Lifestyle, Technology).")
    type: str = Field(..., description="Type of content. Supported values: 'qna', 'user guide'. Must be lowercase.")

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
    category: str = Field(..., description="Category under which the content should be indexed (e.g., Lifestyle, Technology).")
    type: str = Field(..., description="Type of the content. Supported values: 'qna', 'user guide'. Must be lowercase.")
    

    top_docs: Optional[int] = Field(5, description="Number of top documents to keep after accumulation.")
    top_nchunk: Optional[int] = Field(10, description="Number of top chunks to consider from the input content.")
    top_k: Optional[int] = Field(3, description="Number of nearest neighbors to retrieve per chunk.")

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        value_lower = value.lower()
        if value_lower not in SUPPORTED_TYPES:
            raise ValueError(f"type must be one of {SUPPORTED_TYPES}")
        return value_lower