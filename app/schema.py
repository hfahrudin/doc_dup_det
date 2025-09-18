from pydantic import BaseModel
from typing import List

# Request models
class AddContentRequest(BaseModel):
    content: str
    category: str
    tags: List[str]

class DeleteContentRequest(BaseModel):
    ids: List[int]  # list of IDs to delete

class InvokeRequest(BaseModel):
    content: str
    category: str
    tags: List[str]