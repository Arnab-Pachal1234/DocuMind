from typing import Optional
from pydantic import BaseModel


class AskRequest(BaseModel):
    thread_id: str
    question: str


class CreateThreadResponse(BaseModel):
    thread_id: str
    share_url: str