from pydantic import BaseModel, Field
from typing import List, Literal, Optional

Decision = Literal["approve", "decline", "need_more_info", "review"]

class Evidence(BaseModel):
    doc_title: str
    version: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None

class Reason(BaseModel):
    type: Literal["rule", "model", "policy"]
    text: str
    evidence: List[Evidence] = Field(default_factory=list)

class AssistantResponse(BaseModel):
    summary: str
    decision: Decision
    reasons: List[Reason]
    missing_info: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    customer_message_draft: Optional[str] = None
    risk_note: Optional[str] = None
