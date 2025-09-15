from typing import List, Optional
from pydantic import BaseModel, Field

class Classification(BaseModel):
    package_type: str
    summary: str

class ChecklistItem(BaseModel):
    id: str
    text: str

class Checklist(BaseModel):
    submittals: List[ChecklistItem] = Field(default_factory=list)

class Finding(BaseModel):
    req_id: str
    status: str  # present | missing | not_applicable | unclear
    evidence: str = ""

class RunState(BaseModel):
    # File references (not serialized to state)
    spec_pdf_name: Optional[str] = None
    submittal_pdf_name: Optional[str] = None
    
    # Extracted content
    spec_text: Optional[str] = None
    submittal_text: Optional[str] = None
    
    # Analysis results
    classification: Optional[Classification] = None
    checklist: Optional[Checklist] = None
    findings: List[Finding] = Field(default_factory=list)
    
    # Progress tracking
    current_step: str = "start"
    error_message: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True