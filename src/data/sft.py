"""Schema-only preparation for later supervised legal instruction tuning.

No raw judgment is converted to instruction data here; human/curated data creation
remains an explicit later workflow.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class LegalInstructionExample:
    instruction: str
    context: str
    question: str
    answer: str
    rationale: Optional[str] = None
    citations: list[str] = field(default_factory=list)
    language: str = "en"
    domain: str = "General"
