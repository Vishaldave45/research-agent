"""
Pydantic models used to validate/shape data flowing through the app.
These are plain data schemas - no LangChain dependency here.
"""
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    question: str = Field(description="The user's question")
    session_id: Optional[str] = Field(
        default="default", description="Conversation session identifier"
    )


class ChatResponse(BaseModel):
    answer: str = Field(description="The assistant's answer")
    session_id: str = Field(description="The session this response belongs to")


class RAGResponse(BaseModel):
    answer: str = Field(description="The answer generated from retrieved context")
    sources: list[str] = Field(
        default_factory=list, description="Source document snippets used"
    )


class AgentResponse(BaseModel):
    answer: str = Field(description="The agent's final answer")
    tools_used: list[str] = Field(
        default_factory=list, description="Names of tools the agent invoked"
    )


class ResearchResult(BaseModel):
    topic: str = Field(description="The research topic")
    findings: list[str] = Field(description="Key research findings")
    sources: list[str] = Field(description="Sources or references")
    confidence: float = Field(description="Confidence score 0.0-1.0")


class SummaryResult(BaseModel):
    title: str = Field(description="Title of the summarized content")
    summary: str = Field(description="2-3 sentence summary")
    key_points: list[str] = Field(description="Bullet-point key takeaways")
    word_count: int = Field(description="Approximate word count of original")


class FactCheckResult(BaseModel):
    claim: str = Field(description="The claim being checked")
    verdict: str = Field(description="TRUE, FALSE, PARTIALLY TRUE, or UNVERIFIABLE")
    evidence: list[str] = Field(description="Supporting evidence points")
    confidence: float = Field(description="Confidence score 0.0-1.0")


class QAResult(BaseModel):
    question: str = Field(description="The original question")
    answer: str = Field(description="The answer")
    reasoning: str = Field(description="Step-by-step reasoning")
    confidence: float = Field(description="Confidence score 0.0-1.0")


class ReportResult(BaseModel):
    title: str = Field(description="Report title")
    sections: list[dict] = Field(description="List of dicts representing sections, with 'heading' and 'content' keys")
    conclusion: str = Field(description="Concluding summary")
    references: list[str] = Field(description="References used")

