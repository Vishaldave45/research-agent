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
