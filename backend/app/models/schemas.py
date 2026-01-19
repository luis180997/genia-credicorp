"""GenAI Agent - Pydantic Models and Schemas"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Message role in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Single chat message."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sources: Optional[List[str]] = None  # Document sources used for RAG


class ChatRequest(BaseModel):
    """Request to chat endpoint."""
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(..., min_length=1, max_length=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "¿Cuáles son las políticas de inversión?",
                "session_id": "user-session-123"
            }
        }


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    response: str
    session_id: str
    sources: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Las políticas de inversión de Credicorp Capital incluyen...",
                "session_id": "user-session-123",
                "sources": ["politicas_inversion.pdf"],
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class Conversation(BaseModel):
    """Full conversation with history."""
    session_id: str
    messages: List[ChatMessage] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    filename: str
    status: str
    message: str
    chunks_created: int = 0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    services: dict
