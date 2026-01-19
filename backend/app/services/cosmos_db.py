"""
Genia AI Agent - Cosmos DB Service
Supports both Azure Cosmos DB and in-memory storage.
"""
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
import json

from ..config import settings
from ..models.schemas import Conversation, ChatMessage, MessageRole


class CosmosDBService:
    """
    Conversation history storage service with Azure Cosmos DB support.
    Falls back to in-memory storage when Azure is not configured.
    
    For Azure migration:
    1. Set USE_AZURE_COSMOS=true in .env
    2. Provide AZURE_COSMOS_ENDPOINT
    3. Provide AZURE_COSMOS_KEY
    4. Provide AZURE_COSMOS_DATABASE
    5. Provide AZURE_COSMOS_CONTAINER
    """
    
    def __init__(self):
        self.use_azure = settings.use_azure_cosmos
        
        if self.use_azure:
            self._init_azure_client()
        else:
            self._init_memory_storage()
    
    def _init_azure_client(self):
        """Initialize Azure Cosmos DB client."""
        try:
            from azure.cosmos import CosmosClient, PartitionKey
            
            self.cosmos_client = CosmosClient(
                settings.azure_cosmos_endpoint,
                credential=settings.azure_cosmos_key
            )
            
            # Create database if not exists
            self.database = self.cosmos_client.create_database_if_not_exists(
                id=settings.azure_cosmos_database
            )
            
            # Create container if not exists
            self.container = self.database.create_container_if_not_exists(
                id=settings.azure_cosmos_container,
                partition_key=PartitionKey(path="/session_id"),
                #offer_throughput=400
            )
            
            print(f"[INFO] Conectado a Azure Cosmos DB: {settings.azure_cosmos_database}/{settings.azure_cosmos_container}")
        except Exception as e:
            print(f"[ERROR] Azure Cosmos DB no disponible, usando memoria: {e}")
            self.use_azure = False
            self._init_memory_storage()
    
    def _init_memory_storage(self):
        """Initialize in-memory storage."""
        self._conversations: Dict[str, Conversation] = {}
        print("[INFO] Usando almacenamiento en memoria para conversaciones")
    
    def get_conversation(self, session_id: str) -> Conversation:
        """
        Get conversation by session ID.
        Creates new conversation if not exists.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Conversation object
        """
        if self.use_azure:
            return self._get_from_azure(session_id)
        return self._get_from_memory(session_id)
    
    def _get_from_azure(self, session_id: str) -> Conversation:
        """Get conversation from Azure Cosmos DB."""
        try:
            query = f"SELECT * FROM c WHERE c.session_id = '{session_id}'"
            items = list(self.container.query_items(query, enable_cross_partition_query=True))
            
            if items:
                item = items[0]
                return Conversation(
                    session_id=item['session_id'],
                    messages=[ChatMessage(**msg) for msg in item.get('messages', [])],
                    created_at=datetime.fromisoformat(item['created_at']),
                    updated_at=datetime.fromisoformat(item['updated_at'])
                )
        except Exception as e:
            print(f"Error fetching from Cosmos DB: {e}")
        
        return Conversation(session_id=session_id)
    
    def _get_from_memory(self, session_id: str) -> Conversation:
        """Get conversation from memory."""
        if session_id not in self._conversations:
            self._conversations[session_id] = Conversation(session_id=session_id)
        return self._conversations[session_id]
    
    def save_conversation(self, conversation: Conversation) -> bool:
        """
        Save conversation to storage.
        
        Args:
            conversation: Conversation to save
            
        Returns:
            True if successful
        """
        conversation.updated_at = datetime.utcnow()
        
        if self.use_azure:
            return self._save_to_azure(conversation)
        return self._save_to_memory(conversation)
    
    def _save_to_azure(self, conversation: Conversation) -> bool:
        """Save conversation to Azure Cosmos DB."""
        try:
            item = {
                'id': conversation.session_id,
                'session_id': conversation.session_id,
                'messages': [msg.model_dump() for msg in conversation.messages],
                'created_at': conversation.created_at.isoformat(),
                'updated_at': conversation.updated_at.isoformat()
            }
            # Convert datetime objects in messages
            for msg in item['messages']:
                if isinstance(msg.get('timestamp'), datetime):
                    msg['timestamp'] = msg['timestamp'].isoformat()
            
            self.container.upsert_item(item)
            return True
        except Exception as e:
            print(f"Error saving to Cosmos DB: {e}")
            return False
    
    def _save_to_memory(self, conversation: Conversation) -> bool:
        """Save conversation to memory."""
        self._conversations[conversation.session_id] = conversation
        return True
    
    def add_message(
        self, 
        session_id: str, 
        role: MessageRole, 
        content: str,
        sources: Optional[List[str]] = None
    ) -> Conversation:
        """Add a message to conversation history."""
        conversation = self.get_conversation(session_id)
        
        # Asegurar encoding correcto de caracteres UTF-8 (acentos, ñ, etc)
        # This ensures the string itself is properly encoded before being stored.
        if isinstance(content, str):
            content = content.encode('utf-8', errors='ignore').decode('utf-8')
        
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            sources=sources
        )
        conversation.messages.append(message)
        
        self.save_conversation(conversation)
        return conversation
    
    def get_message_history(self, session_id: str, limit: int = 10) -> List[ChatMessage]:
        """
        Get recent message history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of recent messages
        """
        conversation = self.get_conversation(session_id)
        return conversation.messages[-limit:]
    
    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        if self.use_azure:
            return self._delete_from_azure(session_id)
        return self._delete_from_memory(session_id)
    
    def _delete_from_azure(self, session_id: str) -> bool:
        """Delete conversation from Azure Cosmos DB."""
        try:
            self.container.delete_item(item=session_id, partition_key=session_id)
            return True
        except Exception:
            return False
    
    def _delete_from_memory(self, session_id: str) -> bool:
        """Delete conversation from memory."""
        if session_id in self._conversations:
            del self._conversations[session_id]
            return True
        return False
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        if self.use_azure:
            return self._list_azure_sessions()
        return list(self._conversations.keys())
    
    def _list_azure_sessions(self) -> List[str]:
        """List sessions from Azure Cosmos DB."""
        try:
            query = "SELECT DISTINCT c.session_id FROM c"
            items = list(self.container.query_items(query, enable_cross_partition_query=True))
            return [item['session_id'] for item in items]
        except Exception:
            return []
    
    def get_status(self) -> dict:
        """Get service status for health check."""
        session_count = len(self.list_sessions())
        return {
            "type": "azure_cosmos" if self.use_azure else "in_memory",
            "available": True,
            "session_count": session_count,
            "database": settings.azure_cosmos_database if self.use_azure else "memory"
        }
