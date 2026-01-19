"""
Genia AI Agent - Ajustes de Configuración
Soporta tanto alternativas locales como servicios de Azure para una fácil migración.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    """Configuración de la aplicación con soporte para migración a Azure."""
    
    # Información de la App
    app_name: str = "Genia"
    app_version: str = "1.0.0"
    debug: bool = True
    
    # ======================
    # Configuración OpenAI
    # ======================
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    
    # Azure OpenAI (para migración)
    use_azure_openai: bool = Field(default=False, alias="USE_AZURE_OPENAI")
    azure_openai_api_key: Optional[str] = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: Optional[str] = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment_name: Optional[str] = Field(default=None, alias="AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_api_version: str = Field(default="2024-02-15-preview", alias="AZURE_OPENAI_API_VERSION")
    
    # Parámetros del Agente
    agent_temperature: float = Field(default=0.0, alias="AGENT_TEMPERATURE")
    agent_max_tokens: int = Field(default=1500, alias="AGENT_MAX_TOKENS")
    memory_window_k: int = Field(default=10, alias="MEMORY_WINDOW_K")
    
    # ======================
    # Configuración de Almacenamiento
    # ======================
    use_azure_blob: bool = Field(default=False, alias="USE_AZURE_BLOB")
    azure_storage_connection_string: Optional[str] = Field(default=None, alias="AZURE_STORAGE_CONNECTION_STRING")
    azure_storage_container_name: str = Field(default="genia-documents", alias="AZURE_STORAGE_CONTAINER_NAME")
    local_storage_path: str = Field(default="./data/documents", alias="LOCAL_STORAGE_PATH")
    
    # ======================
    # Configuración Cosmos DB
    # ======================
    use_azure_cosmos: bool = Field(default=False, alias="USE_AZURE_COSMOS")
    azure_cosmos_endpoint: Optional[str] = Field(default=None, alias="AZURE_COSMOS_ENDPOINT")
    azure_cosmos_key: Optional[str] = Field(default=None, alias="AZURE_COSMOS_KEY")
    azure_cosmos_database: str = Field(default="genia", alias="AZURE_COSMOS_DATABASE")
    azure_cosmos_container: str = Field(default="conversations", alias="AZURE_COSMOS_CONTAINER")
    azure_cosmos_documents_container: str = Field(default="documents", alias="AZURE_COSMOS_DOCUMENTS_CONTAINER")
    
    # ======================
    # Vector Store (FAISS Local)
    # ======================
    local_vector_store_path: str = Field(default="./data/vector_store", alias="LOCAL_VECTOR_STORE_PATH")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    
    # Parámetros RAG
    chunk_size: int = Field(default=1500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=300, alias="CHUNK_OVERLAP")
    search_k_retrieval: int = Field(default=5, alias="SEARCH_K_RETRIEVAL")
    
    # ======================
    # Azure AI Search (Para Migración Azure)
    # ======================
    use_azure_search: bool = Field(default=False, alias="USE_AZURE_SEARCH")
    azure_search_endpoint: Optional[str] = Field(default=None, alias="AZURE_SEARCH_ENDPOINT")
    azure_search_key: Optional[str] = Field(default=None, alias="AZURE_SEARCH_KEY")
    azure_search_index_name: str = Field(default="genia-documents", alias="AZURE_SEARCH_INDEX_NAME")
    
    # Modo Indexador: Azure procesa documentos desde Blob Storage automaticamente
    # Si es True, al subir un archivo se activa el indexador en lugar de indexar manualmente
    use_azure_search_indexer_mode: bool = Field(default=True, alias="USE_AZURE_SEARCH_INDEXER_MODE")
    # Nombre del indexador en Azure AI Search (creado via Import Data en el portal)
    azure_search_indexer_name: str = Field(default="rag-1768774420992-indexer", alias="AZURE_SEARCH_INDEXER_NAME")
    # Nombres de campos del indice (ajustar segun la configuracion del Indexador)
    azure_search_vector_field: str = Field(default="text_vector", alias="AZURE_SEARCH_VECTOR_FIELD")
    azure_search_content_field: str = Field(default="chunk", alias="AZURE_SEARCH_CONTENT_FIELD")
    azure_search_title_field: str = Field(default="title", alias="AZURE_SEARCH_TITLE_FIELD")
    
    # ======================
    # Límites de Carga de Archivos
    # ======================
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    allowed_extensions: str = Field(default=".pdf,.txt,.docx,.doc", alias="ALLOWED_EXTENSIONS")
    
    @property
    def max_file_size_bytes(self) -> int:
        """Tamaño máximo de archivo en bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def allowed_extensions_set(self) -> set:
        """Extensiones de archivo permitidas como conjunto."""
        return {ext.strip().lower() for ext in self.allowed_extensions.split(",")}
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    def get_llm_config(self) -> dict:
        """Obtener configuración de LLM basada en ajustes actuales."""
        if self.use_azure_openai:
            return {
                "type": "azure",
                "api_key": self.azure_openai_api_key,
                "endpoint": self.azure_openai_endpoint,
                "deployment_name": self.azure_openai_deployment_name,
                "api_version": self.azure_openai_api_version,
            }
        return {
            "type": "openai",
            "api_key": self.openai_api_key,
            "model": self.openai_model,
        }


# Instancia global de configuración
settings = Settings()

# Verificación de carga
print("--- VERIFICACIÓN DE CARGA ---")
print(f"Modo Debug: {settings.debug}")
print(f"Azure Activado: {settings.use_azure_openai}")

# Ojo: No imprimas claves completas en producción, solo los primeros caracteres para validar
if settings.openai_api_key:
    print(f"Key OpenAI: {settings.openai_api_key[:5]}...")
else:
    print("Key OpenAI: NO CARGADA (es None)")

if settings.azure_openai_api_key:
    print(f"Key Azure: {settings.azure_openai_api_key[:5]}...")
else:
    print("Key Azure: NO CARGADA (es None)")
