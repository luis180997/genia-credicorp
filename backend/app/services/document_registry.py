"""
Genia AI Agent - Document Registry Service
Tracks document hashes to prevent duplicate uploads.
Soporta tanto Cosmos DB (Azure) como almacenamiento local JSON.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, BinaryIO
from datetime import datetime

from ..config import settings


class DocumentRegistry:
    """
    Registry for tracking uploaded documents by content hash.
    Prevents duplicate content from being indexed multiple times.
    
    Modos de almacenamiento:
    - USE_AZURE_COSMOS=true: Almacena en Cosmos DB (contenedor 'documents')
    - USE_AZURE_COSMOS=false: Almacena en archivo JSON local
    """
    
    REGISTRY_FILENAME = "document_registry.json"
    
    def __init__(self):
        self.use_azure = settings.use_azure_cosmos
        self._registry: Dict[str, dict] = {}
        
        if self.use_azure:
            self._init_cosmos_client()
        else:
            self._init_local_storage()
    
    def _init_cosmos_client(self):
        """Inicializar cliente de Cosmos DB para registro de documentos."""
        try:
            from azure.cosmos import CosmosClient, PartitionKey
            
            self.cosmos_client = CosmosClient(
                settings.azure_cosmos_endpoint,
                credential=settings.azure_cosmos_key
            )
            
            # Obtener o crear base de datos
            self.database = self.cosmos_client.create_database_if_not_exists(
                id=settings.azure_cosmos_database
            )
            
            # Crear contenedor para documentos si no existe
            self.container = self.database.create_container_if_not_exists(
                id=settings.azure_cosmos_documents_container,
                partition_key=PartitionKey(path="/content_hash")
            )
            
            print(f"[INFO] Registro de documentos conectado a Cosmos DB: {settings.azure_cosmos_database}/{settings.azure_cosmos_documents_container}")
        except Exception as e:
            print(f"[ERROR] Cosmos DB no disponible, usando almacenamiento local: {e}")
            self.use_azure = False
            self._init_local_storage()
    
    def _init_local_storage(self):
        """Inicializar almacenamiento local JSON."""
        self.registry_path = Path(settings.local_storage_path) / self.REGISTRY_FILENAME
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_registry()
    
    def _load_registry(self):
        """Cargar registro desde disco (solo modo local)."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    self._registry = json.load(f)
                print(f"[INFO] Registro de documentos cargado: {len(self._registry)} documentos")
            except Exception as e:
                print(f"[ERROR] No se pudo cargar el registro de documentos: {e}")
                self._registry = {}
        else:
            print("[INFO] Registro de documentos inicializado (vacio)")
    
    def _save_registry(self):
        """Guardar registro en disco (solo modo local)."""
        if not self.use_azure:
            try:
                with open(self.registry_path, 'w', encoding='utf-8') as f:
                    json.dump(self._registry, f, indent=2, default=str)
            except Exception as e:
                print(f"[ERROR] Could not save document registry: {e}")
    
    @staticmethod
    def calculate_hash(file: BinaryIO) -> str:
        """
        Calcular hash SHA-256 del contenido del archivo.
        
        Args:
            file: Objeto archivo a hashear
            
        Returns:
            Hash SHA-256 en hexadecimal
        """
        sha256_hash = hashlib.sha256()
        for chunk in iter(lambda: file.read(8192), b""):
            sha256_hash.update(chunk)
        file.seek(0)
        return sha256_hash.hexdigest()
    
    def is_duplicate(self, content_hash: str) -> Optional[dict]:
        """
        Verificar si el hash de contenido ya existe.
        
        Args:
            content_hash: Hash SHA-256 del contenido
            
        Returns:
            Info del documento existente si es duplicado, None de lo contrario
        """
        if self.use_azure:
            return self._check_cosmos_duplicate(content_hash)
        return self._registry.get(content_hash)
    
    def _check_cosmos_duplicate(self, content_hash: str) -> Optional[dict]:
        """Verificar duplicado en Cosmos DB."""
        try:
            query = f"SELECT * FROM c WHERE c.content_hash = '{content_hash}'"
            items = list(self.container.query_items(query, enable_cross_partition_query=True))
            if items:
                return items[0]
        except Exception as e:
            print(f"[ERROR] Error checking duplicate in Cosmos DB: {e}")
        return None
    
    def register_document(
        self, 
        content_hash: str, 
        filename: str,
        chunks_created: int = 0,
        file_size: int = 0
    ) -> dict:
        """
        Registrar un nuevo documento.
        
        Args:
            content_hash: Hash SHA-256 del contenido
            filename: Nombre original del archivo
            chunks_created: Numero de chunks indexados
            file_size: Tamano del archivo en bytes
            
        Returns:
            Entrada del documento registrado
        """
        entry = {
            "id": content_hash,  # Requerido por Cosmos DB
            "content_hash": content_hash,
            "filename": filename,
            "chunks_created": chunks_created,
            "file_size": file_size,
            "uploaded_at": datetime.utcnow().isoformat(),
        }
        
        if self.use_azure:
            self._save_to_cosmos(entry)
        else:
            self._registry[content_hash] = entry
            self._save_registry()
        
        return entry
    
    def _save_to_cosmos(self, entry: dict):
        """Guardar documento en Cosmos DB."""
        try:
            self.container.upsert_item(entry)
        except Exception as e:
            print(f"[ERROR] Error saving to Cosmos DB: {e}")
    
    def get_document_by_filename(self, filename: str) -> Optional[dict]:
        """Obtener info de documento por nombre de archivo."""
        if self.use_azure:
            try:
                query = f"SELECT * FROM c WHERE c.filename = '{filename}'"
                items = list(self.container.query_items(query, enable_cross_partition_query=True))
                return items[0] if items else None
            except Exception:
                return None
        
        for doc in self._registry.values():
            if doc.get("filename") == filename:
                return doc
        return None
    
    def unregister_document(self, content_hash: str) -> bool:
        """Eliminar documento del registro."""
        if self.use_azure:
            try:
                self.container.delete_item(item=content_hash, partition_key=content_hash)
                return True
            except Exception:
                return False
        
        if content_hash in self._registry:
            del self._registry[content_hash]
            self._save_registry()
            return True
        return False
    
    def list_documents(self) -> list:
        """Listar todos los documentos registrados."""
        if self.use_azure:
            try:
                query = "SELECT * FROM c"
                return list(self.container.query_items(query, enable_cross_partition_query=True))
            except Exception:
                return []
        return list(self._registry.values())
    
    def get_stats(self) -> dict:
        """Obtener estadisticas del registro."""
        documents = self.list_documents()
        total_size = sum(doc.get("file_size", 0) for doc in documents)
        total_chunks = sum(doc.get("chunks_created", 0) for doc in documents)
        return {
            "document_count": len(documents),
            "total_size_bytes": total_size,
            "total_chunks": total_chunks,
            "storage_type": "cosmos_db" if self.use_azure else "local_json"
        }
