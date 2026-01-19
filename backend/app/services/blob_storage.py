"""
Genia AI Agent - Blob Storage Service
Supports both Azure Blob Storage and local filesystem.
"""
import os
import shutil
from pathlib import Path
from typing import List, Optional, BinaryIO
from datetime import datetime

from ..config import settings


class BlobStorageService:
    """
    Document storage service with Azure Blob Storage support.
    Falls back to local filesystem when Azure is not configured.
    
    For Azure migration:
    1. Set USE_AZURE_BLOB=true in .env
    2. Provide AZURE_STORAGE_CONNECTION_STRING
    3. Provide AZURE_STORAGE_CONTAINER_NAME
    """
    
    def __init__(self):
        self.use_azure = settings.use_azure_blob
        self.local_path = Path(settings.local_storage_path)
        
        if self.use_azure:
            self._init_azure_client()

        else:
            self._init_local_storage()
    
    def _init_azure_client(self):
        """Initialize Azure Blob Storage client."""
        try:
            from azure.storage.blob import BlobServiceClient
            
            self.blob_service = BlobServiceClient.from_connection_string(
                settings.azure_storage_connection_string
            )
            self.container_client = self.blob_service.get_container_client(
                settings.azure_storage_container_name
            )
            # Create container if not exists
            if not self.container_client.exists():
                self.container_client.create_container()
            print(f"[INFO] Conectado a Azure Blob Storage: {settings.azure_storage_container_name}")
        except Exception as e:
            print(f"[ERROR] Azure Blob Storage no disponible, usando almacenamiento local: {e}")
            self.use_azure = False
            self._init_local_storage()
    
    def _init_local_storage(self):
        """Initialize local filesystem storage."""
        self.local_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Usando almacenamiento local: {self.local_path.absolute()}")
    
    def upload_file(self, file: BinaryIO, filename: str) -> str:
        """
        Upload a file to storage.
        
        Args:
            file: File-like object to upload
            filename: Name for the stored file
            
        Returns:
            Path/URL to the stored file
        """
        # Sanitize filename
        safe_filename = self._sanitize_filename(filename)
        
        if self.use_azure:
            print("Subiendo a Azure Blob Storage...")
            return self._upload_to_azure(file, safe_filename)
        print("Subiendo a almacenamiento local...")
        return self._upload_to_local(file, safe_filename)
    
    def _upload_to_azure(self, file: BinaryIO, filename: str) -> str:
        """Upload file to Azure Blob Storage."""
        blob_client = self.container_client.get_blob_client(filename)
        blob_client.upload_blob(file, overwrite=True)
        return blob_client.url
    
    def _upload_to_local(self, file: BinaryIO, filename: str) -> str:
        """Upload file to local filesystem."""
        file_path = self.local_path / filename
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file, f)
        return str(file_path.absolute())
    
    def download_file(self, filename: str) -> Optional[bytes]:
        """
        Download a file from storage.
        
        Args:
            filename: Name of the file to download
            
        Returns:
            File contents as bytes, or None if not found
        """
        if self.use_azure:
            return self._download_from_azure(filename)
        return self._download_from_local(filename)
    
    def _download_from_azure(self, filename: str) -> Optional[bytes]:
        """Download file from Azure Blob Storage."""
        try:
            blob_client = self.container_client.get_blob_client(filename)
            return blob_client.download_blob().readall()
        except Exception:
            return None
    
    def _download_from_local(self, filename: str) -> Optional[bytes]:
        """Download file from local filesystem."""
        file_path = self.local_path / filename
        if file_path.exists():
            return file_path.read_bytes()
        return None
    
    def list_files(self) -> List[str]:
        """List all files in storage."""
        if self.use_azure:
            return self._list_azure_files()
        return self._list_local_files()
    
    def _list_azure_files(self) -> List[str]:
        """List files in Azure Blob Storage."""
        return [blob.name for blob in self.container_client.list_blobs()]
    
    def _list_local_files(self) -> List[str]:
        """List files in local storage."""
        return [f.name for f in self.local_path.iterdir() if f.is_file()]
    
    def delete_file(self, filename: str) -> bool:
        """Delete a file from storage."""
        if self.use_azure:
            return self._delete_from_azure(filename)
        return self._delete_from_local(filename)
    
    def _delete_from_azure(self, filename: str) -> bool:
        """Delete file from Azure Blob Storage."""
        try:
            blob_client = self.container_client.get_blob_client(filename)
            blob_client.delete_blob()
            return True
        except Exception:
            return False
    
    def _delete_from_local(self, filename: str) -> bool:
        """Delete file from local storage."""
        file_path = self.local_path / filename
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def get_file_path(self, filename: str) -> Optional[str]:
        """Get the full path to a file (for local processing)."""
        if self.use_azure:
            # For Azure, download to temp and return path
            content = self._download_from_azure(filename)
            if content:
                temp_path = self.local_path / f"_temp_{filename}"
                temp_path.write_bytes(content)
                return str(temp_path)
            return None
        
        file_path = self.local_path / filename
        return str(file_path) if file_path.exists() else None
    
    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove path components
        filename = os.path.basename(filename)
        # Replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        return filename
    
    def get_status(self) -> dict:
        """Get service status for health check."""
        return {
            "type": "azure_blob" if self.use_azure else "local_filesystem",
            "available": True,
            "path": str(self.local_path) if not self.use_azure else settings.azure_storage_container_name,
            "file_count": len(self.list_files())
        }
    
    def run_indexer(self, indexer_name: str) -> dict:
        """
        Ejecutar el indexador de Azure AI Search inmediatamente.
        
        Esto activa el indexador sin esperar el intervalo programado,
        util despues de subir un nuevo documento.
        
        Args:
            indexer_name: Nombre del indexador en Azure AI Search
            
        Returns:
            dict con status y mensaje
        """
        if not settings.use_azure_search:
            return {"status": "skipped", "message": "Azure AI Search no esta habilitado"}
        
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents.indexes import SearchIndexerClient
            
            # Crear cliente del indexador
            indexer_client = SearchIndexerClient(
                endpoint=settings.azure_search_endpoint,
                credential=AzureKeyCredential(settings.azure_search_key)
            )
            
            # Ejecutar el indexador inmediatamente
            indexer_client.run_indexer(indexer_name)
            
            print(f"[INFO] Indexador '{indexer_name}' ejecutado exitosamente")
            return {
                "status": "success", 
                "message": f"Indexador '{indexer_name}' iniciado. Los documentos se procesaran en unos segundos."
            }
            
        except Exception as e:
            print(f"[ERROR] Error ejecutando indexador: {e}")
            return {"status": "error", "message": str(e)}
