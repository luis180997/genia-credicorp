"""
Genia AI Agent - Aplicación Principal FastAPI
API RESTful para el asistente de inversiones potenciado por RAG.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .models.schemas import (
    ChatRequest, 
    ChatResponse, 
    DocumentUploadResponse,
    HealthResponse,
    Conversation
)
from .services.blob_storage import BlobStorageService
from .services.cosmos_db import CosmosDBService
from .services.vector_store import VectorStoreService
from .services.document_registry import DocumentRegistry
from .agent.rag_agent import RAGAgent


# Instancias globales de servicios
blob_service: Optional[BlobStorageService] = None
cosmos_service: Optional[CosmosDBService] = None
vector_service: Optional[VectorStoreService] = None
document_registry: Optional[DocumentRegistry] = None
rag_agent: Optional[RAGAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestor de ciclo de vida de la aplicación - inicializa servicios al inicio."""
    global blob_service, cosmos_service, vector_service, document_registry, rag_agent
    
    print("\n" + "="*50)
    print("Iniciando Backend del Agente AI Genia")
    print("="*50 + "\n")
    
    # Inicializar servicios
    blob_service = BlobStorageService()
    cosmos_service = CosmosDBService()
    vector_service = VectorStoreService()
    document_registry = DocumentRegistry()
    rag_agent = RAGAgent(vector_service, cosmos_service)
    
    print("\n" + "="*50)
    print("Todos los servicios inicializados exitosamente")
    print("="*50 + "\n")
    
    yield
    
    # Limpieza al apagar
    print("\nDeteniendo Backend del Agente AI Genia")


# Crear aplicación FastAPI
app = FastAPI(
    title=settings.app_name,
    description="Agente de IA para consultas de inversión con RAG - Credicorp Capital",
    version=settings.app_version,
    lifespan=lifespan
)

# Middleware CORS para frontend Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=JSONResponse)
async def root():
    """Endpoint raíz con información de la API."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Genia AI Agent - Asistente de inversiones de Credicorp Capital",
        "endpoints": {
            "chat": "POST /chat",
            "upload": "POST /documents/upload",
            "conversations": "GET /conversations/{session_id}",
            "health": "GET /health"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de verificación de salud con estado de servicios."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        services={
            "blob_storage": blob_service.get_status() if blob_service else {"available": False},
            "cosmos_db": cosmos_service.get_status() if cosmos_service else {"available": False},
            "vector_store": vector_service.get_status() if vector_service else {"available": False},
            "agent": rag_agent.get_status() if rag_agent else {"available": False}
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Procesar un mensaje de chat usando RAG.
    
    - **message**: La pregunta o mensaje del usuario
    - **session_id**: Identificador único de sesión para continuidad de conversación
    """
    if rag_agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        response, sources = await rag_agent.chat(
            message=request.message,
            session_id=request.session_id
        )

        return ChatResponse(
            response=response,
            session_id=request.session_id,
            sources=sources
        )
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing message: {str(e)}"
        )


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Subir un documento para indexación RAG.
    
    Formatos soportados: PDF, TXT, DOCX
    Tamaño máximo: Configurable vía MAX_FILE_SIZE_MB (default 50MB)
    
    Retorna error si:
    - Tipo de archivo no soportado
    - Archivo muy grande
    - Contenido duplicado ya indexado
    """
    if blob_service is None or vector_service is None or document_registry is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    # Validar tipo de archivo
    ext = Path(file.filename).suffix.lower()
    
    if ext not in settings.allowed_extensions_set:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {settings.allowed_extensions}"
        )
    
    # Leer contenido para validación de tamaño y hash
    try:
        content = await file.read()
        file_size = len(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    
    # Validar tamaño de archivo
    if file_size > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB, received: {file_size / (1024*1024):.2f}MB"
        )
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file not allowed")
    
    # Calcular hash de contenido para detección de duplicados
    import hashlib
    content_hash = hashlib.sha256(content).hexdigest()
    print(f"[DEBUG] Hash del archivo: {content_hash[:16]}...")
    
    # Verificar duplicados
    existing_doc = document_registry.is_duplicate(content_hash)
    
    if existing_doc:
        print(f"[DEBUG] Documento duplicado detectado: {existing_doc.get('filename')}")
        return DocumentUploadResponse(
            filename=file.filename,
            status="duplicate",
            message=f"Documento con contenido identico ya existe como '{existing_doc['filename']}'",
            chunks_created=0
        )
    
    try:
        # Crear objeto BytesIO para servicio blob
        from io import BytesIO
        file_obj = BytesIO(content)
        
        # Subir a blob storage
        file_path = blob_service.upload_file(file_obj, file.filename)
        
        # =====================================================
        # MODO INDEXADOR: Solo subir, el usuario sincroniza despues
        # =====================================================
        if settings.use_azure_search_indexer_mode:
            # En modo indexador, solo subimos. El usuario ejecuta el indexador manualmente
            # Registrar documento (sin chunks ya que Azure los crea)
            document_registry.register_document(
                content_hash=content_hash,
                filename=file.filename,
                chunks_created=0,  # Azure crea los chunks
                file_size=file_size
            )
            
            return DocumentUploadResponse(
                filename=file.filename,
                status="success",
                message="Documento subido. Usa 'Sincronizar' para indexar.",
                chunks_created=0
            )
        
        # =====================================================
        # MODO MANUAL (PUSH): Nosotros indexamos el documento
        # =====================================================
        # Obtener ruta local para procesamiento
        local_path = blob_service.get_file_path(file.filename)
        
        if local_path is None:
            raise HTTPException(status_code=500, detail="Failed to access uploaded file")
        
        # Indexar en vector store
        chunks_created = vector_service.add_document(local_path, file.filename)
        
        # Registrar documento para prevenir duplicados futuros
        document_registry.register_document(
            content_hash=content_hash,
            filename=file.filename,
            chunks_created=chunks_created,
            file_size=file_size
        )
        
        return DocumentUploadResponse(
            filename=file.filename,
            status="success",
            message=f"Document uploaded and indexed successfully",
            chunks_created=chunks_created
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )


@app.get("/conversations/{session_id}", response_model=Conversation)
async def get_conversation(session_id: str):
    """Obtener historial de conversación para una sesión."""
    if cosmos_service is None:
        raise HTTPException(status_code=503, detail="Cosmos DB service not initialized")
    
    conversation = cosmos_service.get_conversation(session_id)
    return conversation


@app.delete("/conversations/{session_id}")
async def clear_conversation(session_id: str):
    """Limpiar historial de conversación para una sesión."""
    if cosmos_service is None or rag_agent is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    rag_agent.clear_session(session_id)
    
    return {"status": "success", "message": f"Conversation {session_id} cleared"}


@app.get("/documents")
async def list_documents():
    """Listar todos los documentos indexados."""
    if blob_service is None:
        raise HTTPException(status_code=503, detail="Blob service not initialized")
    
    files = blob_service.list_files()
    return {"documents": files, "count": len(files)}


@app.post("/indexer/run")
async def run_indexer():
    """
    Ejecutar el indexador de Azure AI Search manualmente.
    
    Esto sincroniza los documentos nuevos en Blob Storage con el indice.
    Limite: Solo se puede ejecutar cada 3 minutos en el tier basico.
    """
    if blob_service is None:
        raise HTTPException(status_code=503, detail="Blob service not initialized")
    
    if not settings.use_azure_search_indexer_mode:
        return {
            "status": "skipped",
            "message": "Modo indexador no habilitado. Los documentos se indexan manualmente."
        }
    
    result = blob_service.run_indexer(settings.azure_search_indexer_name)
    return result


# Ejecutar con: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
