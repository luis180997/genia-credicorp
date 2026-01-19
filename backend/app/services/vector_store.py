"""
Genia AI Agent - Servicio de Vector Store
Soporta tanto FAISS (local) como Azure AI Search para recuperación RAG.
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple, Any
import hashlib
import re

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain.schema import Document
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    pass

from ..config import settings


class VectorStoreService:
    """
    Servicio de vector store que soporta tanto FAISS como Azure AI Search.
    
    Alternar entre backends:
    - USE_AZURE_SEARCH=false -> Usa FAISS (local)
    - USE_AZURE_SEARCH=true -> Usa Azure AI Search (cloud)
    """
    
    def __init__(self):
        self.use_azure_search = settings.use_azure_search
        self.use_indexer_mode = settings.use_azure_search_indexer_mode
        self.vector_store_path = Path(settings.local_vector_store_path)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Cliente SDK nativo para modo Indexador
        self._search_client = None
        
        # Inicializar embeddings (no necesario en modo Indexador, pero lo mantenemos para compatibilidad)
        self.embeddings = self._init_embeddings()
        
        # =============================================
        # Configuración del Divisor de Texto Semántico
        # =============================================
        # Separadores ordenados por prioridad semántica.
        # NOTA: NO incluimos viñetas (bullets) como separadores de alta prioridad
        # para evitar crear fragmentos de items individuales de listas.
        # Las listas se mantendrán juntas hasta que excedan el tamaño del fragmento.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",           # 1. Párrafos (máxima prioridad)
                "\n# ",           # 2. Encabezados Markdown H1
                "\n## ",          # 3. Encabezados Markdown H2
                "\n### ",         # 4. Encabezados Markdown H3
                "\n---",          # 5. Separadores horizontales
                "\n",             # 6. Salto de línea simple
                ". ",             # 7. Fin de oración
                "; ",             # 8. Punto y coma
                " ",              # 9. Espacio
                ""                # 10. Último recurso (carácter por carácter)
            ]
        )
        
        # Inicializar el almacén de vectores adecuado
        self.vector_store: Any = None
        
        if self.use_azure_search:
            if self.use_indexer_mode:
                self._init_azure_search_indexer_mode()
            else:
                self._init_azure_search()
        else:
            self._init_faiss()
    
    def _init_embeddings(self):
        """Inicializar modelo de embeddings (OpenAI o Azure OpenAI)."""
        if settings.use_azure_openai:
            from langchain_openai import AzureOpenAIEmbeddings
            return AzureOpenAIEmbeddings(
                azure_deployment=settings.embedding_model,
                openai_api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version
            )
        else:
            return OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key
            )
    
    # =========================================
    # Backend FAISS (Local)
    # =========================================
    
    def _init_faiss(self):
        """Inicializar almacén de vectores FAISS."""
        
        index_path = self.vector_store_path / "index.faiss"
        if index_path.exists():
            try:
                self.vector_store = FAISS.load_local(
                    str(self.vector_store_path),
                    self.embeddings,
                    #allow_dangerous_deserialization=True
                )
                print(f"Almacen FAISS cargado desde {self.vector_store_path}")
            except Exception as e:
                print(f"No se pudo cargar el almacen FAISS: {e}")
                self.vector_store = None
        else:
            print("Almacen FAISS inicializado (vacio)")
    
    def _save_faiss(self):
        """Guardar almacén de vectores FAISS en disco."""
        if self.vector_store is not None and not self.use_azure_search:
            self.vector_store.save_local(str(self.vector_store_path))
    
    # =========================================
    # Backend Azure AI Search (Cloud)
    # =========================================
    
    def _init_azure_search(self):
        """
        Inicializar almacén de vectores Azure AI Search.
        
        El wrapper de AzureSearch de LangChain automáticamente:
        1. Creará el índice si no existe
        2. Configurará perfiles de búsqueda vectorial
        3. Configurará los campos necesarios
        
        Nota: Esto NO crea un Indexador. Los documentos se suben
        programáticamente vía método add_document().
        """
        try:
            from langchain_community.vectorstores.azuresearch import AzureSearch
            from azure.search.documents.indexes.models import (
                SearchableField,
                SearchField,
                SimpleField,
                SearchFieldDataType,
                VectorSearch,
                HnswAlgorithmConfiguration,
                VectorSearchProfile,
            )
            
            # Validar configuraciones requeridas
            if not settings.azure_search_endpoint:
                raise ValueError("AZURE_SEARCH_ENDPOINT is required")
            if not settings.azure_search_key:
                raise ValueError("AZURE_SEARCH_KEY is required")
            
            # Inicializar Azure AI Search con definición EXPLÍCITA de campos
            # Esto corrige el error 'vectorSearchProfile set'
            
            fields = [
                SimpleField(
                    name="id",
                    type=SearchFieldDataType.String,
                    key=True,
                    filterable=True,
                ),
                SearchableField(
                    name="content",
                    type=SearchFieldDataType.String,
                    searchable=True,
                ),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="myHnswProfile"
                ),
                SearchableField(
                    name="metadata",
                    type=SearchFieldDataType.String,
                    searchable=False,
                ),
                SimpleField(
                    name="source",
                    type=SearchFieldDataType.String,
                    filterable=True,
                ),
            ]
            
            # 2. Definir Configuración de Búsqueda Vectorial (REQUERIDO)
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(name="myHnsw"),
                ],
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="myHnsw",
                    ),
                ],
            )
            
            self.vector_store = AzureSearch(
                azure_search_endpoint=settings.azure_search_endpoint,
                azure_search_key=settings.azure_search_key,
                index_name=settings.azure_search_index_name,
                embedding_function=self.embeddings.embed_query,
                fields=fields,
                vector_search=vector_search,
            )
            
            print(f"Conectado a Azure AI Search: {settings.azure_search_index_name}")
            print(f"  -> Endpoint: {settings.azure_search_endpoint}")
            print(f"  -> El indice se creara automaticamente al subir el primer documento")
            
        except ImportError as e:
            print(f"azure-search-documents no instalado: {e}")
            print("  -> Usando FAISS como alternativa")
            self.use_azure_search = False
            self._init_faiss()
        except ValueError as e:
            print(f"Error de configuracion en Azure AI Search: {e}")
            print("  -> Usando FAISS como alternativa")
            self.use_azure_search = False
            self._init_faiss()
        except Exception as e:
            print(f"Fallo al inicializar Azure AI Search: {e}")
            print("  -> Usando FAISS como alternativa")
            self.use_azure_search = False
            self._init_faiss()
    
    def _init_azure_search_indexer_mode(self):
        """
        Inicializar Azure AI Search en MODO INDEXADOR.
        
        En este modo:
        1. El Indexador de Azure procesa documentos desde Blob Storage
        2. Usamos SDK nativo para buscar (no LangChain wrapper)
        3. Los campos del indice pueden tener nombres personalizados
        """
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents import SearchClient
            
            if not settings.azure_search_endpoint:
                raise ValueError("AZURE_SEARCH_ENDPOINT is required")
            if not settings.azure_search_key:
                raise ValueError("AZURE_SEARCH_KEY is required")
            
            # Crear cliente de busqueda nativo
            self._search_client = SearchClient(
                endpoint=settings.azure_search_endpoint,
                index_name=settings.azure_search_index_name,
                credential=AzureKeyCredential(settings.azure_search_key)
            )
            
            # Marcar como disponible
            self.vector_store = {"type": "azure_indexer_mode"}
            
            print(f"Conectado a Azure AI Search (MODO INDEXADOR): {settings.azure_search_index_name}")
            print(f"  -> Campo vectorial: {settings.azure_search_vector_field}")
            print(f"  -> Campo contenido: {settings.azure_search_content_field}")
            
        except ImportError as e:
            print(f"azure-search-documents no instalado: {e}")
            self.use_azure_search = False
            self.use_indexer_mode = False
            self._init_faiss()
        except Exception as e:
            print(f"Fallo en modo Indexador de Azure AI Search: {e}")
            self.use_azure_search = False
            self.use_indexer_mode = False
            self._init_faiss()
    
    # =========================================
    # Interfaz Común (Ambos Backends)
    # =========================================
    
    def add_document(self, file_path: str, filename: str) -> int:
        """
        Agregar un documento al almacén de vectores.
        
        Args:
            file_path: Ruta al archivo del documento
            filename: Nombre original del archivo (para metadatos)
            
        Returns:
            Número de fragmentos creados
        """
        # Cargar documento según el tipo de archivo
        documents = self._load_document(file_path, filename)
        
        if not documents:
            return 0
        
        # Dividir en fragmentos
        chunks = self.text_splitter.split_documents(documents)
        
        # =============================================
        # Post-procesamiento Semántico
        # =============================================
        # Enriquecer cada fragmento con contexto de la sección padre
        chunks = self._enrich_chunks_with_context(chunks, filename)
        
        # Enriquecer metadatos para base semántica
        document_type = self._infer_document_type(filename)
        indexed_at = self._get_timestamp()
        
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'source': filename,
                'chunk_id': self._generate_chunk_id(chunk.page_content),
                'chunk_index': i,
                'total_chunks': len(chunks),
                'document_type': document_type,
                'indexed_at': indexed_at,
            })
        
        # Agregar al almacén de vectores
        if self.use_azure_search:
            # Azure AI Search - agregar documentos
            self.vector_store.add_documents(chunks)
        else:
            # FAISS
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.vector_store.add_documents(chunks)
            self._save_faiss()
        
        print(f"Added {len(chunks)} chunks from {filename} (type: {document_type}) to {'Azure AI Search' if self.use_azure_search else 'FAISS'}")
        return len(chunks)
    
    def _enrich_chunks_with_context(self, chunks: List[Document], filename: str) -> List[Document]:
        """
        Enriquece cada fragmento con contexto semántico.
        
        Problema que resuelve:
        Si un fragmento dice "Gastos: $30M" pero perdió el título "Resultados Trimestrales",
        la búsqueda no podrá asociar el dato con su contexto.
        
        Solución:
        1. Detectar encabezados/secciones en el texto completo
        2. Propagar el último encabezado visto a fragmentos huérfanos
        3. Agregar contexto como prefijo y metadatos
        """
        
        # Patrones para detectar encabezados (comunes en documentos financieros)
        header_patterns = [
            r'^#{1,3}\s+(.+)$',           # Markdown: # Título, ## Subtítulo
            r'^([A-Z][A-Z\s]{5,})$',      # MAYÚSCULAS: RESULTADOS TRIMESTRALES
            r'^(\d+\.\s+[A-Z].+)$',       # Numerado: 1. Introducción
            r'^([IVXLC]+\.\s+.+)$',       # Romano: I. Resumen Ejecutivo
        ]
        
        current_section = filename  # Por defecto: usar nombre del archivo como contexto
        enriched_chunks = []
        
        for chunk in chunks:
            content = chunk.page_content
            
            # Buscar si este fragmento contiene un encabezado
            for pattern in header_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                if matches:
                    # Actualizar la sección actual
                    current_section = matches[0].strip()[:100]  # Limitar longitud
                    break
            
            # Agregar contexto de sección al fragmento
            chunk.metadata['section'] = current_section
            
            # Si el fragmento NO comienza con su propio encabezado,
            # agregar el contexto de la sección como prefijo
            has_own_header = any(
                re.match(pattern, content.split('\n')[0]) 
                for pattern in header_patterns
            )
            
            if not has_own_header and current_section != filename:
                # Prefijo semántico para dar contexto
                context_prefix = f"[Sección: {current_section}]\n\n"
                chunk.page_content = context_prefix + content
            
            enriched_chunks.append(chunk)
        
        return enriched_chunks
    
    def _infer_document_type(self, filename: str) -> str:
        """
        Inferir tipo de documento desde el nombre de archivo para metadatos semánticos más ricos.
        
        En producción, esto podría usar clasificación NLP o etiquetado manual.
        """
        filename_lower = filename.lower()
        
        # Coincidencia de patrones para tipos de documentos financieros comunes
        if any(word in filename_lower for word in ['politica', 'policy', 'procedimiento']):
            return 'politica'
        elif any(word in filename_lower for word in ['reporte', 'report', 'informe']):
            return 'reporte'
        elif any(word in filename_lower for word in ['manual', 'guia', 'guide']):
            return 'manual'
        elif any(word in filename_lower for word in ['prospecto', 'prospectus', 'ficha']):
            return 'prospecto'
        elif any(word in filename_lower for word in ['contrato', 'contract', 'acuerdo']):
            return 'contrato'
        elif any(word in filename_lower for word in ['faq', 'preguntas']):
            return 'faq'
        else:
            return 'general'
    
    @staticmethod
    def _get_timestamp() -> str:
        """Obtener timestamp actual para metadatos."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def _load_document(self, file_path: str, filename: str) -> List[Document]:
        """Cargar documento según el tipo de archivo."""
        ext = Path(filename).suffix.lower()
        
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            else:
                # Intentar como archivo de texto
                loader = TextLoader(file_path, encoding='utf-8')
            
            return loader.load()
        except Exception as e:
            print(f"Error loading document {filename}: {e}")
            return []
    
    def search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """
        Buscar documentos relevantes.
        
        Args:
            query: Consulta de búsqueda
            k: Número de resultados a retornar (default: settings.search_k_retrieval)
            
        Returns:
            Lista de tuplas (documento, puntaje)
        """
        if k is None:
            k = settings.search_k_retrieval
            
        print(f"Searching for: '{query}' (k={k})")
        
        if self.vector_store is None:
            print("   Vector store is None - no documents indexed")
            return []
        
        try:
            # =====================================================
            # MODO INDEXADOR: Usar SDK nativo con VectorizableTextQuery
            # =====================================================
            if self.use_azure_search and self.use_indexer_mode:
                return self._search_indexer_mode(query, k)
            
            # =====================================================
            # MODO PUSH: Usar wrapper de LangChain
            # =====================================================
            if self.use_azure_search:
                # Azure AI Search: usar similarity_search
                results = self.vector_store.similarity_search(query, k=k)
                results_with_score = [(doc, 1.0) for doc in results]
                print(f"   Found {len(results_with_score)} results (Azure AI Search)")
                return results_with_score
            else:
                # FAISS: soporta similarity_search_with_score
                results = self.vector_store.similarity_search_with_score(query, k=k)
                print(f"   Found {len(results)} results (FAISS)")
                return results
        except Exception as e:
            import traceback
            print(f"   Search error: {type(e).__name__}: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            return []
    
    def _search_indexer_mode(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """
        Busqueda usando SDK nativo para indices creados por Indexador.
        
        Usa VectorizableTextQuery para que Azure vectorice la query internamente.
        """
        from azure.search.documents.models import VectorizableTextQuery
        
        # Query vectorial con texto (Azure lo vectoriza)
        vector_query = VectorizableTextQuery(
            text=query,
            k_nearest_neighbors=k,
            fields=settings.azure_search_vector_field
        )
        
        # Ejecutar busqueda hibrida (texto + vector)
        results = self._search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=[settings.azure_search_content_field, settings.azure_search_title_field],
            top=k
        )
        
        # Convertir a formato Document de LangChain
        documents_with_scores = []
        for result in results:
            content = result.get(settings.azure_search_content_field, "")
            title = result.get(settings.azure_search_title_field, "Documento Azure")
            score = result.get("@search.score", 1.0)
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": title,
                    "search_score": score
                }
            )
            documents_with_scores.append((doc, score))
        
        print(f"   Found {len(documents_with_scores)} results (Azure AI Search - INDEXER MODE)")
        return documents_with_scores
    
    def get_retriever(self, k: int = None):
        """
        Obtener un retriever para usar con LangChain.
        
        Args:
            k: Número de documentos a recuperar (default: settings.search_k_retrieval)
            
        Returns:
            Objeto Retriever o None
        """
        if k is None:
            k = settings.search_k_retrieval
            
        if self.vector_store is None:
            return None
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    
    def delete_document(self, filename: str) -> bool:
        """
        Eliminar un documento del almacén de vectores.
        
        Args:
            filename: Nombre del archivo a eliminar
            
        Returns:
            True si fue exitoso
        """
        if self.use_azure_search:
            # Azure AI Search soporta eliminación por filtro
            try:
                # Nota: Requiere implementación personalizada con Azure SDK
                print(f"Document deletion for Azure AI Search requires custom filter")
                return False
            except Exception as e:
                print(f"Delete error: {e}")
                return False
        else:
            # FAISS no soporta eliminación directa
            print(f"Document deletion not fully implemented for FAISS")
            return False
    
    def clear(self):
        """Limpiar todo el almacén de vectores."""
        if self.use_azure_search:
            # Para Azure AI Search, necesitaríamos eliminar y recrear el índice
            print("Clear not implemented for Azure AI Search (delete index manually)")
        else:
            self.vector_store = None
            # Eliminar archivos persistidos
            for file in self.vector_store_path.iterdir():
                if file.suffix in ['.faiss', '.pkl']:
                    file.unlink()
            print("FAISS vector store cleared")
    
    @staticmethod
    def _generate_chunk_id(content: str) -> str:
        """Generar un ID único para un fragmento."""
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def get_status(self) -> dict:
        """Obtener estado del servicio para verificación de salud."""
        doc_count = 0
        
        if self.use_azure_search:
            # Azure AI Search - no se puede obtener conteo fácilmente sin consulta
            store_type = "azure_ai_search"
            path_or_index = settings.azure_search_index_name
        else:
            store_type = "faiss"
            path_or_index = str(self.vector_store_path)
            if self.vector_store is not None:
                try:
                    doc_count = self.vector_store.index.ntotal
                except:
                    pass
        
        return {
            "type": store_type,
            "available": self.vector_store is not None,
            "path": path_or_index,
            "document_count": doc_count,
            "embedding_model": settings.embedding_model
        }
