# Genia AI Agent

Agente de Inteligencia Artificial para consultas de inversión con capacidades RAG (Retrieval-Augmented Generation). Desarrollado para Credicorp Capital.

https://caso-credicorp.azurewebsites.net/

### Componentes Principales

| Capa | Componente | Descripción |
|------|------------|-------------|
| **Usuario** | Streamlit Frontend | Interfaz de chat interactiva |
| **Aplicación** | FastAPI + LangChain | Backend con agente inteligente |
| **Inteligencia** | Azure OpenAI | Modelo GPT-4o-mini para razonamiento |
| **Conocimiento** | Azure AI Search | Base semántica vectorial (RAG) |
| **Pipeline de Datos** | Azure Data Factory | Orquestación y centralización de fuentes |
| **Almacenamiento** | Azure Blob Storage | Contenedor de usuarios + Contenedor unificado |
| **Persistencia** | Azure Cosmos DB | Historial de conversaciones |


## Herramientas Disponibles

| Herramienta | Descripción | Ejemplo de Uso |
|-------------|-------------|----------------|
| `search_internal_documents` | Busca en documentos internos | "¿Cuál es la comisión del fondo Visión?" |
| `get_stock_info` | Cotizaciones de acciones | "¿Cómo está la acción de Apple?" |
| `get_exchange_rate` | Tipos de cambio | "¿Cuál es el tipo de cambio USD/PEN?" |
| `search_financial_web` | Noticias financieras | "¿Qué pasó con el mercado hoy?" |
| `financial_calculator` | Cálculos financieros | "Calcula el rendimiento de 10000 al 5%" |


## Decisiones Técnicas y Trade-offs

| Decisión | Justificación | Trade-off |
|----------|---------------|-----------|
| **GPT-4o-mini** | Balance costo/calidad óptimo | Menos preciso que GPT-4o pero 10x más económico |
| **Azure AI Search (Indexador)** | Azure gestiona chunking y embeddings | Menos control sobre el proceso, pero menor mantenimiento |
| **LangChain Agent** | Framework maduro con herramientas listas | Más dependencias, pero productividad alta |
| **ConversationSummaryBufferMemory** | Evita overflow de tokens | Resume mensajes antiguos, pierde algo de detalle |
| **Cosmos DB para historial** | Escalable y serverless | Más complejo que SQLite, pero preparado para producción |
| **Fallback a FAISS** | Resiliencia si Azure falla | Requiere mantener dos backends |

## Pipeline de Datos

1. **Carga**: Usuario sube documento → Backend → Azure Blob Storage (Contenedor de Usuarios)
2. **Centralización**: Azure Data Factory copia al Contenedor Unificado (junto con otras fuentes)
3. **Indexación**: Azure AI Search indexa automáticamente desde el contenedor unificado
4. **Consulta**: El agente consulta Azure AI Search para obtener contexto RAG

> **Nota**: El indexador tiene un límite de 1 ejecución cada 3 minutos en el tier básico.

## API Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/chat` | Enviar mensaje al agente |
| `POST` | `/documents/upload` | Subir documento |
| `POST` | `/indexer/run` | Ejecutar indexador manualmente |
| `GET` | `/documents` | Listar documentos |
| `GET` | `/health` | Estado de servicios |

## Consideraciones de Performance

| Dimensión | Valor | Comentario |
|-----------|-------|------------|
| **Latencia** | ~3-5 seg | Dominado por llamada al LLM |
| **Costo por consulta** | ~$0.002 | GPT-4o-mini es económico |
| **Chunks recuperados** | Top-5 | Balance contexto vs ruido |
| **Embeddings** | text-embedding-3-small | 1536 dimensiones |


## Consideraciones Productivas (No Implementadas)

- **Seguridad**: Azure Key Vault para credenciales, Azure AD para autenticación
- **Gobernanza**: RBAC por área de negocio (Inversiones, Compliance, Back Office)
- **Observabilidad**: Azure Monitor, Log Analytics para métricas
