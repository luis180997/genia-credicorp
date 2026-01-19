"""
Genia AI Agent - Implementación de RAG Agéntico
Agente basado en LangChain con herramientas para toma de decisiones inteligente.

Este agente puede:
1. Buscar en documentos internos de Credicorp (RAG)
2. Consultar información financiera externa (Búsqueda Web)
3. Realizar cálculos financieros básicos
4. Mantener contexto conversacional
"""
from typing import List, Tuple, Optional
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import Document, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..config import settings
from ..services.vector_store import VectorStoreService
from ..services.cosmos_db import CosmosDBService
from ..models.schemas import MessageRole
from .prompts import SYSTEM_PROMPT
from .tools import get_all_tools


class RAGAgent:
    """
    Agente RAG Agéntico usando Herramientas LangChain.
    
    A diferencia de una Cadena simple (que siempre ejecuta RAG),
    este Agente DECIDE cuándo usar cada herramienta:
    - Documentos internos para políticas/procedimientos
    - Web para información de mercado actualizada
    - Calculadora para operaciones numéricas
    """
    
    def __init__(
        self,
        vector_store_service: VectorStoreService,
        cosmos_service: CosmosDBService
    ):
        self.vector_store = vector_store_service
        self.cosmos_db = cosmos_service
        
        # Inicializar LLM
        self.llm = self._init_llm()
        
        # Inicializar Herramientas
        self.tools = get_all_tools(vector_store_service)
        
        # Crear Agente
        self.agent_executor = self._create_agent()
        
        # Cache de memorias de sesión
        self._memories = {}
        
        print(f"Agente RAG inicializado con {type(self.llm).__name__}")
        print(f"  -> Tools disponibles: {[t.name for t in self.tools]}")
    
    def _init_llm(self):
        """Inicializar el LLM basado en la configuración."""
        if settings.use_azure_openai:
            return AzureChatOpenAI(
                azure_deployment=settings.azure_openai_deployment_name,
                openai_api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                temperature=settings.agent_temperature,
                max_tokens=settings.agent_max_tokens
            )
        else:
            return ChatOpenAI(
                model=settings.openai_model,
                openai_api_key=settings.openai_api_key,
                temperature=settings.agent_temperature,
                max_tokens=settings.agent_max_tokens
            )
    
    def _create_agent(self) -> AgentExecutor:
        """Crear el agente con herramientas."""
        
        # Definir la plantilla de prompt para el agente
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_agent_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Crear el agente
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Crear ejecutor con manejo de errores
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=settings.debug,
            handle_parsing_errors=True,
            max_iterations=5,  # Prevenir bucles infinitos
            return_intermediate_steps=True,  # Para depuración
        )
        
        return executor
    
    def _get_agent_system_prompt(self) -> str:
        """Obtener el prompt del sistema para el agente."""
        return f"""{SYSTEM_PROMPT}

## Herramientas Disponibles
Tienes acceso a las siguientes herramientas para ayudar al usuario:

1. **search_internal_documents**: Busca en documentos internos de Credicorp Capital.
   - Úsala para: políticas, procedimientos, reportes internos, información de productos Credicorp.
   
2. **search_financial_web**: Busca información financiera actualizada en la web.
   - Úsala para: noticias de mercado, cotizaciones, información de empresas públicas, tendencias.
   
3. **financial_calculator**: Realiza cálculos matemáticos y financieros.
   - Úsala para: calcular rendimientos, porcentajes, conversiones numéricas.

## Estrategia de Uso de Herramientas
1. **Primero intenta con documentos internos** si la pregunta es sobre Credicorp o sus políticas.
2. **Usa búsqueda web** si necesitas información externa o actualizada del mercado.
3. **Combina fuentes** cuando sea apropiado para dar una respuesta completa.
4. **Si ninguna herramienta aplica**, responde con tu conocimiento general pero aclara las limitaciones.

## Formato de Respuesta
- Siempre indica las fuentes de información utilizadas
- Si usaste documentos internos, menciona cuáles
- Si usaste información web, indica que es de fuentes externas
- Sé transparente sobre la confiabilidad de cada fuente
"""
    
    def _get_memory(self, session_id: str) -> ConversationSummaryBufferMemory:
        """Obtener o crear memoria para una sesión."""
        if session_id not in self._memories:
            # Cargar historial desde Cosmos DB
            memory = ConversationSummaryBufferMemory(
                llm=self.llm,  # LLM para generar resúmenes de mensajes antiguos
                max_token_limit=2000,  # Cuando se excede, resume mensajes antiguos
                memory_key="chat_history",
                return_messages=True,
            )
            
            # Restaurar desde Cosmos DB
            history = self.cosmos_db.get_message_history(session_id, limit=settings.memory_window_k * 2)
            for msg in history:
                if msg.role == MessageRole.USER:
                    memory.chat_memory.add_user_message(msg.content)
                elif msg.role == MessageRole.ASSISTANT:
                    memory.chat_memory.add_ai_message(msg.content)
            
            self._memories[session_id] = memory
        
        return self._memories[session_id]
    
    async def chat(
        self, 
        message: str, 
        session_id: str
    ) -> Tuple[str, List[str]]:
        """
        Procesar un mensaje de chat usando el agente.
        
        El agente decidirá autónomamente:
        1. Si usar herramientas
        2. Qué herramientas usar
        3. Cómo combinar información de múltiples fuentes
        
        Args:
            message: Mensaje del usuario
            session_id: Identificador de sesión
            
        Returns:
            Tupla de (texto_respuesta, documentos_fuente)
        """
        # Guardar mensaje del usuario en Cosmos DB
        self.cosmos_db.add_message(session_id, MessageRole.USER, message)
        
        # Obtener historial de conversación
        memory = self._get_memory(session_id)
        chat_history = memory.chat_memory.messages  # La memoria ya gestiona el resumen automáticamente
        
        try:
            # Ejecutar agente
            result = self.agent_executor.invoke({
                "input": message,
                "chat_history": chat_history
            })
            
            response = result.get("output", "Lo siento, no pude procesar tu pregunta.")
            sources = self._extract_sources_from_steps(result.get("intermediate_steps", []))
            
        except Exception as e:
            print(f"Agent execution error: {e}")
            # Fallback a respuesta directa del LLM
            response = await self._fallback_response(message, session_id)
            sources = []
        
        # Guardar respuesta del asistente en Cosmos DB
        self.cosmos_db.add_message(
            session_id, 
            MessageRole.ASSISTANT, 
            response,
            sources=sources if sources else None
        )
        
        # Actualizar memoria
        memory.chat_memory.add_user_message(message)
        memory.chat_memory.add_ai_message(response)
        
        return response, sources
    
    def _extract_sources_from_steps(self, intermediate_steps: List) -> List[str]:
        """Extraer fuentes de los pasos intermedios del agente."""
        sources = set()
        
        for step in intermediate_steps:
            if len(step) >= 2:
                action, observation = step[0], step[1]
                # Extraer nombre de herramienta como indicador de fuente
                tool_name = getattr(action, 'tool', 'unknown')
                
                if tool_name == "search_internal_documents":
                    sources.add("Documentos Internos Credicorp")
                elif tool_name == "search_financial_web":
                    sources.add("Búsqueda Web Financiera")
                elif tool_name == "financial_calculator":
                    sources.add("Cálculo Financiero")
        
        return list(sources)
    
    async def _fallback_response(self, message: str, session_id: str) -> str:
        """Fallback cuando el agente falla - usar LLM directo."""
        memory = self._get_memory(session_id)
        history = memory.chat_memory.messages  # La memoria ya gestiona el resumen
        
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(history)
        messages.append(HumanMessage(content=message))
        
        try:
            response = self.llm.invoke(messages)
            return response.content + "\n\n_Nota: Esta respuesta fue generada sin acceso a herramientas de búsqueda._"
        except Exception as e:
            print(f"Fallback error: {e}")
            return "Lo siento, hubo un error procesando tu mensaje. Por favor intenta de nuevo."
    
    def clear_session(self, session_id: str):
        """Limpiar memoria de sesión."""
        if session_id in self._memories:
            del self._memories[session_id]
        self.cosmos_db.clear_conversation(session_id)
    
    def get_status(self) -> dict:
        """Obtener estado del agente."""
        return {
            "agent_type": "agentic_rag",
            "llm_type": "azure_openai" if settings.use_azure_openai else "openai",
            "model": settings.azure_openai_deployment_name if settings.use_azure_openai else settings.openai_model,
            "tools_available": [t.name for t in self.tools],
            "active_sessions": len(self._memories)
        }
