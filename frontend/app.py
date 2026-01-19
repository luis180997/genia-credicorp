"""
Genia - Interfaz de Chat Streamlit
Interfaz de usuario para el asistente de inversiones con RAG.
"""
import streamlit as st
import requests
import uuid
from datetime import datetime
from pathlib import Path
import os

# Configuración
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
# Ruta absoluta al logo para evitar problemas con el directorio de trabajo
SCRIPT_DIR = Path(__file__).parent.resolve()
LOGO_PATH = SCRIPT_DIR / "assets" / "logo.png"

# Configuración de página
st.set_page_config(
    page_title="Genia - Asistente de Inversiones",
    page_icon="assets/logo.png" if LOGO_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS mínimo - el tema principal viene de .streamlit/config.toml
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 0.9rem;
        opacity: 0.7;
        margin-top: 0.25rem;
    }
    .source-badge {
        background-color: #3d4f5f;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-right: 0.3rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Inicializar variables de estado de sesión."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "backend_available" not in st.session_state:
        st.session_state.backend_available = check_backend_health()


def check_backend_health():
    """Verificar si el backend está disponible."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def send_message(message: str):
    """Enviar mensaje al backend y obtener respuesta."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={
                "message": message,
                "session_id": st.session_state.session_id
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["response"], data.get("sources", [])
        else:
            return f"Error: {response.status_code} - {response.text}", []
    except requests.exceptions.Timeout:
        return "Error: La solicitud tardó demasiado. Por favor intenta de nuevo.", []
    except requests.exceptions.ConnectionError:
        return "Error: No se puede conectar al servidor. Verifica que el backend esté ejecutándose.", []
    except Exception as e:
        return f"Error: {str(e)}", []


def upload_document(file):
    """Subir documento al backend."""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(
            f"{BACKEND_URL}/documents/upload",
            files=files,
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            # Verificar si es duplicado
            if data.get("status") == "duplicate":
                return False, data['message']
            return True, data['message']
        else:
            return False, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def get_documents():
    """Obtener lista de documentos indexados."""
    try:
        response = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        if response.status_code == 200:
            return response.json().get("documents", [])
        return []
    except:
        return []


def clear_conversation():
    """Limpiar conversación actual."""
    try:
        response = requests.delete(
            f"{BACKEND_URL}/conversations/{st.session_state.session_id}",
            timeout=10
        )
        if response.status_code == 200:
            st.session_state.messages = []
            return True
    except:
        pass
    return False


def render_sidebar():
    """Renderizar barra lateral con controles e información."""
    with st.sidebar:
        # Logo
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=80)
        
        st.markdown("### Credicorp Capital")
        st.markdown("---")
        
        # Indicador de estado
        if st.session_state.backend_available:
            st.success("Sistema conectado")
        else:
            st.error("Sistema desconectado")
            if st.button("Reintentar conexión"):
                st.session_state.backend_available = check_backend_health()
                st.rerun()
        
        st.markdown("---")
        
        # Ingesta de documentos
        st.markdown("#### Ingesta de Documentos")
        uploaded_file = st.file_uploader(
            "Sube documentos a la base de conocimiento",
            type=["pdf", "txt", "docx"],
            help="Los documentos se suben al almacenamiento y luego pueden indexarse",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            if st.button("Subir Documento", use_container_width=True):
                with st.spinner("Subiendo documento..."):
                    success, message = upload_document(uploaded_file)
                    if success:
                        st.success("Documento subido correctamente")
                        st.info("Usa 'Sincronizar' para indexar los nuevos documentos")
                    else:
                        st.error(message)
        
        # Sincronizacion del indice
        st.markdown("---")
        st.markdown("#### Sincronizar Indice")
        st.caption("Indexa los documentos nuevos en Azure AI Search")
        
        if st.button("Sincronizar", use_container_width=True):
            with st.spinner("Ejecutando indexador..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/indexer/run", timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "success":
                            st.success("Indexador ejecutado correctamente")
                        else:
                            st.warning(data.get("message", "Indexador en proceso"))
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error de conexion: {e}")
        
        st.caption("Limite: 1 sincronizacion cada 3 minutos")
        
        # Base de conocimiento
        st.markdown("---")
        st.markdown("#### Base de Conocimiento")
        documents = get_documents()
        if documents:
            st.caption(f"{len(documents)} documento(s)")
            with st.expander("Ver documentos", expanded=False):
                for doc in documents:
                    st.markdown(f"• {doc}")
        else:
            st.caption("Sin documentos en la base")
        
        # Controles de sesión
        st.markdown("---")
        st.markdown("#### Configuración")
        
        st.caption(f"Sesión: {st.session_state.session_id[:8]}...")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Limpiar Chat"):
                if clear_conversation():
                    st.success("Chat limpiado")
                    st.rerun()
        
        with col2:
            if st.button("Nueva Sesión"):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.rerun()
        
        # Pie de página con autoria
        st.markdown("---")
        st.caption("Genia v1.0 | Credicorp Capital")
        st.caption("Desarrollado por Luis Agüero")


def render_chat():
    """Renderizar interfaz principal de chat."""
    # Encabezado
    st.markdown('<h1 class="main-header">Genia</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Asistente de Inteligencia Artificial para Inversiones</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Historial de chat
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                if message.get("sources"):
                    st.markdown("**Fuentes consultadas:**")
                    for source in message["sources"]:
                        st.markdown(f'<span class="source-badge">{source}</span>', unsafe_allow_html=True)
    
    # Input de chat
    if prompt := st.chat_input("Escribe tu pregunta sobre inversiones..."):
        # Agregar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Obtener respuesta
        with st.chat_message("assistant"):
            with st.spinner("Procesando consulta..."):
                response, sources = send_message(prompt)
                st.write(response)
                
                if sources:
                    st.markdown("**Fuentes consultadas:**")
                    for source in sources:
                        st.markdown(f'<span class="source-badge">{source}</span>', unsafe_allow_html=True)
        
        # Agregar mensaje del asistente
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })
    
    # Mensaje de bienvenida si no hay mensajes
    if not st.session_state.messages:
        st.info("""
        Bienvenido a **Genia**, tu asistente de inteligencia artificial especializado en inversiones.
        
        Puedo ayudarte con:
        - Consultas sobre políticas de inversión
        - Búsqueda en documentos internos
        - Información sobre estrategias y productos financieros
        
        ¿En qué puedo ayudarte hoy?
        """)


def main():
    """Punto de entrada principal de la aplicación."""
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
