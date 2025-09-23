# services/agent.py
import tempfile
import os
import uuid
import tempfile
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from pathlib import Path
# Importaciones del proyecto (DB, Models)
# **Importamos todo desde tu archivo database.py, NO desde peewee**
from database import Conversation, ChatMessage, db 

# Importaciones de configuración
from decouple import config

# LangChain / LangGraph
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langchain_core.messages import HumanMessage, AIMessage
# LlamaIndex
from llama_index.core import (
    VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader, Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as LI_OpenAI
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from typing import Optional
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter  # si tu versión lo soporta
import os
from decouple import config

# --- LangSmith / LangChain tracing: cargar desde .env y FORZAR en os.environ ---
os.environ["LANGCHAIN_TRACING_V2"] = config("LANGCHAIN_TRACING_V2", default="true")
os.environ["LANGCHAIN_ENDPOINT"]   = config("LANGCHAIN_ENDPOINT", default="https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"]    = config("LANGCHAIN_API_KEY", default="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")        # tu clave lsv2_...
os.environ["LANGCHAIN_PROJECT"]    = config("LANGCHAIN_PROJECT", default="RAG-IES-Agent-MVP")
# ─────────────────────────── Configuración de LangGraph/RAG ───────────────────────────
# Configuraciones asumidas desde tu .env
PG_DSN       = config("DATABASE_URL")
OPENAI_API_KEY = config("OPENAI_API_KEY")
ES_PASSWORD  = config("ES_PASSWORD")

ES_URL       = config("ES_URL", default="http://35.192.2.67:9200")
ES_USER      = config("ES_USER", default="elastic")
ES_INDEX     = config("ES_INDEX", default="producto_v2")
OPENAI_MODEL = config("OPENAI_MODEL", default="gpt-4o-mini")

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    print("⚠️ ADVERTENCIA: OPENAI_API_KEY no configurada.") 

from dataclasses import dataclass
from decouple import config

# índices
ES_INDEX_PRODUCTO   = config("ES_INDEX_PRODUCTO", default="producto_v2")
ES_INDEX_NORMATIVA  = config("ES_INDEX_NORMATIVA", default="ies_normativa_multi_tenant_original")

# campos por índice (AJUSTA a tu mapping real)
PROD_TEXT_FIELD = config("PROD_TEXT_FIELD", default="content")          # p.ej. "content" o "text"
PROD_VEC_FIELD  = config("PROD_VEC_FIELD",  default="embedding")        # p.ej. "embedding" o "content_vector"

NORM_TEXT_FIELD = config("NORM_TEXT_FIELD", default="content")
NORM_VEC_FIELD  = config("NORM_VEC_FIELD",  default="embedding")

@dataclass
class IndexCfg:
    name: str
    text_field: str
    vector_field: str

CFG_PRODUCTO  = IndexCfg(ES_INDEX_PRODUCTO,  PROD_TEXT_FIELD,  PROD_VEC_FIELD)
CFG_NORMATIVA = IndexCfg(ES_INDEX_NORMATIVA, NORM_TEXT_FIELD,  NORM_VEC_FIELD)




# ───────────────────────── LlamaIndex Utils (Globales) ─────────────────────────

from llama_index.embeddings.openai import OpenAIEmbedding  # 👈 Import correcto

def ensure_embed_model():
    # Usamos OpenAI Embeddings en lugar de HuggingFace
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",  # 👈 El modelo que pediste
        api_key=OPENAI_API_KEY
    )

from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings

def get_vector_store_by_cfg(cfg: IndexCfg) -> ElasticsearchStore:
    return ElasticsearchStore(
        es_url=ES_URL,
        es_user=ES_USER,
        es_password=ES_PASSWORD,
        index_name=cfg.name,
        text_field=cfg.text_field,
        vector_field=cfg.vector_field,
        metadata_field="metadata"  
    )
from typing import Optional
from llama_index.core.vector_stores import MetadataFilters
def get_retriever_by_cfg(cfg: IndexCfg, top_k: int = 4, filters: Optional[MetadataFilters] = None): # 👈 1. Acepta un filtro opcional
    ensure_embed_model()
    vstore = get_vector_store_by_cfg(cfg)
    storage_context = StorageContext.from_defaults(vector_store=vstore)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vstore,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
    )
    # 2. Pasa el filtro al retriever al momento de crearlo 👇
    return index.as_retriever(similarity_top_k=top_k, filters=filters)


# ───────────────────────── Lógica de Ingesta/Indexación (Para file_serv.py) ─────────────────────────

# services/agent.py (Función ingest_file_to_es)

def ingest_file_to_es(file_content: bytes, file_name: str) -> bool:
    """
    Ingesta un archivo único en Elasticsearch usando LlamaIndex.
    Utiliza os.tempdir para mayor robustez de permisos que tempfile.TemporaryDirectory().
    """
    import os 
    from pathlib import Path
    import tempfile
    import uuid
    
    # 1. Crear la ruta temporal segura en el directorio TEMP del sistema
    unique_file_name = f"{uuid.uuid4()}_{file_name}"
    # os.path.join(os.tempdir, ...) es la forma más segura de obtener la ruta temporal
    temp_dir_path = tempfile.gettempdir()
    temp_path = Path(os.path.join(temp_dir_path, unique_file_name)) 
    
    try:
        # 2. Escribir el contenido del archivo subido
        with open(temp_path, "wb") as f:
            f.write(file_content)

        # 3. Lógica de LlamaIndex (Lee del path temporal)
        Settings.llm = LI_OpenAI(model="gpt-4o-mini", temperature=0.2) 
        
        # SimpleDirectoryReader requiere una lista de paths de archivos
        docs = SimpleDirectoryReader(input_files=[temp_path]).load_data()

        # Procesamiento y limpieza de documentos (como se definió originalmente)
        clean_docs = []
        for d in docs:
            txt = " ".join(d.text.split())
            meta = dict(d.metadata or {})
            meta["file_name"] = file_name            # 👈 fuerza el nombre original
            meta["source"] = file_name               # opcional, por compatibilidad
            clean_docs.append(
                Document(
                    text=txt,
                    metadata=meta,
                    id_=getattr(d, "id_", None),
                )
            )

        Settings.node_parser = SentenceSplitter(chunk_size=1500, chunk_overlap=50)
        ensure_embed_model()

        vector_store = get_vector_store_by_cfg(CFG_PRODUCTO)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(clean_docs, storage_context=storage_context)

        print(f"✅ Archivo '{file_name}' indexado con éxito.")
        return True
        
    except Exception as e:
        # 4. Imprimir el error detallado para el debugging.
        print(f"❌ ERROR CRÍTICO AL INDEXAR '{file_name}': {e}") 
        return False
        
    finally:
        # 5. Limpiar el archivo temporal (CRÍTICO para evitar conflictos de permisos)
        if temp_path.exists():
            try:
                os.remove(temp_path)
                print(f"Limpieza: Archivo temporal {temp_path.name} eliminado.")
            except Exception as cleanup_e:
                print(f"Advertencia: No se pudo limpiar el archivo temporal {temp_path.name}: {cleanup_e}")
                
import logging # Asegúrate de que logging está importado
logging.basicConfig(level=logging.INFO) # Configura el logging básico

def _format_nodes(results) -> str:
    """Formatea los nodos de LlamaIndex a un string legible, con logging mejorado y tolerancia a fallos."""
    if not results:
        return "La búsqueda no arrojó ningún resultado."

    blocks = []
    for i, n in enumerate(results, 1):
        node = getattr(n, "node", None)
        
        # --- Verificación de robustez ---
        if not node:
            logging.warning(f"El resultado de búsqueda {i} no tiene un 'node' válido (posiblemente por un error de parseo). Objeto recibido: {n}")
            continue

        text = node.get_content().strip()
        if not text:
            logging.warning(f"El resultado de búsqueda {i} tiene un nodo pero no contenido de texto. Saltando.")
            continue
            
        # --- Formateo de la fuente (metadata) ---
        meta = getattr(node, "metadata", {}) or {}
        src = meta.get("file_name") or meta.get("source") or meta.get("filename") or "Fuente Desconocida"
        page = meta.get("page_label") or meta.get("page") or meta.get("page_number")
        
        src_str = f"{src}{f', pág. {page}' if page else ''}"
        
        blocks.append(f"[{i}] {text}\n   Fuente: {src_str}")

    if not blocks:
        return "La búsqueda encontró documentos, pero no se pudo extraer contenido de texto válido de ellos."
        
    return "\n\n".join(blocks)

# ───────────────────────── Herramienta (Tool) para LangGraph ─────────────────────────
from langchain.tools import tool
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from langchain.tools import tool

@tool
def consulta_normativa(query: str) -> str:
    """Busca normativa institucional (index: ies_normativa_multi_tenant_original)."""
    retriever = get_retriever_by_cfg(CFG_NORMATIVA, top_k=4)
    try:
        return _format_nodes(retriever.retrieve(query))
    except Exception as e:
        print(f"consulta_normativa error: {e}")
        return "Error consultando el índice de normativa."

from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

@tool
def consulta_producto(query: str, file_name: str | None = None) -> str:
    """
    Busca en documentos subidos por el cliente (index: producto_v2).
    Si se proporciona un file_name, la búsqueda se limita a ese documento.
    """
    filters = None
    if file_name:
        print(f"🔍 Búsqueda filtrada por archivo: {file_name}")
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="file_name", value=file_name)]  # ✅ clave correcta
        )

    retriever = get_retriever_by_cfg(CFG_PRODUCTO, top_k=4, filters=filters)
    try:
        return _format_nodes(retriever.retrieve(query))
    except Exception as e:
        print(f"consulta_producto error: {e}")
        return "Error consultando el índice de producto."

import re
import unicodedata

DOC_HINTS = [
    "este documento", "este archivo", "este pdf", "adjunto",
    "según el documento", "segun el documento", "de que trata",
    "resumen del documento", "resumen de este", "analiza el pdf",
    "doc ", "archivo ", "pdf", ".pdf", ".doc", ".docx", ".txt"
]
NORM_HINTS = [
    "normativa", "loes", "reglamento", "rloes", "caces",
    "senescyt", "acreditación", "acreditacion", "resolución",
    "resolucion", "decreto", "artículo", "articulo", "ies"
]
CLEAR_HINTS = [
    "ignora el archivo", "ignorar archivo", "no uses el archivo",
    "sin archivo", "limpia contexto", "borrar contexto",
    "usar normativa", "usa normativa", "consulta normativa"
]

def _normalize(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()

def _count_hits(text: str, vocab: list[str]) -> int:
    return sum(1 for k in vocab if k in text)

def _mentions_file_name(text: str, file_ctx: str | None) -> bool:
    if not file_ctx:
        return False
    t = _normalize(text)
    f = _normalize(file_ctx)
    return f in t  # coincide nombre exacto

def _wants_file_context(user_input: str, file_ctx: str | None) -> bool:
    """Regresa True si el mensaje parece referirse al documento subido."""
    if not file_ctx:
        return False
    t = _normalize(user_input)
    # Comandos para NO usar archivo
    if any(h in t for h in CLEAR_HINTS):
        return False
    # Si pide explícitamente normativa (2+ hits), no uses archivo
    if _count_hits(t, NORM_HINTS) >= 2 and _count_hits(t, DOC_HINTS) == 0:
        return False
    # Si menciona el nombre del archivo o hints de documento, úsalo
    if _mentions_file_name(user_input, file_ctx):
        return True
    if _count_hits(t, DOC_HINTS) >= 1:
        return True
    return False

def _is_normative_intent(user_input: str) -> bool:
    t = _normalize(user_input)
    return _count_hits(t, NORM_HINTS) >= 1

# ───────────────────────── Servicio Principal del Agente ─────────────────────────
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

class RAGAgentService:
    """
    Servicio que encapsula la lógica del Agente RAG (LangGraph), memoria, 
    y utilidades como la generación de títulos.
    """
    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.agent = self._initialize_agent()
        
    def _initialize_agent(self):
        # 1) Checkpointer seguro
        checkpointer = None
        if PG_DSN:  # DATABASE_URL
            try:
                connection_kwargs = {"autocommit": True, "prepare_threshold": 0}
                pg_pool = ConnectionPool(conninfo=PG_DSN, max_size=10, kwargs=connection_kwargs)
                checkpointer = PostgresSaver(pg_pool)
                checkpointer.setup()
                print("✅ LangGraph PostgresSaver activo")
            except Exception as e:
                print(f"⚠️ LangGraph PostgresSaver deshabilitado: {e}. Usando MemorySaver.")
                checkpointer = MemorySaver()
        else:
            print("ℹ️ DATABASE_URL no definido. Usando MemorySaver para LangGraph.")
            checkpointer = MemorySaver()

        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)

        system_prompt = """
        ## Tu Identidad y Misión
        Eres LQA-1 (Legal Quantitative Assistant), un asistente legal de alta precisión especializado en la normativa de Educación Superior de Ecuador.
        Tu misión es doble:
        1.  Responder preguntas de conocimiento basándote **única y exclusivamente** en la evidencia recuperada de tus herramientas.
        2.  Mantener una conversación fluida y con contexto, recordando interacciones pasadas dentro de este chat.
        Responde siempre en español.

        ---

        ## Memoria y Contexto Conversacional (NUEVA SECCIÓN)
        El historial completo de nuestra conversación está disponible para ti. Si te pregunto sobre algo que dijimos antes, un documento que subí, o te pido un resumen de la charla, debes revisar los mensajes anteriores para responder. **Para estas preguntas sobre el chat, no necesitas usar una herramienta.**

        ---

        ## Herramientas a tu Disposición
        1.  **`consulta_normativa(query: str)`**: Para cualquier consulta sobre el marco legal y normativo de Ecuador.
        2.  **`consulta_producto(query: str, file_name: str | None)`**: **Solo** para preguntas sobre un documento que el usuario haya subido.

        ---

        ## Proceso de Razonamiento y Selección de Herramientas
        Sigue esta jerarquía estricta para preguntas que requieran buscar conocimiento:

        1.  **Análisis de Intención Explícita:** (La lógica que ya tenías sigue aquí...)
        2.  **Manejo de la Ambigüedad:** (La lógica que ya tenías sigue aquí...)
        3.  **Comportamiento por Defecto:** (La lógica que ya tenías sigue aquí...)

        ---

        ## Reglas Críticas para la Respuesta Final
        - **Cero Alucinaciones:** Basa el 100% de tus respuestas de conocimiento en el texto recuperado.
        - **Declara la Insuficiencia:** Si una herramienta no devuelve información, infórmalo claramente.
        - **Cita Obligatoria:** Siempre cita tus fuentes al final de las respuestas basadas en herramientas.

        ##- Si el usuario pregunta sobre el historial de esta conversación (p. ej., “¿qué te envié primero?”, “¿de qué hablamos?”), 
  usa el contexto conversacional disponible (memoria LangGraph) y responde sin llamar herramientas.

        """

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{messages}")]
        )

        # 👇 usa las tools nuevas, no `consulta_corpus`
        toolkit = [consulta_normativa, consulta_producto]

        agent_executor = create_react_agent(
            llm,
            toolkit,
            checkpointer=checkpointer,
            prompt=prompt,
        )
        return agent_executor
        
    def generate_title_from_text(self, text: str) -> str:
        """Usa el LLM para generar un título de conversación conciso."""
        llm_title = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Eres un experto en titular. Genera un título conciso de 5 palabras o menos basado en el siguiente mensaje."
                ),
                ("user", "{message}"),
            ]
        )
        chain = prompt | llm_title
        
        try:
            result = chain.invoke({"message": text})
            title = result.content.strip().replace('"', '')
            return (title[:80] if len(title) > 80 else title)
        except Exception as e:
            print(f"Error al generar título: {e}. Usando título por defecto.")
            return "Conversación sin título"

    def _log_message(self, conversation_id: str, role: str, content: str, attachments: list = None):
        """Guarda el mensaje en la base de datos Peewee."""
        message_id = str(uuid.uuid4())
        try:
            with db.atomic():
                # Crear el mensaje
                message = ChatMessage.create(
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                )
                
                # Crear archivos adjuntos si existen
                if attachments:
                    from database import AttachFile
                    for attachment in attachments:
                        AttachFile.create(
                            message=message,
                            name=attachment.get('name', ''),
                            url=attachment.get('url', ''),
                            type=attachment.get('type', '')
                        )
                
                Conversation.update(updated_at=datetime.now()).where(
                    Conversation.id == conversation_id
                ).execute()
        except Exception as e:
            print(f"Error al guardar mensaje en Peewee: {e}")
            message_id = "error_logging"

        return message_id
    def _memorize_turn(self, thread_id: str, user_text: str, ai_text: str) -> None:
            
            """
            Persiste (usuario, asistente) en la memoria de LangGraph para este thread_id.
            Usa update_state si está disponible (no ejecuta el grafo), y si no, hace un
            invoke de baja fricción para que el checkpointer escriba el estado.
            """
            try:
                if getattr(self, "agent", None) and hasattr(self.agent, "update_state"):
                    # ✅ escribe directamente al estado sin correr el grafo
                    self.agent.update_state(
                        config={"configurable": {"thread_id": thread_id}},
                        values={"messages": [HumanMessage(content=user_text), AIMessage(content=ai_text)]},
                    )
                elif getattr(self, "agent", None):
                    # Fallback: invoca el grafo (puede generar un paso extra, pero persiste memoria)
                    self.agent.invoke(
                        {"messages": [HumanMessage(content=user_text), AIMessage(content=ai_text)]},
                        config={"configurable": {"thread_id": thread_id}},
                    )
                else:
                    print("⚠️ _memorize_turn: no hay self.agent para persistir memoria.")
            except Exception as e:
                print(f"⚠️ _memorize_turn falló: {e}")

    # services/agent.py -> dentro de la clase RAGAgentService
    def run_agent(self, conversation_id: str, thread_id: str, user_input: str) -> Dict[str, Any]:
        from peewee import DoesNotExist
        from datetime import datetime
        from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        user_message_id = self._log_message(conversation_id, "user", user_input)
        final_text = ""

        # 1) Resolver conversación por id o por thread_id
        conv = Conversation.get_or_none(Conversation.id == conversation_id)
        if conv is None:
            conv = Conversation.get_or_none(Conversation.thread_id == conversation_id)
            if conv:
                conversation_id = conv.id
        if conv is None:
            final_text = f"No encontré la conversación '{conversation_id}'."
            agent_message_id = self._log_message(conversation_id, "assistant", final_text)
            return {
                "idUser": user_message_id,
                "agent": {
                    "id": agent_message_id,
                    "content": final_text,
                    "sender": "assistant",
                    "timestamp": int(datetime.now().timestamp() * 1000),
                },
            }

        file_ctx = conv.last_file_context or None

        # 2) Router de intención
        t = _normalize(user_input)
        clear_ctx = any(h in t for h in CLEAR_HINTS)
        normative_intent = _is_normative_intent(user_input)
        use_file = (not clear_ctx) and _wants_file_context(user_input, file_ctx)

        if clear_ctx:
            print("🧹 Solicitud de ignorar/limpiar contexto: este turno NO usará archivo.")

        # 3) Ruta A: usar archivo solo si la intención lo amerita
        if use_file:
            print(f"⚡️ Contexto de archivo detectado: '{file_ctx}'. Forzando búsqueda en producto_v2.")
            try:
                # Filtro por file_name exacto
                try:
                    filters = MetadataFilters(filters=[ExactMatchFilter(key="file_name", value=file_ctx)])
                except Exception:
                    filters = None

                retriever = get_retriever_by_cfg(CFG_PRODUCTO, top_k=self.top_k, filters=filters)
                results = retriever.retrieve(user_input)

                # Fallback: si no sale nada (docs viejos), reintenta sin filtro y filtra en memoria por sufijo
                if not results:
                    retriever2 = get_retriever_by_cfg(CFG_PRODUCTO, top_k=max(self.top_k, 8))
                    raw = retriever2.retrieve(user_input or "resumen del documento")
                    results = [r for r in raw if (getattr(r.node, "metadata", {}) or {}).get("file_name","").lower().endswith((file_ctx or "").lower())]

                if not results:
                    final_text = "No pude encontrar información relevante en el documento que subiste."
                else:
                    blocks = []
                    for i, n in enumerate(results, 1):
                        meta = getattr(n.node, "metadata", {}) or {}
                        src = meta.get("file_name") or meta.get("source") or "N/A"
                        page = meta.get("page_label") or meta.get("page") or meta.get("page_number") or ""
                        src_str = f"{src}{f', p. {page}' if page else ''}"
                        snippet = n.node.get_content().strip()
                        blocks.append(f"[{i}] {snippet}\nFuente: {src_str}")
                    context_text = "\n\n".join(blocks)

                    final_prompt = ChatPromptTemplate.from_messages([
                        ("system",
                        "Eres un asistente que responde únicamente con base en el CONTEXTO dado. "
                        "Si el contexto no tiene la respuesta, dilo claramente. "
                        "Resume y contesta en español, citando [n] cuando corresponda."),
                        ("human", "CONTEXTO:\n---\n{contexto}\n---\nPREGUNTA: {pregunta}")
                    ])
                    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
                    response = (final_prompt | llm).invoke({"contexto": context_text, "pregunta": user_input})
                    final_text = response.content or "⚠️ Sin respuesta textual del LLM."
            except Exception as e:
                print(f"Error durante la ejecución con archivo '{file_ctx}': {e}")
                final_text = "Lo siento, ocurrió un error al procesar el documento."
                self._memorize_turn(thread_id, user_input, final_text)   # 👈 añade esto

        # 4) Ruta B: sin archivo
        else:
            if normative_intent:
                print("📚 Intención normativa detectada. Consultando índice de normativa.")
                try:
                    retriever = get_retriever_by_cfg(CFG_NORMATIVA, top_k=self.top_k)
                    results = retriever.retrieve(user_input)
                    if not results:
                        final_text = "No encontré evidencia suficiente en la base normativa para responder. ¿Quieres que intente con los documentos subidos?"
                    else:
                        blocks = []
                        for i, n in enumerate(results, 1):
                            meta = getattr(n.node, "metadata", {}) or {}
                            src = meta.get("file_name") or meta.get("source") or "N/A"
                            page = meta.get("page_label") or meta.get("page") or meta.get("page_number") or ""
                            src_str = f"{src}{f', p. {page}' if page else ''}"
                            snippet = n.node.get_content().strip()
                            blocks.append(f"[{i}] {snippet}\nFuente: {src_str}")
                        context_text = "\n\n".join(blocks)

                        final_prompt = ChatPromptTemplate.from_messages([
                            ("system",
                            "Eres un asistente que responde con base en CONTEXTO normativo. "
                            "Cita [n] y evita inventar artículos. Si no hay evidencia, dilo."),
                            ("human", "CONTEXTO:\n---\n{contexto}\n---\nPREGUNTA: {pregunta}")
                        ])
                        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)
                        response = (final_prompt | llm).invoke({"contexto": context_text, "pregunta": user_input})
                        final_text = response.content or "⚠️ Sin respuesta textual del LLM."
                except Exception as e:
                    print(f"Error consultando normativa: {e}")
                    final_text = "Ocurrió un error consultando la base normativa."
                    self._memorize_turn(thread_id, user_input, final_text) 

            else:
                print("▶️ Consulta general sin archivo. Dejo que el agente ReAct decida la herramienta.")
                try:
                    result = self.agent.invoke(
                        {"messages": [HumanMessage(content=user_input)]},
                        config={
                            "configurable": {"thread_id": thread_id},
                            "tags": ["rag", "langgraph"],
                            "metadata": {"conversation_id": conversation_id, "top_k": self.top_k, "env": "dev"},
                        },
                    )
                    final_message = result["messages"][-1]
                    final_text = getattr(final_message, "content", None) or "⚠️ Sin respuesta textual del agente."
                    self._memorize_turn(thread_id, user_input, final_text)
                except Exception as e:
                    print(f"Error al ejecutar el agente LangGraph: {e}")
                    final_text = "Lo siento, ocurrió un error interno al procesar tu solicitud."
        # 5) Log y retorno
        agent_message_id = self._log_message(conversation_id, "assistant", final_text)
        return {
            "idUser": user_message_id,
            "agent": {
                "id": agent_message_id,
                "content": final_text,
                "sender": "assistant",
                "timestamp": int(datetime.now().timestamp() * 1000),
            },
        }

# Inicialización Singleton
rag_agent_service = RAGAgentService()