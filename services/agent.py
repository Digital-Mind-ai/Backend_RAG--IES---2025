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

# LlamaIndex
from llama_index.core import (
    VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader, Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as LI_OpenAI
from llama_index.vector_stores.elasticsearch import ElasticsearchStore


# ─────────────────────────── Configuración de LangGraph/RAG ───────────────────────────
# Configuraciones asumidas desde tu .env
PG_DSN       = config("DATABASE_URL")
OPENAI_API_KEY = config("OPENAI_API_KEY")
ES_PASSWORD  = config("ES_PASSWORD")

ES_URL       = config("ES_URL", default="http://35.192.2.67:9200")
ES_USER      = config("ES_USER", default="elastic")
ES_INDEX     = config("ES_INDEX", default="producto")
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


# ───────────────────────── LlamaIndex Utils (Globales) ─────────────────────────

def ensure_embed_model():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        device=DEVICE,
    )

def get_vector_store() -> ElasticsearchStore:
    return ElasticsearchStore(
        es_url=ES_URL,
        es_user=ES_USER,
        es_password=ES_PASSWORD,
        index_name=ES_INDEX,
    )

def get_retriever(top_k: int = 4):
    ensure_embed_model()
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
    )
    return index.as_retriever(similarity_top_k=top_k)


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
            clean_docs.append(
                Document(
                    text=txt,
                    metadata=d.metadata,
                    id_=getattr(d, "id_", None),
                )
            )

        Settings.node_parser = SentenceSplitter(chunk_size=1500, chunk_overlap=50)
        ensure_embed_model()

        vector_store = get_vector_store()
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


# ───────────────────────── Herramienta (Tool) para LangGraph ─────────────────────────
@tool
def consulta_corpus(query: str) -> str:
    """Busca respuestas en el corpus indexado (Elasticsearch vía LlamaIndex)."""
    retriever = get_retriever(top_k=4)
    results = retriever.retrieve(query)
    
    if not results:
        return "Sin resultados en el corpus."

    blocks = []
    for i, n in enumerate(results, 1):
        meta = n.node.metadata or {}
        src = meta.get("source") or meta.get("file_name") or "N/A"
        page = meta.get("page_label") or meta.get("page") or ""
        src_str = f"{src}{f', p. {page}' if page else ''}"
        blocks.append(f"[{i}] {n.node.get_content().strip()}\nFuente: {src_str}")
        
    return "\n\n".join(blocks)


# ───────────────────────── Servicio Principal del Agente ─────────────────────────

class RAGAgentService:
    """
    Servicio que encapsula la lógica del Agente RAG (LangGraph), memoria, 
    y utilidades como la generación de títulos.
    """
    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.agent = self._initialize_agent()
        
    def _initialize_agent(self):
        """Inicializa el checkpointer de LangGraph y el agente ReAct."""
        connection_kwargs = {"autocommit": True, "prepare_threshold": 0}
        pg_pool = ConnectionPool(conninfo=PG_DSN, max_size=20, kwargs=connection_kwargs)
        checkpointer = PostgresSaver(pg_pool)
        checkpointer.setup()
        
        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
        
        # NOTA: Usar una versión válida de tu prompt system aquí:
        system_prompt = """Eres un asistente que puede usar herramientas SOLO cuando el usuario pida información del corpus/documentos.
Reglas:
- Si el mensaje es saludo, cortesía, charla pequeña o no pide datos del corpus → RESPONDE tú mismo, NO uses herramientas.
- Usa la herramienta `consulta_corpus` solo si la intención es buscar contenido en el corpus (documentos subidos / índice).
- Si no hay evidencia suficiente en el corpus, dilo.
- Responde en español y cita la fuente (archivo y página) cuando uses el corpus.
"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{messages}"),
            ]
        )
        
        toolkit = [consulta_corpus]
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

    def _log_message(self, conversation_id: str, role: str, content: str):
        """Guarda el mensaje en la base de datos Peewee."""
        message_id = str(uuid.uuid4())
        try:
            with db.atomic():
                ChatMessage.create(
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                )
                Conversation.update(updated_at=datetime.now()).where(
                    Conversation.id == conversation_id
                ).execute()
        except Exception as e:
            print(f"Error al guardar mensaje en Peewee: {e}")
            message_id = "error_logging"

        return message_id

    def run_agent(self, conversation_id: str, thread_id: str, user_input: str) -> Dict[str, Any]:
        """Ejecuta el agente RAG para un mensaje de usuario específico."""
        
        user_message_id = self._log_message(conversation_id, "user", user_input)
        final_text = ""
        
        try:
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": thread_id}},
            )
            
            final_message = result["messages"][-1]
            final_text = final_message.content if final_message.content else "⚠️ Sin respuesta textual del agente."
            
        except Exception as e:
            print(f"Error al ejecutar el agente LangGraph: {e}") 
            final_text = "Lo siento, ha ocurrido un error interno al procesar tu solicitud RAG."
        
        agent_message_id = self._log_message(conversation_id, "assistant", final_text)
        
        return {
            "idUser": user_message_id, 
            "agent": {
                "id": agent_message_id,
                "content": final_text,
                "sender": "assistant",
                "timestamp": int(datetime.now().timestamp() * 1000) 
            }
        }

# Inicialización Singleton
rag_agent_service = RAGAgentService()