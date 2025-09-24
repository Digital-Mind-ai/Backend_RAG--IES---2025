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
from peewee import DoesNotExist
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
from langchain_core.messages import AIMessage, HumanMessage # Asegúrate de que AIMessage esté importado
from typing import List, Optional, Tuple
import re
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator # 👈 Importa los componentes correctos
# Reranker opcional (alta precisión normativa) 
try:
    from FlagEmbedding import BGEM3FlagReranker
    _RERANK_OK = True
except Exception:
    _RERANK_OK = False

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


# services/agent.py -> AÑADE ESTO EN LA SECCIÓN DE IMPORTS

from elasticsearch import Elasticsearch
from llama_index.core.schema import TextNode # Necesaria para reconstruir los nodos

# ... (después de tus configuraciones de ES_URL, ES_USER, etc.)

# AÑADE LA CREACIÓN DEL CLIENTE DE ELASTICSEARCH
try:
    es_client = Elasticsearch(
        hosts=[ES_URL],
        basic_auth=(ES_USER, ES_PASSWORD)
    )
    print("✅ Cliente de Elasticsearch conectado exitosamente.")
except Exception as e:
    es_client = None
    print(f"❌ No se pudo conectar el cliente de Elasticsearch: {e}")


# services/agent.py -> AÑADE ESTA FUNCIÓN AUXILIAR

def _get_nodes_by_filename_raw(index_name: str, file_name: str) -> List[TextNode]:
    if not es_client:
        print("❌ Cliente de Elasticsearch no disponible.")
        return []

    # ✅ Soporta "metadata.file_name" y "file_name" al nivel superior
    query = {
        "bool": {
            "should": [
                {"term": {"metadata.file_name.keyword": file_name}},
                {"term": {"file_name.keyword": file_name}}
            ],
            "minimum_should_match": 1
        }
    }

    try:
        response = es_client.search(index=index_name, query=query, size=200)
        nodes = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"] or {}
            # ✅ fusiona metadatos de ambos esquemas
            meta_from_nested = dict(src.get("metadata") or {})
            meta_from_top = {k: v for k, v in src.items() if k not in ("content", "embedding", "vector", "metadata")}
            merged_meta = {**meta_from_top, **meta_from_nested}
            if "file_name" not in merged_meta and "metadata" in src:
                # último intento de fijar nombre de archivo
                merged_meta["file_name"] = meta_from_nested.get("file_name") or meta_from_top.get("file_name")

            node = TextNode(
                text=src.get("content", "") or "",
                metadata=merged_meta,
                id_=hit["_id"]
            )
            nodes.append(node)

        print(f"📄 Búsqueda directa en ES encontró {len(nodes)} nodos para el archivo '{file_name}'.")
        return nodes
    except Exception as e:
        print(f"❌ Error en la búsqueda directa en Elasticsearch: {e}")
        return []

def _match_uploaded_file(user_input: str) -> str | None:
    all_files = [conv.last_file_context for conv in Conversation.select()]
    for f in filter(None, all_files):
        if f.lower() in user_input.lower():
            return f
    return None
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
    """
    Crea la conexión con Elasticsearch.
    Se especifica `metadata_field="metadata"` porque los metadatos se guardan bajo ese campo.
    """
    return ElasticsearchStore(
        es_url=ES_URL,
        es_user=ES_USER,
        es_password=ES_PASSWORD,
        index_name=cfg.name,
        text_field=cfg.text_field,
        vector_field=cfg.vector_field,
        metadata_field="metadata"
        # --- CAMBIO CLAVE ---
       

    )



def get_retriever_by_cfg(cfg: IndexCfg, top_k: int = 4, filters: Optional[MetadataFilters] = None):
    """
    Obtiene un retriever de LlamaIndex, pasando los filtros directamente.
    Esta función no cambia, pero ahora recibirá filtros mucho más potentes.
    """
    ensure_embed_model()
    vstore = get_vector_store_by_cfg(cfg)
    storage_context = StorageContext.from_defaults(vector_store=vstore)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vstore,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
    )
    # Pasa el filtro directamente al retriever. LlamaIndex lo traducirá a la consulta de Elasticsearch.
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
    clean_file_name = (file_name or "").strip()
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

# services/agent.py -> REEMPLAZA tu función _format_nodes

def _format_nodes(results) -> str:
    """Formatea los nodos de LlamaIndex a un string legible, con logging mejorado y tolerancia a fallos."""
    if not results:
        return "La búsqueda no arrojó ningún resultado."

    blocks = []
    for i, n in enumerate(results, 1):
        
        # ▼▼▼ ESTA ES LA ÚNICA LÍNEA QUE CAMBIA ▼▼▼
        # Intenta obtener el nodo, ya sea de un diccionario o de un objeto.
        node = n.get("node") if isinstance(n, dict) else getattr(n, "node", None)
        
        # --- Verificación de robustez (El resto de la función es igual) ---
        if not node:
            logging.warning(f"El resultado de búsqueda {i} no tiene un 'node' válido. Objeto recibido: {n}")
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
import re

# Encabezado de artículos: "Artículo 18", "Art. 18"
_ARTICLE_HDR = re.compile(r'(?im)^\s*(?:art[íi]culo|art\.)\s*(\d+[A-Za-z]?)\s*[\.:–—-]?\s*')

def _cut_article_block(text: str, num: str) -> str | None:
    """Corta desde 'Artículo num' hasta el siguiente encabezado de artículo (o fin)."""
    if not text:
        return None
    start_pat = re.compile(rf'(?im)^\s*(?:art[íi]culo|art\.)\s*{re.escape(num)}\s*[\.:–—-]?\s*')
    m = start_pat.search(text)
    if not m:
        return None
    start = m.start()
    next_m = _ARTICLE_HDR.search(text, pos=m.end())
    end = next_m.start() if next_m else len(text)
    block = text[start:end].strip()
    block = re.sub(r'[ \t]+', ' ', block)
    return block if block else None

def _maybe_loes(meta: dict) -> bool:
    """Heurística simple para identificar nodos de la LOES por metadatos."""
    s = " ".join([str(meta.get(k, "")) for k in ("file_name", "title", "source")]).lower()
    return ("loes" in s) or ("ley organica de educacion superior" in s) or ("ley orgánica de educación superior" in s)
import unicodedata, re

def _norm_txt(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()






# ───────────────────────── Herramienta (Tool) para LangGraph ─────────────────────────

from langchain.tools import tool
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from langchain.tools import tool
# services/agent.py -> Pega esto en la sección de herramientas @tool

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter, FilterCondition

# ===============================================
# HERRAMIENTAS DE ANÁLISIS DE DOCUMENTOS DE CLIENTE
# ===============================================

@tool
def resumir_documento_cliente(file_name: str) -> str:
    """
    Genera un resumen ejecutivo claro del documento del cliente identificado por `file_name`.
    """
    if not file_name:
        return "Error: No se ha especificado un nombre de archivo para resumir."
        
    print(f"📄 Iniciando resumen para el archivo (vía directa ES): {file_name}")
    
    # Usamos nuestra nueva función de búsqueda directa
    nodes = _get_nodes_by_filename_raw(index_name=CFG_PRODUCTO.name, file_name=file_name)

    if not nodes:
        return f"No se encontró contenido para el archivo '{file_name}'. Verifica que se haya indexado correctamente."

    full_text = "\n\n---\n\n".join([node.get_content() for node in nodes if node.get_content()])
    
    if not full_text.strip():
        return f"Se encontró el archivo '{file_name}', pero no se pudo extraer contenido textual para resumir."

    # El resto de la lógica de resumen con el LLM no cambia
    summarizer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un asistente fiel a la fuente. REGLAS: (1) SOLO usa el texto provisto, "
     "(2) NO inventes normativa/fechas, (3) Si falta info, di 'no presente en el documento', "
     "(4) Incluye páginas si existen en metadatos."),
    ("human",
     "Texto del documento (concatenado):\n\n{document_text}\n\n"
     "Tarea: resumen ejecutivo en 5–8 viñetas y luego 'Evidencias textuales' con 3–5 citas literales breves.")
    ])
    chain = prompt | summarizer_llm
    response = chain.invoke({"document_text": full_text})
    return response.content or "No se pudo generar el resumen."

@tool
def analizar_caso_con_normativa(query: str, file_name: str) -> str:
    """
    Resuelve una petición del cliente usando un archivo y la normativa.
    """
    if not file_name:
        return "Error: Para analizar un caso, necesito saber qué archivo del cliente debo usar."
    
    print(f"🛠️ Analizando caso (vía directa ES) para '{file_name}'")
    
    # Obtenemos el contexto del cliente con nuestra nueva función directa
    client_nodes = _get_nodes_by_filename_raw(index_name=CFG_PRODUCTO.name, file_name=file_name)
    client_context = _format_nodes([{'node': n} for n in client_nodes]) if client_nodes else ""
    normative_context = buscar_normativa_avanzada.invoke({"query": query})

    if not client_context.strip():
        return "No se encontró información utilizable en el documento del cliente."
    if "no arrojó resultados" in (normative_context or "").lower():
        return "No se halló normativa relevante con los filtros dados."
        # La búsqueda de normativa sigue usando la lógica de antes, que funciona bien
   

    # El resto de la lógica de síntesis no cambia
    synthesis_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un analista legal. REGLAS: "
     "1) SOLO usa literalmente el CONTEXTO CLIENTE y la NORMATIVA APLICABLE provistos; "
     "2) Si falta algo, dilo explícitamente; "
     "3) No cites artículos o acuerdos que no aparezcan en el contexto; "
     "4) Devuelve referencias [n] de _format_nodes cuando apliquen."),
    ("human",
     "PREGUNTA: {query}\n\nCONTEXTO CLIENTE:\n{client_context}\n\nNORMATIVA APLICABLE:\n{normative_context}\n\n"
     "Entrega: análisis en bullets + lista de referencias mencionadas ([n])")
    ])

    chain = prompt | synthesis_llm
    response = chain.invoke({"query": query, "client_context": client_context, "normative_context": normative_context})
    return response.content or "No se pudo completar el análisis del caso."

# ===============================================
# HERRAMIENTAS DE BÚSQUEDA (VERSIONS ANTERIORES)
# ===============================================




# services/agent.py -> En la sección de @tool

# (ELIMINA las herramientas de búsqueda de normativa anteriores)

# services/agent.py -> REEMPLAZA LA FUNCIÓN COMPLETA

@tool
def buscar_normativa_avanzada(
    query: str,
    institution: Optional[str] = None,
    themes: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    article_number: Optional[str] = None,
    document_title: Optional[str] = None,
    top_k: int = 5
) -> str:
    """
    Herramienta principal para buscar en la base de conocimiento de normativa.
    Realiza una búsqueda vectorial amplia y luego aplica filtros en Python para máxima flexibilidad.
    """
    try:
        print(f"🔎 Búsqueda (fase 1 - recall): query='{query}', institution='{institution}', title='{document_title}'")
        
        # 1. Búsqueda vectorial amplia SIN pre-filtrado en LlamaIndex.
        # Traemos más documentos (ej. 15) para tener una buena base para filtrar.
        retriever = get_retriever_by_cfg(CFG_NORMATIVA, top_k=15, filters=None)
        nodes = retriever.retrieve(query)

        if not nodes:
            return "La búsqueda vectorial inicial no arrojó resultados."

        # 2. Post-filtrado en Python. Aquí es donde aplicamos nuestra lógica "contains".
        filtered_nodes = []
        
        # Si no se especifican filtros, pasamos todos los nodos.
        if not any([institution, document_title, article_number, themes]):
             filtered_nodes = nodes
        else:
            for node_with_score in nodes:
                meta = node_with_score.node.metadata
                
                # Comprobamos cada condición. El nodo debe cumplir TODAS las que se especifiquen.
                institution_match = not institution or (institution.lower() in meta.get("institution", "").lower())
                title_match = not document_title or (document_title.lower() in meta.get("document_title", "").lower())
                article_match = not article_number or (article_number.lower() in meta.get("article_number", "").lower())
                
                # Para temas/keywords, comprobamos si ALGUNO de los temas está presente.
                themes_text = " ".join(meta.get("themes", [])).lower()
                themes_match = not themes or any(theme.lower() in themes_text for theme in themes)

                if institution_match and title_match and article_match and themes_match:
                    filtered_nodes.append(node_with_score)

        print(f"🔎 Búsqueda (fase 2 - post-filtrado): {len(nodes)} nodos recuperados -> {len(filtered_nodes)} nodos filtrados.")

        if not filtered_nodes:
            return "La búsqueda no arrojó resultados con los filtros especificados. Intenta ser menos restrictivo."

        # (Opcional) Rerank sobre los nodos ya filtrados para máxima precisión.
        if _RERANK_OK:
            filtered_nodes = _rerank_by_query(query, filtered_nodes, top_k=top_k)

        # 3. Devolver los mejores N resultados que pasaron el filtro.
        return _format_nodes(filtered_nodes[:top_k])
        
    except Exception as e:
        # Imprimimos el traceback para ver el error completo en el log
        import traceback
        print(f"Error en buscar_normativa_avanzada: {e}")
        traceback.print_exc()
        return "Ocurrió un error técnico al realizar la búsqueda en la normativa."

from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

@tool
def consulta_producto(query: str, file_name: str | None = None) -> str:
    """
    Busca y recupera contenido de los documentos subidos por el cliente (índice: producto_v2).
    Es una herramienta de alta precisión para localizar pasajes relevantes del/los archivo(s)
    del cliente a partir de una consulta en lenguaje natural.

    - Si se proporciona `file_name`, la búsqueda se limita estrictamente a ese documento.
    - Si no se proporciona `file_name`, la búsqueda se realiza sobre todos los documentos
      del cliente disponibles en el índice.
    - Devuelve fragmentos textuales formateados con su fuente y, cuando esté disponible,
      la página de origen.

    Parámetros:
        query (str): Consulta en lenguaje natural o términos clave a buscar en los documentos.
        file_name (str | None): Nombre exacto del archivo del cliente a restringir la búsqueda
            (por ejemplo, "Agenda de capacitación.pdf"). Si es None, busca en todos.

    Retorna:
        str: Texto legible con los fragmentos más relevantes (top-k), incluyendo la referencia
        de la fuente (nombre de archivo y página cuando aplique). Si no se encuentran resultados,
        devuelve un mensaje informativo sin alucinaciones.

    Notas:
        - No introduce normativa ni contenido externo: se limita al corpus del cliente.
        - Para análisis cruzado con normativa usar herramientas específicas (p. ej.,
          `analizar_caso_con_normativa`).
    """
    filters = None
    if file_name:
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="metadata.file_name", value=file_name)]
        )
    retriever = get_retriever_by_cfg(CFG_PRODUCTO, top_k=4, filters=filters)
    try:
        return _format_nodes(retriever.retrieve(query))
    except Exception as e:
        print(f"consulta_producto error: {e}")
        return "Error consultando el índice de producto."


    
# -------- TOOLS: Búsquedas y acciones sobre normativa -------- 



# services/agent.py -> REEMPLAZA la herramienta extraer_articulo
# services/agent.py -> REEMPLAZA las versiones antiguas de estas herramientas

@tool
def extraer_articulo(query: str, articulo: str) -> str:
    """
    Devuelve el texto COMPLETO y específico de un número de artículo de una norma, usualmente la LOES.
    Es una herramienta de alta precisión. Usa la búsqueda avanzada para encontrar el documento correcto y luego
    extrae el bloque de texto relevante.
    """
    try:
        # 1. Búsqueda enfocada usando la herramienta avanzada para encontrar el chunk relevante
        # Asumimos que la mayoría de las solicitudes de artículos son para la LOES.
        # El LLM puede invocar la herramienta con un título diferente si es necesario.
        contexto = buscar_normativa_avanzada.invoke({
            "query": f"texto íntegro del artículo {articulo}",
            "article_number": articulo,
            "document_title": "Ley Orgánica de Educación Superior", # Asume LOES por defecto
            "top_k": 3 # Traemos pocos chunks, pero muy relevantes
        })

        if "no arrojó resultados" in contexto:
            return f"No pude encontrar el artículo {articulo} en la LOES. Verifica el número o la ley."

        # 2. Pedir al LLM que extraiga y resuma el texto a partir del contexto recuperado
        summarizer_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Eres un experto legal que extrae información literal. Del siguiente CONTEXTO, extrae únicamente el texto completo del artículo solicitado. Luego, crea un resumen de 2-3 puntos clave. Si el texto exacto no está, indícalo."),
            ("human", "Artículo solicitado: {articulo}\n\nCONTEXTO:\n{contexto}")
        ])
        
        chain = prompt | summarizer_llm
        response = chain.invoke({"articulo": articulo, "contexto": contexto})
        
        return response.content or "No se pudo procesar la extracción del artículo."

    except Exception as e:
        print(f"extraer_articulo error: {e}")
        return f"Ocurrió un error técnico al extraer el artículo {articulo}."


@tool
def comparar_normas(tema: str, entidad_a: str, entidad_b: str) -> str:
    """
    Compara cómo dos entidades diferentes (ej. CES y CACES) tratan un mismo 'tema'.
    Realiza dos búsquedas filtradas y presenta los resultados para su comparación.
    """
    try:
        # Búsqueda para la Entidad A
        contexto_a = buscar_normativa_avanzada.invoke({
            "query": tema,
            "institution": entidad_a,
            "top_k": 3
        })
        
        # Búsqueda para la Entidad B
        contexto_b = buscar_normativa_avanzada.invoke({
            "query": tema,
            "institution": entidad_b,
            "top_k": 3
        })
        
        # El LLM en el paso final se encargará de sintetizar la comparación.
        # Aquí solo devolvemos los contextos recuperados de forma estructurada.
        respuesta_estructurada = (
            f"### Perspectiva de {entidad_a.upper()} sobre '{tema}':\n{contexto_a}\n\n"
            "---\n\n"
            f"### Perspectiva de {entidad_b.upper()} sobre '{tema}':\n{contexto_b}"
        )
        return respuesta_estructurada

    except Exception as e:
        print(f"comparar_normas error: {e}")
        return "Error al preparar la comparación de normativas."
@tool
def limpiar_contexto() -> str:
    """Instrucción de usuario para ignorar el archivo subido en este turno.""" 
    # La lógica real se maneja en run_agent, esto es para que el agente la pueda 'llamar'
    return "Contexto de archivo ignorado para este turno."

@tool
def set_contexto_archivo(file_name: str) -> str:
    """Permite fijar manualmente el archivo de contexto.""" 
    # La lógica real debería estar en run_agent o en el gestor de estado.
    return f"Contexto de archivo establecido en: {file_name}"

@tool
def listar_citas(n: int = 5) -> str:
    """Devuelve las últimas 'n' citas del turno anterior.""" 
    return "Por favor, utiliza las referencias [n] de la respuesta previa para ubicar la fuente y página." 
################################## Hints ########################################################################
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
    """
    Regresa True si debemos usar el documento subido.
    Política: si hay archivo en contexto, se usa por defecto,
    salvo que el usuario pida explícitamente no usarlo.
    """
    if not file_ctx:
        return False
    t = _normalize(user_input)

    # Comandos para NO usar archivo (opt-out)
    if any(h in t for h in CLEAR_HINTS):
        return False

    # Si menciona el nombre del archivo, obvio sí
    if _mentions_file_name(user_input, file_ctx):
        return True

    # Si pide normativa "pura" y explícita (muchos hits de normativa y nada de doc),
    # podemos dejar que no use archivo
    if _count_hits(t, NORM_HINTS) >= 3 and _count_hits(t, DOC_HINTS) == 0:
        return False

    # 👇 Por defecto: usar archivo si existe file_ctx
    return True


def _is_normative_intent(user_input: str) -> bool:
    t = _normalize(user_input)
    return _count_hits(t, NORM_HINTS) >= 1
def _should_use_file_ctx(user_input: str, file_ctx: str | None, turns_remaining: int) -> bool:
    """
    Política "smart-stick":
    - Usa archivo si: (a) hay file_ctx y lo menciona o hay hints de documento, o
                      (b) estamos dentro de la ventanita de turnos (turns_remaining > 0)
    - No usa archivo si: el usuario pide explícitamente ignorarlo (CLEAR_HINTS) o no hay file_ctx.
    """
    if not file_ctx:
        return False
    t = _normalize(user_input)

    # Opt-out explícito
    if any(h in t for h in CLEAR_HINTS):
        return False

    # Si menciona el archivo por nombre o hay hints de documento, úsalo
    if _mentions_file_name(user_input, file_ctx) or _count_hits(t, DOC_HINTS) >= 1:
        return True

    # Si no lo menciona pero aún estamos en la ventanita, úsalo
    if turns_remaining > 0:
        return True

    # Caso contrario, no usar
    return False


def _update_file_ctx_window(triggered_by_doc_mention: bool, previous_window: int) -> int:
    """
    Reglas de actualización:
    - Si el turno fue "por documento" (lo mencionó o hubo hints), refresca la ventana a 1 (solo el próximo turno).
    - Si el turno usó el archivo únicamente por ventana previa, decrementa.
    - Si no se usó el archivo, resetea a 0.
    """
    if triggered_by_doc_mention:
        return 1  # 1 turno más "pegajoso"
    if previous_window > 0:
        return max(0, previous_window - 1)
    return 0


# services/agent.py

# ... (justo después de los imports)

# -------- Helpers de intención y filtros específicos de normativa -------- 
_ART_RE = re.compile(r"(art[íi]culo(?:s)?\s*(\d+[A-Za-z]?))|(\bArt\.\s*\d+[A-Za-z]?)", re.IGNORECASE)

def _extract_article_number(q: str) -> Optional[str]:
    """Intenta extraer 'artículo X' del texto.""" 
    m = _ART_RE.search(q or "")
    if not m:
        return None
    g = m.group(0) 
    num = re.findall(r"\d+[A-Za-z]?", g)
    return num[0] if num else None 

def _mk_filters_normativa(
    entidad: Optional[str] = None,
    tipo: Optional[str] = None,
    anio_desde: Optional[int] = None,
    anio_hasta: Optional[int] = None,
    institucion: Optional[str] = None,
    # Añadimos un parámetro específico para número de artículo
    article_number: Optional[str] = None
) -> Optional[MetadataFilters]:
    """
    Construye MetadataFilters para buscar en los campos del nivel superior.
    """
    fl = []
    # NOTA: Los campos 'entidad', 'tipo', 'anio' no existen en tu esquema actual,
    # pero los dejamos por si los añades en el futuro.
    if entidad:
        # Tendrías que tener un campo "entidad" en tu documento de ES
        fl.append(ExactMatchFilter(key="entidad", value=entidad.upper()))
    if tipo:
        fl.append(ExactMatchFilter(key="tipo", value=tipo.lower()))

    # --- CAMBIO CLAVE: Filtramos directamente por 'article_number' ---
    if article_number:
        fl.append(ExactMatchFilter(key="article_number", value=str(article_number)))
        
    return MetadataFilters(filters=fl) if fl else None
# -------- Reranking (mejora precisión) -------- 
_reranker = BGEM3FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) if _RERANK_OK else None

def _rerank_by_query(query: str, nodes, top_k: int = 6):
    """nodes: lista de objetos con .node.get_content() (como retorna LlamaIndex)""" 
    if not _RERANK_OK or not nodes:
        return nodes[:top_k] 
    pairs = [(query, getattr(n.node, "text", None) or n.node.get_content() or "") for n in nodes]
    scores = _reranker.compute_score(pairs)
    ranked = [d for _, d in sorted(zip(scores, nodes), key=lambda x: x[0], reverse=True)]
    return ranked[:top_k]

# ───────────────────────── Servicio Principal del Agente ─────────────────────────
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from collections import defaultdict
class RAGAgentService:
    """
    Servicio que encapsula la lógica del Agente RAG (LangGraph), memoria, 
    y utilidades como la generación de títulos.
    """
    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.agent = self._initialize_agent()
        self._file_ctx_window_by_thread = defaultdict(int)
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

        # services/agent.py -> DENTRO de RAGAgentService._initialize_agent

        system_prompt = """
        ## Misión y Personalidad
        Eres LQA-1 (Legal Quantitative Assistant), un asistente de alta precisión especializado en la normativa de Educación Superior de Ecuador. Tu única misión es responder basándote estricta y exclusivamente en la información recuperada por tus herramientas. Eres metódico, preciso y nunca inventas información. Responde siempre en español.

        ---

        ## Jerarquía de Decisión y Herramientas
        Para cada pregunta, sigue rigurosamente este orden de prioridades para elegir la herramienta adecuada:

        1.  **BÚSQUEDA DE NORMATIVA (Herramienta Principal):**
            * Para **CUALQUIER PREGUNTA sobre leyes, reglamentos o resoluciones**, tu herramienta por defecto es `buscar_normativa_avanzada`.
            * **Analiza la pregunta del usuario para extraer entidades y conceptos clave.** Mapea estos conceptos a los parámetros de la herramienta:
                * `institution`: Si mencionan "CES", "CACES", "SENESCYT".
                * `document_title`: Si mencionan el nombre de un documento como "Reglamento de Institutos de Formación Técnica y Tecnológica" o "LOES".
                * `themes` o `keywords`: Si preguntan por "becas", "acreditación", "alianzas".
                * `article_number`: Si mencionan "artículo 18", "art. 25".
            * **EJEMPLO DE RAZONAMIENTO:** Si el usuario pregunta "Según el Reglamento de Institutos del CES, ¿qué dice de las alianzas?", debes invocar la herramienta así: `buscar_normativa_avanzada(query="alianzas", institution="CES", document_title="Reglamento de Institutos")`.

        2.  **HERRAMIENTAS ESPECIALIZADAS (Úsalas solo si la intención es clara):**
            * **`extraer_articulo`**: Úsala **únicamente** si el usuario pide el texto completo y explícito de un artículo.
            * **`comparar_normas`**: Úsala **únicamente** si el usuario pide explícitamente una comparación entre dos entidades.
        
        3.  **HERRAMIENTAS DE ARCHIVOS DE USUARIO:**
            * (El resto de las instrucciones no cambian...)

        ---

        ## Reglas Críticas Inquebrantables
        - (El resto de las instrucciones no cambian...)
        """
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{messages}")]
        )

        # 👇 usa las tools nuevas, no `consulta_corpus`
        toolkit = [
            buscar_normativa_avanzada,
            extraer_articulo,
            comparar_normas,
            resumir_documento_cliente,
            analizar_caso_con_normativa,
            consulta_producto,
            limpiar_contexto,
            set_contexto_archivo,
            listar_citas
        ]

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
    

    # services/agent.py -> REEMPLAZA TU FUNCIÓN run_agent CON ESTA VERSIÓN




    def run_agent(self, conversation_id: str, thread_id: str, user_input: str) -> Dict[str, Any]:
        # --- 1. INICIALIZACIÓN Y LOGGING ---
        user_message_id = self._log_message(conversation_id, "user", user_input)
        final_text = ""
        
        try:
            conv = Conversation.get(Conversation.id == conversation_id)
            file_ctx = conv.last_file_context or None
            matched = _match_uploaded_file(user_input)
            if matched:
                file_ctx = matched
        except DoesNotExist:
            final_text = f"Error crítico: No encontré la conversación '{conversation_id}'."
            agent_message_id = self._log_message(conversation_id, "assistant", final_text)
            return { "idUser": user_message_id, "agent": { "id": agent_message_id, "content": final_text, "sender": "assistant", "timestamp": int(datetime.now().timestamp() * 1000) } }

        # --- 2. CARGAR HISTORIAL DE LA CONVERSACIÓN ---
        print(f"🧠 Cargando historial para la conversación ID: {conversation_id}")
        history_messages = []
        try:
            past_messages = (ChatMessage
                            .select()
                            .where(ChatMessage.conversation_id == conversation_id)
                            .order_by(ChatMessage.ts.desc())
                            .limit(10))
            for msg in reversed(past_messages):
                if msg.role == "user":
                    history_messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    history_messages.append(AIMessage(content=msg.content))
            print(f"🧠 {len(history_messages)} mensajes cargados del historial.")
        except Exception as e:
            print(f"⚠️ No se pudo cargar el historial de la conversación: {e}")

        # --- 3. CONTEXTO DE ARCHIVO con ventana corta ---
        turns_remaining = self._file_ctx_window_by_thread.get(thread_id, 0)
        FILE_LIST_HINTS = ["qué archivos", "qué documentos", "qué tienes en memoria", "qué archivos tengo"]
        if any(h in _normalize(user_input) for h in FILE_LIST_HINTS):
            files = [c.last_file_context for c in Conversation.select().where(Conversation.id == conversation_id)]
            files_txt = ", ".join(filter(None, files)) or "Sin archivos registrados en esta conversación."
            agent_message_id = self._log_message(conversation_id, "assistant", files_txt)
            try:
                self._memorize_turn(thread_id, user_input, files_txt)
            except Exception as e:
                print(f"⚠️ _memorize_turn falló: {e}")
            return {
                "idUser": user_message_id,
                "agent": {"id": agent_message_id, "content": files_txt, "sender": "assistant",
                        "timestamp": int(datetime.now().timestamp() * 1000)}
            }


        use_file_ctx = _should_use_file_ctx(user_input, file_ctx, turns_remaining)

        if use_file_ctx:
            print(f"⚡️ Modo archivo activo (ventana restante={turns_remaining}).")

            # ¿El usuario realmente mencionó el archivo/hints este turno?
            triggered_by_doc_mention = (
                _mentions_file_name(user_input, file_ctx) or
                _count_hits(_normalize(user_input), DOC_HINTS) >= 1
            )

            # Heurística: ¿quiere normativa + archivo?
            if _is_normative_intent(user_input):
                print(f"⚡️ Llamando a 'analizar_caso_con_normativa' con file_ctx.")
                try:
                    final_text = analizar_caso_con_normativa.invoke({
                        "query": user_input,
                        "file_name": file_ctx
                    })
                except Exception as e:
                    final_text = f"Ocurrió un error al analizar el documento con la normativa: {e}"
            else:
                print(f"⚡️ Llamando a 'resumir_documento_cliente' con file_ctx.")
                try:
                    final_text = resumir_documento_cliente.invoke({
                        "file_name": file_ctx
                    })
                except Exception as e:
                    final_text = f"Ocurrió un error al procesar el documento: {e}"

            # Actualiza ventana
            self._file_ctx_window_by_thread[thread_id] = _update_file_ctx_window(
                triggered_by_doc_mention=triggered_by_doc_mention,
                previous_window=turns_remaining
            )

            agent_message_id = self._log_message(conversation_id, "assistant", final_text)
            try:
                self._memorize_turn(thread_id, user_input, final_text)
            except Exception as e:
                print(f"⚠️ _memorize_turn falló: {e}")
            return {
                "idUser": user_message_id,
                "agent": {
                    "id": agent_message_id,
                    "content": final_text,
                    "sender": "assistant",
                    "timestamp": int(datetime.now().timestamp() * 1000)
                }
            }
        else:
            print("▶️ Modo general (archivo NO aplicado).")
            # Si el usuario dijo explícitamente limpiar/ignorar, resetea ventana
            tnorm = _normalize(user_input)
            if any(h in tnorm for h in CLEAR_HINTS):
                self._file_ctx_window_by_thread[thread_id] = 0
        # --- 4. SI NO ES SOBRE UN ARCHIVO, SEGUIMOS CON EL AGENTE GENERAL ---
        print("▶️ Consulta general. Pasando al agente ReAct para que decida.")
        try:
            messages_with_history = history_messages + [HumanMessage(content=user_input)]
            
            result = self.agent.invoke(
                {"messages": messages_with_history},
                config={"configurable": {"thread_id": thread_id}}
            )
            
            final_message = result["messages"][-1]
            final_text = getattr(final_message, "content", None) or "⚠️ Sin respuesta textual del agente."

        except Exception as e:
            print(f"Error al ejecutar el agente LangGraph: {e}")
            final_text = "Lo siento, ocurrió un error interno al procesar tu solicitud."

        # --- 5. LOG Y RETORNO FINAL ---
        agent_message_id = self._log_message(conversation_id, "assistant", final_text)
        try:
            self._memorize_turn(thread_id, user_input, final_text)
        except Exception as e:
            print(f"⚠️ _memorize_turn falló: {e}")

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