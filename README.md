# Backend — Agente IA + RAG (FastAPI, LangGraph) -
# Backend_RAG--IES---2025
Backend_RAG - IES - 2025 de Digital Mind


## ✨ Resumen

API de preguntas y respuestas especializada en normativa de educación superior de Ecuador. Utiliza un sistema **RAG** (Retrieval-Augmented Generation) y un **agente orquestador con LangGraph** para ofrecer respuestas precisas con citas. La solución expone endpoints para autenticación, gestión de conversaciones, ingesta de documentos y consultas. Está desplegada en una **VM de Google Cloud Platform**.

## 🧱 Arquitectura

- **Framework**: **FastAPI** con Uvicorn para el servicio asíncrono.
- **Orquestación**: Agente **ReAct** implementado con **LangGraph** que decide entre un conjunto de herramientas especializadas.
- **RAG**: Pipeline de **LlamaIndex** y **LangChain** que utiliza **Elasticsearch** como base de datos vectorial y modelos de `text-embedding-3-small` para los embeddings.
- **LLM**: **OpenAI GPT-4o** para la generación de respuestas y síntesis de información.
- **Base de Datos**: **PostgreSQL** para persistir usuarios, conversaciones, mensajes y como `checkpointer` para la memoria del agente de LangGraph.
- **Observabilidad**: Integración nativa con **LangSmith** para el trazado y depuración de las cadenas y el agente.
- **Toolkit del Agente**:
  - `buscar_normativa_avanzada`: Busca en el corpus normativo.
  - `resumir_documento_cliente`: Resume un archivo subido por el usuario.
  - `analizar_caso_con_normativa`: Cruza la información de un archivo con la normativa.
  - `extraer_articulo`: Extrae el texto completo de un artículo específico.
  - Y más...

## 🚀 Endpoints Principales

- `POST /api/v1/auth/login`: Autenticación de usuarios.
- `POST /api/v1/conversation/`: Crea una nueva conversación.
- `POST /api/v1/conversation/upload_file/{conv_id}`: Sube e indexa un archivo en una conversación.
- `POST /api/v1/message/`: Envía un mensaje a una conversación y obtiene la respuesta del agente.
- `GET /`: Endpoint de salud de la aplicación.

#### Ejemplo de consulta con `curl`
```bash
TOKEN="tu_jwt_token"
CONV_ID="id_de_la_conversacion"

curl -X POST http://<IP_DE_TU_VM>:8080/api/v1/message/ \
-H "Authorization: Bearer $TOKEN" \
-H "Content-Type: application/json" \
-d '{
  "conv_id": "'"$CONV_ID"'",
  "content": "¿Cuáles son los requisitos para ser rector según la LOES?"
}'

### Desarrollo
Colocar las variables de entorno ~.env~ 

- Base de datos: DATABASE_URL
- docker-compose -f docker-compose.dev.yml up


