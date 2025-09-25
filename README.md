# Backend — Agente IA + RAG (FastAPI, LangGraph)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-green?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangChain-blueviolet?logo=langchain" alt="LangChain">
  <img src="https://img.shields.io/badge/Elasticsearch-orange?logo=elasticsearch" alt="Elasticsearch">
  <img src="https://img.shields.io/badge/PostgreSQL-darkblue?logo=postgresql&logoColor=white" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/Docker-blue?logo=docker&logoColor=white" alt="Docker">
</p>

## Resumen

API de preguntas y respuestas especializada en normativa de educación superior de Ecuador. Utiliza un sistema **RAG** (Retrieval-Augmented Generation) y un **agente orquestador con LangGraph** para ofrecer respuestas precisas con citas. La solución expone endpoints para autenticación, gestión de conversaciones, ingesta de documentos y consultas. Está diseñada para ser desplegada en una **VM de Google Cloud Platform**.

---

## Tabla de Contenidos
1. [Arquitectura](#-arquitectura)
2. [Endpoints Principales](#-endpoints-principales)
3. [Instalación y Ejecución Local](#️-instalación-y-ejecución-local)
4. [Despliegue en VM de Google Cloud](#️-despliegue-en-vm-de-google-cloud)
5. [Cómo Probar el Agente](#-cómo-probar-el-agente)
6. [Variables de Entorno](#-variables-de-entorno)
7. [Estructura del Código](#-estructura-del-código)

---

## 🧱 Arquitectura

- **Framework**: **FastAPI** con Uvicorn para el servicio asíncrono.
- **Orquestación**: Agente **ReAct** implementado con **LangGraph** que decide entre un conjunto de herramientas especializadas.
- **RAG**: Pipeline de **LlamaIndex** y **LangChain** que utiliza **Elasticsearch** como base de datos vectorial y modelos de `text-embedding-3-small` para los embeddings.
- **LLM**: **OpenAI GPT-4o** para la generación de respuestas y síntesis de información.
- **Base de Datos**: **PostgreSQL** para persistir usuarios, conversaciones, mensajes y como `checkpointer` para la memoria del agente de LangGraph.
- **Observabilidad**: Integración nativa con **LangSmith** para el trazado y depuración de las cadenas y el agente.
- **Toolkit del Agente**:
  - `buscar_normativa_avanzada`: Realiza búsquedas dentro del corpus normativo.
  - `extraer_articulo`: Obtiene el texto completo de un artículo específico.
  - `comparar_normas`: Contrasta y compara diferentes normas o disposiciones.
  - `resumir_documento_cliente`: Resume un archivo cargado por el usuario.
  - `analizar_caso_con_normativa`: Cruza la información de un documento con la normativa aplicable.
  - `consulta_producto`: Permite consultar información sobre productos normativos o regulatorios.
  - `limpiar_contexto`: Reinicia o limpia el contexto de trabajo actual.
  - `set_contexto_archivo`: Define un archivo como referencia en el contexto de análisis.
  - `listar_citas`: Extrae y organiza las citas normativas de un documento.
## 🚀 Endpoints Principales

- `POST /api/v1/auth/login`: Autenticación de usuarios.
- `POST /api/v1/conversation/`: Crea una nueva conversación.
- `POST /api/v1/conversation/upload_file/{conv_id}`: Sube e indexa un archivo en una conversación.
- `POST /api/v1/message/`: Envía un mensaje a una conversación y obtiene la respuesta del agente.
- `GET /`: Endpoint de salud de la aplicación.

## ⚙️ Instalación y Ejecución Local

```bash
# 1. Clonar el repositorio
git clone [https://github.com/Digital-Mind-ai/Backend_RAG--IES---2025.git](https://github.com/Digital-Mind-ai/Backend_RAG--IES---2025.git)
cd Backend_RAG--IES---2025

# 2. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
nano .env # Llenar con tus credenciales

# 5. Ejecutar localmente
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## ☁️ Despliegue en VM de Google Cloud

Construir la imagen de Docker:

```bash
docker build -t rag-ies-backend .
```

Ejecutar el contenedor:

```bash
docker run -d -p 8080:8080 --env-file .env   --restart always --name rag-backend rag-ies-backend
```

Configurar Firewall:  
Asegúrate de crear una regla de firewall en tu proyecto de GCP para permitir el tráfico TCP de entrada en el puerto 8080.

## 🧪 Cómo Probar el Agente

Sigue estos pasos usando curl o un cliente de API como Postman.

Autenticarse y guardar el TOKEN:

```bash
# Reemplaza '...' con tu usuario y contraseña
TOKEN=$(curl -s -X POST http://<IP_DE_TU_VM>:8080/api/v1/auth/login -H "Content-Type: application/json" -d '{"username":"...","password":"..."}' | jq -r .data.token)

echo "Token obtenido: $TOKEN"
```

Crear una conversación y guardar el CONV_ID:

```bash
# Reemplaza '...' con el primer mensaje que desees
CONV_ID=$(curl -s -X POST http://<IP_DE_TU_VM>:8080/api/v1/conversation/ -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"firstMessage": "Hola, necesito ayuda con una consulta."}' | jq -r .data.id)

echo "ID de conversación: $CONV_ID"
```

Enviar una pregunta al agente:

```bash
curl -X POST http://<IP_DE_TU_VM>:8080/api/v1/message/ -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{
  "conv_id": "'"$CONV_ID"'",
  "content": "¿Cuáles son los requisitos para ser rector según la LOES?"
}'
```

Verificar la respuesta:  
La salida JSON debe contener el texto generado por el agente y las referencias a las fuentes consultadas.

## 🔑 Variables de Entorno

Crea un archivo .env a partir de .env.example con las siguientes claves:

```bash
# Conexión a la base de datos relacional
DATABASE_URL="postgresql://user:password@host:port/dbname"

# Clave de OpenAI
OPENAI_API_KEY="sk-..."

# Credenciales de Elasticsearch
ES_URL="http://es_host:9200"
ES_USER="elastic"
ES_PASSWORD="..."
ES_INDEX_PRODUCTO="producto_v2"
ES_INDEX_NORMATIVA="ies_normativa_multi_tenant_original"

# Claves de LangChain/LangSmith para observabilidad
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="lsv2_..."
LANGCHAIN_PROJECT="RAG-IES-Agent-MVP"
```

## 🧩 Estructura del Código

```
.
├── routes/        # Definición de endpoints de la API
├── services/      # Lógica de negocio (agente, auth, conversaciones)
├── database.py    # Configuración de Peewee + modelos de datos
├── main.py        # Punto de entrada de la aplicación FastAPI
├── requirements.txt
└── Dockerfile
```
