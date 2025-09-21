from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# from routes.auth import auth_router
from routes.conversation import conversation_router
from routes.message import message_router
from utils.handle_respose import send_error_response

app = FastAPI()


# Configuración de CORS
origins = [
    "http://localhost:5173",  # Para permitir solicitudes desde localhost
    "*"  # Para permitir testing en plataformas como PORT de vscode
    # "https://jeffersondaviid.github.io",  # Dominio de producción permitido
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Orígenes permitidos
    allow_credentials=True,
      # Permitir cookies y credenciales
    allow_methods=["*"],  # Métodos permitidos (GET, POST, etc.)
    allow_headers=["*"],  # Encabezados permitidos
)


@app.get("/", response_class=HTMLResponse)
def read_root():
    return "<h1>Bienvenido a DIGITAL MIND</h1>"


# app.include_router(auth_router, prefix="/api/v1/auth")
app.include_router(conversation_router, prefix="/api/v1/conversation")
app.include_router(message_router, prefix="/api/v1/message")


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc: HTTPException):
    # Comprobamos si `exc.detail` es un dict, y lo formateamos.
    if isinstance(exc.detail, dict):
        return send_error_response(
            exc.status_code, exc.detail["message"], exc.detail["error"]
        )
    return send_error_response(exc.status_code, exc.detail)
