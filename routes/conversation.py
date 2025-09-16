from fastapi import APIRouter

from models.conversation_model import CreateConversationModel
from services.conversation_serv import create_conversation
from utils.error_handle import get_details_error
from utils.handle_respose import send_success_response

conversation_router = APIRouter()


@conversation_router.post("/")
# def create_conversation_ctrl(data: CreateConversationModel):
def create_conversation_ctrl():
    try:
        
        conversation = create_conversation("user_label_example", "Título de ejemplo")
        print(f"Conversación creada: {conversation}")
        
        return send_success_response(201, "Conversación creada", conversation)

    except Exception as error:
        return get_details_error(error)

@conversation_router.post("/upload_file")
# def create_conversation_ctrl(data: CreateConversationModel):
def create_conversation_ctrl():
    try:
        # Lógica para manejar la subida de archivos
        files = []
        
        # respuesta de éxito
        return send_success_response(201, "Archivos creados", files)

    except Exception as error:
        return get_details_error(error)
