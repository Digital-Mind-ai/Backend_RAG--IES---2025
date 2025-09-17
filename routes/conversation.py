from fastapi import APIRouter

from models.conversation_model import CreateConversationModel, UploadFileModel
from services.conversation_serv import create_conversation
from utils.error_handle import get_details_error
from utils.handle_respose import send_success_response

conversation_router = APIRouter()

@conversation_router.post("/")
def create_conversation_ctrl(data: CreateConversationModel):
    try:
        print(f"Conversación creada: {data}")
        conversation = create_conversation("user_label_example", data.firstMessage)
        
        return send_success_response(201, "Conversación creada", conversation)

    except Exception as error:
        return get_details_error(error)

@conversation_router.post("/upload_file")
def upload_file_ctrl(file: UploadFileModel):
    try:
        # Lógica para manejar la subida de archivos
        files = []
        
        # respuesta de éxito
        return send_success_response(201, "Archivos subidos")

    except Exception as error:
        return get_details_error(error)

@conversation_router.delete("/{conv_id}")
def delete_conversation_ctrl(conv_id: str):
    try:
        # Lógica para manejar la eliminación de la conversación
        
        # respuesta de éxito
        return send_success_response(200, "Conversación eliminada")

    except Exception as error:
        return get_details_error(error)
    
@conversation_router.put("/rename/{conv_id}/to/{new_name}")
def rename_conversation_ctrl(conv_id: str, new_name: str):
    try:
        # Lógica para manejar el renombrado de la conversación
        
        # respuesta de éxito
        return send_success_response(200, "Conversación renombrada")

    except Exception as error:
        return get_details_error(error)

@conversation_router.put("/archive/{conv_id}")
def archive_conversation_ctrl(conv_id: str):
    try:
        # Lógica para manejar el archivado de conversaciones
        
        # respuesta de éxito
        return send_success_response(200, "Conversación archivada")

    except Exception as error:
        return get_details_error(error)

