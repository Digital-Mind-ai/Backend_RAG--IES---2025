from fastapi import APIRouter, File, UploadFile
from typing import List

from models.conversation_model import CreateConversationModel
from services.conversation_serv import create_conversation
from services.file_serv import upload_files_serv
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

@conversation_router.post("/upload_file/{conv_id}")
async def upload_file_ctrl(conv_id: str, files: List[UploadFile] = File(...)):
    try:
        print(f"Subiendo {len(files)} archivo(s) a la conversación {conv_id}")
        
        # Procesar archivos usando el servicio
        result = await upload_files_serv(files, conv_id)
        
        # Determinar el mensaje de respuesta
        if result["failed"] == 0:
            message = f"Todos los archivos ({result['successful']}) se subieron correctamente"
            status_code = 201
        elif result["successful"] == 0:
            message = "No se pudo subir ningún archivo"
            status_code = 400
        else:
            message = f"Se subieron {result['successful']} archivos, {result['failed']} fallaron"
            status_code = 207  # Multi-Status
        
        return send_success_response(status_code, message, result)

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

