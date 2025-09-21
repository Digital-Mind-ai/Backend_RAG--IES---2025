from fastapi import APIRouter, File, UploadFile
from typing import List
from fastapi import APIRouter, File, UploadFile
from typing import List
from services.conversation_serv import (
    create_conversation_serv,  # <-- ESTE ES EL NOMBRE CORRECTO
    rename_conversation_serv, 
    archive_conversation_serv, 
    delete_conversation_serv
)
# ... (otras importaciones) ...
from models.conversation_model import CreateConversationModel
from services.conversation_serv import create_conversation_serv 
from services.file_serv import upload_files_serv
from services.message_serv import send_and_log_message_serv 
from utils.error_handle import get_details_error
from utils.handle_respose import send_success_response
from models.conversation_model import CreateConversationModel
from services.file_serv import upload_files_serv
from utils.error_handle import get_details_error
from utils.handle_respose import send_success_response

conversation_router = APIRouter()


@conversation_router.post("/")
def create_conversation_ctrl(data: CreateConversationModel):
    try:
        # ⚠️ IMPORTANTE: Aquí se debe obtener el ID del usuario LÓGICO (username) 
        # del token JWT o del contexto de la solicitud, no usar un valor fijo.
        user_id = "user_label_example" 
        
        # 1. Crear la conversación, generando el título con el primer mensaje
        conversation = create_conversation_serv(user_id, data.firstMessage)
        
        response_data = {
            "conversation": conversation,
            "initial_agent_response": None
        }

        # 2. Procesar el primer mensaje con el Agente RAG 
        if data.firstMessage:
            # Esta llamada loguea el mensaje del usuario y la respuesta del asistente
            agent_result = send_and_log_message_serv(
                conversation_id=conversation["id"],
                user_input=data.firstMessage
            )
            response_data["initial_agent_response"] = agent_result

        return send_success_response(201, "Conversación creada y mensaje procesado", {"thread_id":conversation["thread_id"], "title":conversation["title"]})

    except Exception as error:
        # Se recomienda manejar el error de la base de datos (ej. usuario no existe) aquí
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
        success = delete_conversation_serv(conv_id)
        if not success:
            return send_success_response(404, "Conversación no encontrada o ya eliminada")
        
        return send_success_response(200, "Conversación eliminada")
    except Exception as error:
        return get_details_error(error)
    
@conversation_router.put("/rename/{conv_id}/to/{new_name}")
def rename_conversation_ctrl(conv_id: str, new_name: str):
    try:
        success = rename_conversation_serv(conv_id, new_name)
        if not success:
            return send_success_response(404, "Conversación no encontrada")

        return send_success_response(200, "Conversación renombrada")
    except Exception as error:
        return get_details_error(error)

@conversation_router.put("/archive/{conv_id}")
def archive_conversation_ctrl(conv_id: str):
    try:
        success = archive_conversation_serv(conv_id)
        if not success:
            return send_success_response(404, "Conversación no encontrada")
            
        return send_success_response(200, "Conversación archivada")
    except Exception as error:
        return get_details_error(error)
    

