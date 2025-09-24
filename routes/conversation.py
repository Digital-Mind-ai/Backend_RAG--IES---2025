from fastapi import APIRouter, File, UploadFile, Depends, Request
from typing import List
from services.conversation_serv import (
    create_conversation_serv,  # <-- ESTE ES EL NOMBRE CORRECTO
    rename_conversation_serv, 
    archive_conversation_serv,
    unarchive_conversation_serv,
    delete_conversation_serv
)
# ... (otras importaciones) ...
from models.conversation_model import CreateConversationModel
from services.file_serv import upload_files_serv
from services.message_serv import send_and_log_message_serv 
from utils.error_handle import get_details_error
from utils.handle_respose import send_success_response
from models.conversation_model import CreateConversationModel
from services.file_serv import upload_files_serv
from utils.error_handle import get_details_error
from utils.handle_respose import send_success_response
from database import Conversation
from fastapi import APIRouter, File, UploadFile
from typing import List
from database import Conversation, db
from models.conversation_model import CreateConversationModel
from services.conversation_serv import create_conversation_serv
from services.file_serv import upload_files_serv
from services.message_serv import send_and_log_message_serv
from utils.error_handle import get_details_error
from utils.handle_respose import send_success_response
from middlewares.verify_session import session_validator

conversation_router = APIRouter(dependencies=[Depends(session_validator)])

def _resolve_conversation(conv_id_or_thread: str):
    conv = Conversation.get_or_none(Conversation.id == conv_id_or_thread)
    if conv is None:
        conv = Conversation.get_or_none(Conversation.thread_id == conv_id_or_thread)
    return conv

@conversation_router.post("/")
def create_conversation_ctrl(request: Request, data: CreateConversationModel):
    try:
        # Tomar el `username` desde el payload del JWT colocado en `request.state.user`
        user_payload = getattr(request.state, "user", {}) or {}
        user_id = user_payload.get("username")
        if not user_id:
            # Si por alguna razón no viene, se rechaza por seguridad
            raise Exception("No se pudo determinar el usuario desde el token")
        
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

        return send_success_response(201, "Conversación creada y mensaje procesado", conversation)
    

    except Exception as error:
        # Se recomienda manejar el error de la base de datos (ej. usuario no existe) aquí
        return get_details_error(error)

@conversation_router.post("/upload_file/{conv_id}")
async def upload_file_ctrl(request: Request, conv_id: str, files: List[UploadFile] = File(...)):
    try:
        user_payload = getattr(request.state, "user", {}) or {}
        user_id = user_payload.get("username")
        if not user_id:
            raise Exception("No se pudo determinar el usuario desde el token")

        print(f"Subiendo {len(files)} archivo(s) a la conversación {conv_id}")

        # 1) Procesar archivos
        raw_result = await upload_files_serv(files, conv_id)

        # 2) Normalizar resultado (siempre con estas claves)
        result = {
            "conversation_id": conv_id,
            "total_files": raw_result.get("total_files", len(files)) if isinstance(raw_result, dict) else len(files),
            "successful": raw_result.get("successful", 0) if isinstance(raw_result, dict) else 0,
            "failed":     raw_result.get("failed", 0) if isinstance(raw_result, dict) else 0,
            "files":      raw_result.get("files", []) if isinstance(raw_result, dict) else [],
        }

        # Log detallado por archivo (para diagnosticar por qué falló)
        for f in result["files"]:
            print("UPLOAD_RESULT:", f)

        # # 3) Si hubo éxito, guardar contexto por ID real (aunque la URL traiga thread_id)
        # if result["successful"] > 0 and files:
        #     first_filename = files[0].filename
        #     conv = _resolve_conversation(conv_id)
        #     if not conv:
        #         print(f"⚠️ No existe conversación con id/thread_id='{conv_id}'. No se setea contexto.")
        #     else:
        #         with db.atomic():
        #             rows = (Conversation
        #                     .update(last_file_context=first_filename)
        #                     .where(Conversation.id == conv.id)    # ✅ actualizar por UUID real
        #                     .execute())
        #         if rows == 1:
        #             print(f"✅ Contexto de archivo '{first_filename}' guardado para la conversación {conv.thread_id}")
        #         else:
        #             print(f"⚠️ No se pudo guardar contexto de archivo (filas actualizadas={rows}) para {conv.thread_id}")

        # 4) Código HTTP coherente con el resultado
        if result["successful"] > 0 and result["failed"] == 0:
            status_code = 201
            message = f"Todos los archivos ({result['successful']}) se subieron correctamente"
        elif result["successful"] == 0:
            status_code = 400
            message = "No se pudo subir ningún archivo"
        else:
            status_code = 207  # Multi-Status
            message = f"Se subieron {result['successful']} archivos; {result['failed']} fallaron"

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

@conversation_router.put("/unarchive/{conv_id}")
def unarchive_conversation_ctrl(conv_id: str):
    try:
        success = unarchive_conversation_serv(conv_id)
        if not success:
            return send_success_response(404, "Conversación no encontrada")

        return send_success_response(200, "Conversación desarchivada")
    except Exception as error:
        return get_details_error(error)


