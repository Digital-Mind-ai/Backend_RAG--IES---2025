# routes/message.py 
from fastapi import APIRouter, Depends

from models.message_model import AddMessageModel
#  Importa la función RAG real, NO la simulada
from services.message_serv import send_and_log_message_serv, get_messages_with_attachments_serv
from utils.error_handle import get_details_error
from utils.handle_respose import send_success_response
from middlewares.verify_session import session_validator

message_router = APIRouter(dependencies=[Depends(session_validator)])

@message_router.get("/{conversation_id}")
def get_messages_ctrl(conversation_id: str):
    """Obtiene todos los mensajes de una conversación con sus archivos adjuntos."""
    try:
        messages = get_messages_with_attachments_serv(conversation_id)
        return send_success_response(200, "Mensajes obtenidos exitosamente", messages)
    except Exception as error:
        return get_details_error(error)

@message_router.post("/")
def add_message_ctrl(message: AddMessageModel):
    try:
        
        print(f"Nuevo mensaje en la conversación {message.conv_id}: {message.content}")
        
        # Convertir los attachments de Pydantic a diccionarios
        attachments_data = None
        if message.attachments:
            attachments_data = [
                {
                    'name': attachment.name,
                    'url': attachment.url,
                    'type': attachment.type
                }
                for attachment in message.attachments
            ]
        
        # Llama a la función que ejecuta el RAG completo.
        # Esta función hace el log del usuario, ejecuta el agente, y loguea la respuesta.
        agent_result = send_and_log_message_serv(
            conversation_id=message.conv_id, 
            user_input=message.content,
            attachments=attachments_data
        )

        # Retornamos la respuesta del agente
        return send_success_response(201, "Mensaje enviado y respuesta recibida", agent_result)

    except Exception as error:
        return get_details_error(error)

 
