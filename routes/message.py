# routes/message.py 
from fastapi import APIRouter

from models.message_model import AddMessageModel
#  Importa la función RAG real, NO la simulada
from services.message_serv import send_and_log_message_serv 
from utils.error_handle import get_details_error
from utils.handle_respose import send_success_response

message_router = APIRouter()

@message_router.post("/")
def add_message_ctrl(message: AddMessageModel):
    try:
        
        print(f"Nuevo mensaje en la conversación {message.conv_id}: {message.content}")
        
        # Llama a la función que ejecuta el RAG completo.
        # Esta función hace el log del usuario, ejecuta el agente, y loguea la respuesta.
        agent_result = send_and_log_message_serv(
            conversation_id=message.conv_id, 
            user_input=message.content
        )

        # Retornamos la respuesta del agente
        return send_success_response(201, "Mensaje enviado y respuesta recibida", agent_result)

    except Exception as error:
        return get_details_error(error)
