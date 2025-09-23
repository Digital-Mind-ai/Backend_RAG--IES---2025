# routes/message.py 
from fastapi import APIRouter

from models.message_model import AddMessageModel
#  Importa la función RAG real, NO la simulada
from services.message_serv import send_and_log_message_serv 
from utils.error_handle import get_details_error
from utils.handle_respose import send_success_response

message_router = APIRouter()

@message_router.post("/")
def add_message_ctrl(message: AddMessageModel): # message ahora solo tiene conv_id y content
    try:
        print(f"Nuevo mensaje en la conversación {message.conv_id}: {message.content}")
        
        # Llama al servicio actualizado (sin file_context)
        agent_result = send_and_log_message_serv(
            conversation_id=message.conv_id, 
            user_input=message.content
        )

        return send_success_response(201, "Mensaje enviado y respuesta recibida", agent_result)
    except Exception as error:
        return get_details_error(error)