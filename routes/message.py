from fastapi import APIRouter

from models.message_model import AddMessageModel
from services.message_serv import log_message
from utils.error_handle import get_details_error
from utils.handle_respose import send_success_response

message_router = APIRouter()

@message_router.post("/")
def add_message_ctrl(message: AddMessageModel):
    try:
        
        print(f"Nuevo mensaje en la conversación {message}")
        # guardar mensaje del usuario
        messageUser = log_message(message.conv_id, "user", message.content)

        # logica asistente IA
        response_agent = "Respuesta del asistente IA"

        messageAgent = log_message(message.conv_id, "asistente", response_agent)

        return send_success_response(201, "Mensaje creado", {
            "idUser": messageUser["id"],
            "agent": messageAgent
        })

    except Exception as error:
        return get_details_error(error)

 
