from fastapi import APIRouter

from models.message_model import AddMessageModel
from services.message_serv import log_message
from utils.error_handle import get_details_error
from utils.handle_respose import send_success_response

message_router = APIRouter()


@message_router.post("/")
def add_message_ctrl(data: AddMessageModel):
    try:
        # guardar mensaje del usuario
        messageUser = log_message(data.conv_id, "user", data.content)
        
        # logica agente IA (omitir por ahora)
        
        # guardar respuesta del agente IA (omitir por ahora)
        messageAgent = log_message(data.conv_id, "agent", "Respuesta del agente IA")
        
        return send_success_response(201, "Mensaje creado", {
            "idUser": messageUser["id"],
            "agent": messageAgent
        })

    except Exception as error:
        return get_details_error(error)
