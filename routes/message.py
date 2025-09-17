from fastapi import APIRouter

from services.message_serv import log_message
from utils.error_handle import get_details_error
from utils.handle_respose import send_success_response
from models.message_model import AddMessageModel
message_router = APIRouter()


@message_router.post("/")
async def add_message_ctrl(message: AddMessageModel):
    try:
        # guardar mensaje del usuario
        messageUser = log_message(message.conv_id, "user", message.content)

        # logica agente IA
        response_agent = "Respuesta del agente IA"

        messageAgent = log_message(message.conv_id, "agent", response_agent)

        return send_success_response(201, "Mensaje creado", {
            "idUser": messageUser["id"],
            "agent": messageAgent
        })

    except Exception as error:
        return get_details_error(error)

 
