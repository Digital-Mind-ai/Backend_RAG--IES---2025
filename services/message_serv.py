import uuid

# services/message_serv.py (ADAPTADO)
from typing import Dict, Any, List
from peewee import DoesNotExist

# Importaciones del proyecto
from database import Conversation, ChatMessage, AttachFile
from services.agent import rag_agent_service 
from peewee import DoesNotExist
from database import Conversation 
from services.agent import rag_agent_service 

from peewee import DoesNotExist
from database import Conversation
from services.agent import rag_agent_service

def send_and_log_message_serv(conversation_id: str, user_input: str):
    # `conversation_id` puede venir como UUID (id) o como slug (thread_id)
    conv = Conversation.get_or_none(Conversation.id == conversation_id)
    if conv is None:
        conv = Conversation.get_or_none(Conversation.thread_id == conversation_id)
    if conv is None:
        raise ValueError(f"No existe una conversación con identificador '{conversation_id}'")

    # ⚠️ Importante: al agente pásale SIEMPRE el UUID real + el thread_id
    return rag_agent_service.run_agent(
        conversation_id=conv.id,
        thread_id=conv.thread_id,
        user_input=user_input
    )