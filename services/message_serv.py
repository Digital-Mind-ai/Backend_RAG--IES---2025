import uuid

# services/message_serv.py (ADAPTADO)
from typing import Dict, Any
from peewee import DoesNotExist

# Importaciones del proyecto
from database import Conversation 
from services.agent import rag_agent_service 


def send_and_log_message_serv(conversation_id: str, user_input: str) -> Dict[str, Any]:
    """
    Orquesta el envío del mensaje al Agente RAG.
    
    Se encarga de:
    1. Obtener el thread_id de LangGraph (de la conversación Peewee).
    2. Ejecutar el agente RAG (que loguea ambos mensajes en la DB).
    3. Retornar la respuesta del agente.
    """
    
    try:
        conversation = Conversation.get(Conversation.id == conversation_id)
        thread_id = conversation.thread_id
        
    except DoesNotExist:
        # Usamos ValueError ya que HTTPException no se lanza en servicios
        raise ValueError(f"La conversación con ID {conversation_id} no existe.")

    # 2. Llamar al Agente RAG Service
    agent_response_data = rag_agent_service.run_agent(
        conversation_id=conversation_id,
        thread_id=thread_id,
        user_input=user_input
    )

    print("Respuesta del agente RAG:", agent_response_data)
    
    return agent_response_data

# (Dejamos la función log_message original por si otras rutas la necesitan)
def log_message(conversation_id: str, role: str, content: str) -> Dict[str, Any]:
    """
    Función de logueo de mensajes que NO pasan por la lógica del Agente RAG.
    """
    from services.agent import rag_agent_service 
    from datetime import datetime
    
    message_id = rag_agent_service._log_message(conversation_id, role, content)
    
    return {"id": message_id,
            "content": content,
            "sender": role,
            "timestamp": "154845418"}