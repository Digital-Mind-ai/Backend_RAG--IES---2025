import uuid


def log_message(conversation_id: str, role: str, content: str):
    # cambiar por toda la logica de guardado en base de datos
    message_id = str(uuid.uuid4())
    
    # No cambiar este return, es la estructura que espera el frontend
    return {"id": message_id,
            "text": content,
            "sender": role,
            "timestamp": "154845418"}
