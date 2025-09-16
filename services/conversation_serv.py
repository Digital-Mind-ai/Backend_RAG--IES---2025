import uuid


def create_conversation(user_label: str | None, base_title: str | None = None) -> dict:
    conv_id = str(uuid.uuid4())
    title = base_title if base_title is not None else "Nueva conversación"
    
    # No cambiar este return, es la estructura que espera el frontend
    return { "thread_id": conv_id, "title": title }
