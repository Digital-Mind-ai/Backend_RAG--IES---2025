# services/conversation_serv.py 
import uuid
from datetime import datetime
import uuid
import re
from peewee import DoesNotExist 

# Importaciones del proyecto (DB, Models)
from database import User, Conversation, db
# Importamos el servicio del agente para la generación de títulos
from services.agent import rag_agent_service 
from datetime import datetime
import uuid
import re
from peewee import DoesNotExist, IntegrityError # Importamos IntegrityError

# Importaciones del proyecto (DB, Models)
from database import User, Conversation, db
from services.agent import rag_agent_service 

def _slugify(s: str) -> str:
    """Función para limpiar un string a un slug, útil para thread_id."""
    return re.sub(r'[^a-z0-9]+', '-', (s or '').lower()).strip('-')

def create_conversation_serv(user_id: str, first_message: str | None = None) -> dict:
    """
    Crea y persiste una nueva conversación.
    Si el usuario no existe, lo crea automáticamente (Permite usuarios anónimos/de prueba).
    """
    conv_id = str(uuid.uuid4())
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 1. UPSERT (Crear o asegurar) el Usuario
    try:
        # Intenta obtener el usuario
        User.get(User.username == user_id)
        
    except DoesNotExist:
        # Si el usuario NO existe, lo creamos.
        # NOTA: Como la contraseña es CharField NOT NULL, le damos una simple/temporal.
        try:
            with db.atomic():
                User.create(
                    username=user_id,
                    password="auto_generated_password" # Usa una contraseña placeholder
                )
            print(f"✅ Usuario '{user_id}' creado automáticamente.")
        except IntegrityError as e:
            # Esto maneja una rara condición de carrera si dos procesos lo crean a la vez
            print(f"Advertencia: El usuario '{user_id}' ya existe o hubo un error de integridad: {e}")
            pass
        except Exception as e:
            # Si la creación falla por otra razón (ej. conexión DB), lanzamos el error
            raise ValueError(f"No se pudo crear el usuario '{user_id}': {e}")
            
    # 2. Generar Título usando el Agente (mismo código que antes)
    if first_message:
        title = rag_agent_service.generate_title_from_text(first_message)
    else:
        title = f"Chat con {user_id}"

    # 3. Crear el thread_id único para LangGraph y persistir la conversación (mismo código)
    base = _slugify(user_id) or "anon"
    thread_id = f"{base}-{ts}-{conv_id[:8]}"
    
    try:
        with db.atomic():
            new_conv = Conversation.create(
                id=conv_id,
                user=user_id,
                title=title, 
                thread_id=thread_id,
                user_label=user_id
            )
            
            return {
                "id": new_conv.id, 
                "thread_id": new_conv.id,
                "title": new_conv.title
            }
            
    except Exception as e:
        print(f"Error al crear conversación: {e}")
        raise ValueError(f"No se pudo crear la conversación: {e}")
    
    # services/conversation_serv.py (AÑADIR ESTAS FUNCIONES)

def rename_conversation_serv(conv_id: str, new_name: str) -> bool:
    """Renombra una conversación existente en la DB."""
    try:
        with db.atomic():
            query = Conversation.update(
                title=new_name[:80], updated_at=datetime.now()
            ).where(Conversation.id == conv_id)
            return query.execute() > 0
    except Exception:
        return False

def archive_conversation_serv(conv_id: str) -> bool:
    """Archiva/desarchiva una conversación cambiando el campo isArchived."""
    try:
        with db.atomic():
            # Alterna el estado de archivado (si quieres) o lo establece a True
            query = Conversation.update(
                isArchived=True, updated_at=datetime.now()
            ).where(Conversation.id == conv_id)
            return query.execute() > 0
    except Exception:
        return False

def delete_conversation_serv(conv_id: str) -> bool:
    """Elimina una conversación (y sus mensajes por CASCADE) de la DB."""
    try:
        with db.atomic():
            # La eliminación del registro Conversation elimina ChatMessage por CASCADE
            query = Conversation.delete().where(Conversation.id == conv_id)
            return query.execute() > 0
    except Exception:
        return False