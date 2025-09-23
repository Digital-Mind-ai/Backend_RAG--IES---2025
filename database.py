# database.py

from datetime import datetime
from urllib.parse import urlparse

from decouple import config
from peewee import (
    CharField,
    DateTimeField,
    FloatField,
    BooleanField,
    ForeignKeyField,
    IntegerField,
    Model,
    PostgresqlDatabase,
    TextField,
    IntegerField,
    BigAutoField,
)

#  Ojo: Lee la variable DATABASE_URL de tu archivo .env
DATABASE_URL = config("DATABASE_URL") 

# Parsea la URL de conexión
url = urlparse(DATABASE_URL)
db_params = {
    "database": url.path[1:],  # Ignorar el primer "/" en la ruta
    "user": url.username,
    "password": url.password,
    "host": url.hostname,
    "port": url.port,
}
# La instancia de conexión es 'db'
db = PostgresqlDatabase(**db_params)


# Modelo base
class BaseModel(Model):
    class Meta:
        database = db


# Modelo de Usuario (Asumimos que el user_id es el username)
class User(BaseModel):
    # La clave primaria de esta tabla es el 'username'
    username = CharField(primary_key=True)
    # NOTA: En tu lógica de registro, esto es 'cedulaT', pero aquí lo mantendremos como 'username'
    password = CharField()


class Conversation(BaseModel):
    id = CharField(primary_key=True)
    user = ForeignKeyField(User, backref="conversations", on_delete="CASCADE")
    title = CharField()
    thread_id = CharField()
    user_label = CharField()
    isArchived = BooleanField(default=False)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)
    # 👇 NUEVO CAMPO
    last_file_context = TextField(null=True)  # o CharField(null=True)


# Modelo de Mensaje de Chat
class ChatMessage(BaseModel):
    id = BigAutoField(primary_key=True)  # Definir como clave primaria
    # Relación a la conversación. Cuando se borra la conversación, se borran los mensajes (CASCADE)
    conversation_id = ForeignKeyField(
        Conversation, backref="chat_messages", on_delete="CASCADE"
    )  
    role = CharField(
        choices=[("user", "User"), ("assistant", "Assistant"), ("tool", "Tool")]
    )
    content = TextField()
    ts = DateTimeField(default=datetime.now)


# Conectar y crear la tabla si no existe
# Esto se ejecuta en el inicio de la aplicación

def ensure_conversation_has_last_file_context():
    # Comprueba si la columna existe; si no, la crea
    with db.connection_context():
        row = db.execute_sql("""
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'conversation'
              AND column_name = 'last_file_context'
            LIMIT 1
        """).fetchone()
        if not row:
            db.execute_sql('ALTER TABLE "conversation" ADD COLUMN last_file_context TEXT;')
            print("✅ Column 'last_file_context' añadida a table conversation.")

db.connect()
ensure_conversation_has_last_file_context()   # 👈 añade esta línea
db.create_tables([User, Conversation, ChatMessage], safe=True)

