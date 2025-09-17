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
)

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
db = PostgresqlDatabase(**db_params)


# Modelo base
class BaseModel(Model):
    class Meta:
        database = db


# Modelo de Usuario
class User(BaseModel):
    username = CharField(primary_key=True)
    password = CharField()


# Modelo de Conversación
class Conversation(BaseModel):
    id = CharField(primary_key=True)
    user = ForeignKeyField(User, backref="conversations", on_delete="CASCADE")
    title = CharField()
    thread_id = CharField()
    user_label = CharField()
    isArchived = BooleanField(default=False)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)


# Modelo de Mensaje de Chat
class ChatMessage(BaseModel):
    id = IntegerField(primary_key=True)  # Definir como clave primaria
    conversation_id = ForeignKeyField(
        Conversation, backref="chat_messages", on_delete="CASCADE"
    )  # Relación de clave foránea con Conversación
    role = CharField(
        choices=[("user", "User"), ("assistant", "Assistant"), ("tool", "Tool")]
    )
    content = TextField()
    ts = DateTimeField(default=datetime.now)


# Conectar y crear la tabla si no existe
db.connect()
db.create_tables([User, Conversation, ChatMessage], safe=True)
