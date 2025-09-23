from datetime import datetime, timedelta
from typing import Dict

from decouple import config
from jwt import ExpiredSignatureError, InvalidTokenError, decode, encode

# Obtener la clave secreta de JWT (podría venir de una variable de entorno)
JWT_SECRET = config("JWT_SECRET")

if not JWT_SECRET:
    raise ValueError("La clave secreta de JWT no está configurada")


# Función para generar el token
def generate_token(payload: Dict, expires_in: timedelta) -> str:
    # Calcular la fecha de expiración
    expiration = datetime.now() + expires_in
    payload["exp"] = expiration  # Agregar la expiración al payload

    # Generar el token con expiración
    token = encode(payload, JWT_SECRET, algorithm="HS256")
    return token


# Función para verificar el token
def verify_token(token: str):
    try:
        user = decode(token, JWT_SECRET, algorithms=["HS256"])
        return user
    except ExpiredSignatureError:
        raise Exception("El token ha expirado")
    except InvalidTokenError:
        raise Exception("Token inválido")
