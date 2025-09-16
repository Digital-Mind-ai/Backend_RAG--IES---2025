from bcrypt import checkpw, gensalt, hashpw


# Función para cifrar la contraseña
def encrypt(password: str) -> str:
    # Generamos el hash de la contraseña
    password_hashed = hashpw(password.encode("utf-8"), gensalt())
    return password_hashed.decode("utf-8")


# Función para verificar si la contraseña coincide con el hash
def verified(password: str, password_hash: str) -> bool:
    # Verificamos si la contraseña coincide con el hash almacenado
    is_valid = checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    return is_valid
