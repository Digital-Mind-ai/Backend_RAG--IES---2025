from fastapi import HTTPException, Request, status

from utils.jwt_handle import verify_token


# Función para verificar el token y roles
def session_validator(request: Request):
    try:
        # Obtener el token desde los encabezados Authorization
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            # Si no hay token, se lanza una excepción HTTP con un mensaje de error
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "status": 401,
                    "message": "No tiene autorización",
                },
            )

        token = auth_header.split(" ").pop()  # Extraer el token

        # Verificar el token y obtener el payload
        user = verify_token(token)

        # Guardar el payload del usuario en la request para uso posterior en los controladores
        # Acceso: request.state.user
        request.state.user = user
        
        print(f"Usuario autenticado: {user.get('username')}")

        return user

    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail={
                "status": 401,
                "message": "Error de autorización",
                "error": str(e),
            },
        )
