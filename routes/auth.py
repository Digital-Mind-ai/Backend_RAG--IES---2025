from datetime import timedelta

from fastapi import APIRouter

from models.auth_model import AuthModel
from models.user_model import CreateUserModel
from services.auth_serv import get_auth_user_serv, create_user_serv

from utils.bcrypt_handle import verified
from utils.error_handle import get_details_error
from utils.handle_respose import send_success_response
from utils.jwt_handle import generate_token

auth_router = APIRouter()


@auth_router.post("/login")
def login_users_ctrl(data: AuthModel):
    try:
        user = get_auth_user_serv(data.username)

        if not user:
            return send_success_response(400, "Revise sus credenciales")

        # if not verified(data.password, user["password"]):
            # return send_success_response(401, "Contraseña incorrecta")
            
        if data.password != user["password"]:
            return send_success_response(400, "Revise sus credenciales")

        token = generate_token(user, expires_in=timedelta(minutes=960))

        return send_success_response(
            200,
            "Login exitoso",
            {
                "token": token,
                "user": {
                    "username": user.get("username"),
                },
            },
        )
    except Exception as error:
        return get_details_error(error)

@auth_router.post("/register")
def create_user_ctrl(data: CreateUserModel):
    try:
        user = create_user_serv(data.username, data.password)
        print(f"Usuario creado: {user.username}")
        return send_success_response(201, "Usuario creado")
    except Exception as error:
        return get_details_error(error)

@auth_router.get("/logout")
def logout_therapists_ctrl():
    return send_success_response(200, "Session terminada")
