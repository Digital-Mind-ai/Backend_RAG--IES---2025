from datetime import timedelta

from fastapi import APIRouter

from models.auth_model import AuthModel
from models.therapist_model import TherapistModel
from services.auth_serv import get_auth_therapist_serv
from services.therapist_serv import create_therapist_serv
from utils.bcrypt_handle import verified
from utils.error_handle import get_details_error
from utils.handle_respose import send_success_response
from utils.jwt_handle import generate_token

auth_router = APIRouter()


@auth_router.post("/login")
def login_therapists_ctrl(data: AuthModel):
    try:
        therapist = get_auth_therapist_serv(data.cedulaT)

        if not therapist:
            return send_success_response(404, "Usuario no encontrado")

        if not verified(data.password, therapist["password"]):
            return send_success_response(401, "Contraseña incorrecta")

        token = generate_token(therapist, expires_in=timedelta(minutes=960))

        return send_success_response(
            200,
            "Usuario logueado",
            {
                "token": token,
                "therapist": {
                    "cedulaT": therapist.get("cedulaT"),
                    "name": therapist.get("name"),
                    "lastname": therapist.get("lastname"),
                    "email": therapist.get("email"),
                    "phone": therapist.get("phone"),
                },
            },
        )
    except Exception as error:
        return get_details_error(error)


@auth_router.post("/register")
def create_therapists_ctrl(data: TherapistModel):
    try:
        therapist = create_therapist_serv(data)
        print(f"Usuario creado: {therapist.name}")
        return send_success_response(201, "Usuario creado")
    except Exception as error:
        return get_details_error(error)


@auth_router.get("/logout")
def logout_therapists_ctrl():
    return send_success_response(200, "Session terminada")
