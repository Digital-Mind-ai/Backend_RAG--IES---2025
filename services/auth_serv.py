from src.database import Therapist


def get_auth_therapist_serv(cedula: str):
    try:
        therapist = Therapist.select().where(Therapist.cedulaT == cedula)

        return therapist.dicts().first()
    except Exception as error:
        raise error
