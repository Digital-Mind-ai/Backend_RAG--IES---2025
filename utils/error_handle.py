from peewee import DataError, DoesNotExist, IntegrityError, OperationalError

from utils.handle_respose import send_error_response


def get_details_error(error: Exception):

    if isinstance(error, DataError):
        return send_error_response(400, "Valor es demasiado largo", str(error))

    elif isinstance(error, IntegrityError):
        return send_error_response(409, "El usuario ya existe", str(error))

    elif isinstance(error, OperationalError):
        return send_error_response(
            400, "Error de clave foránea o problema operativo", str(error)
        )

    elif isinstance(error, DoesNotExist):
        return send_error_response(404, "Datos no encontrados", str(error))

    else:
        return send_error_response(500, "Error interno del servidor", str(error))
