from fastapi.responses import JSONResponse


def send_error_response(status: int, message: str = "", error: any = None):
    return JSONResponse(
        status_code=status,
        content={
            "status": status,
            "message": message,
            "error": error,
        },
    )


def send_success_response(status: int, message: str = "", data: any = None):
    return JSONResponse(
        status_code=status,
        content={
            "status": status,
            "message": message,
            "data": data,
        },
    )
