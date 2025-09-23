from pydantic import BaseModel

class CreateUserModel(BaseModel):
    username: str
    password: str