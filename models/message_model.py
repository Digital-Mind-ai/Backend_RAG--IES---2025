from pydantic import BaseModel


class AddMessageModel(BaseModel):
    conv_id: str
    content: str
