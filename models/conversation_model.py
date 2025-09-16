from pydantic import BaseModel


class CreateConversationModel(BaseModel):
    title: str
