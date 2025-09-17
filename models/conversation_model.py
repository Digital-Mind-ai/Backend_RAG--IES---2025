from pydantic import BaseModel


class CreateConversationModel(BaseModel):
    firstMessage: str

class UploadFileModel(BaseModel):
    conv_id: str
    fileName: str
    fileType: str
    file: str  # Base64 encoded file content