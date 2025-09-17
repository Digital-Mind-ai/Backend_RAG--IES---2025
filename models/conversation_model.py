from pydantic import BaseModel
from typing import List, Optional


class CreateConversationModel(BaseModel):
    firstMessage: str


class FileInfoModel(BaseModel):
    """Modelo para información de archivos procesados"""
    filename: str
    content_type: Optional[str] = None
    size: int
    success: bool
    errors: Optional[List[str]] = None
    text_preview: Optional[str] = None
    is_image: Optional[bool] = False


class FileUploadResponseModel(BaseModel):
    """Modelo para la respuesta de subida de archivos"""
    conversation_id: str
    total_files: int
    successful: int
    failed: int
    files: List[FileInfoModel]
