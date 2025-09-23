from pydantic import BaseModel
from typing import List, Optional


class AttachFileModel(BaseModel):
    name: str
    url: str
    type: str

from pydantic import BaseModel
from typing import Optional

class AddMessageModel(BaseModel):
    conv_id: str
    content: str
    attachments: Optional[List[AttachFileModel]] = None


class MessageResponseModel(BaseModel):
    id: str
    content: str
    sender: str  # 'user' | 'assistant' | 'tool'
    attachments: Optional[List[AttachFileModel]] = None
    timestamp: int
