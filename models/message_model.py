from pydantic import BaseModel

from pydantic import BaseModel
from typing import Optional

class AddMessageModel(BaseModel):
    conv_id: str
    content: str
