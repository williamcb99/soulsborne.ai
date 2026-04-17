from pydantic import BaseModel
from classes.Message import Message

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 0.7
    stream: bool = True