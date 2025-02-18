from pydantic import BaseModel
from typing import List, Dict

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str  # The message text

class ClassifiedPart(BaseModel):
    part_id: str
    label: str
    confidence: float
    emotion: str
    description: str

class ClassifiedMessage(BaseModel):
    message_id: str
    content: str
    parts: List[ClassifiedPart]  # Multiple parts per message

class EventRequest(BaseModel):
    event_id: str
    chat_log: List[Message]  # Full conversation (user & assistant)
    metadata: Dict[str, str] = {}

class EventResponse(BaseModel):
    event_id: str
    classified_messages: List[ClassifiedMessage]  # Groups messages with parts

class TextInput(BaseModel):
    text: str