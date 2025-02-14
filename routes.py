from fastapi import APIRouter
from app.models import EventRequest, EventResponse
from app.services import classify_parts_with_gpt

router = APIRouter()

@router.post("/events", response_model=EventResponse)
async def classify_event(event: EventRequest):
    """
    Processes a chat log and returns classified parts for each user message.
    """
    classified_messages = classify_parts_with_gpt(event.chat_log)
    return classified_messages
