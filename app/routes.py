from fastapi import APIRouter
from app.models import EventRequest, EventResponse , TextInput
from app.services import classify_parts_with_gpt , classify_parts_from_text

router = APIRouter()

@router.post("/events", response_model=EventResponse)
async def classify_event(event: EventRequest):
    """
    Processes a chat log and returns classified parts for each user message.
    """
    classified_messages = classify_parts_with_gpt(event.chat_log)
    return classified_messages

@router.post("/blob", response_model=EventResponse)
async def classify_event2(input_data: TextInput):
    """
    Processes a user-provided text blob and returns classified parts.
    """
    classified_messages = classify_parts_from_text(input_data.text)
    return classified_messages

@router.get("/")
async def root():
    return {"message": "service is working"}
