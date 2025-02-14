import json
import uuid
from fastapi import HTTPException
from openai import OpenAI
import os
from app.config import load_env
from typing import List
from app.models import Message, EventResponse, ClassifiedMessage ,ClassifiedPart


# Load environment variables
load_env()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OpenAI API Key. Set OPENAI_API_KEY in .env.")

client = OpenAI(api_key=api_key)

# Store classified events
event_store = {}

def classify_parts_with_gpt(chat_log: List[Message]):
    """
    Processes user messages and classifies each message into IFS parts.
    """

    system_prompt = """You are an Internal Family Systems (IFS) therapy assistant. 
    Your job is to classify **each user message** (but not assistant messages) into psychological 'parts' 
    (such as Exile, Firefighter, and Manager).
    
    - Each user message can have **multiple parts**, with different emotions and needs and percentage confidence scores.
    - Ignore assistant messages in classification, but **use them for context**.
    
    Output a structured **JSON object** where each user message is mapped to its classified parts.
    """

    chat_text = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_log])

    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chat_text}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "part_classification",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "classified_messages": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "message_id": {"type": "string"},
                                        "content": {"type": "string"},
                                        "parts": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "part_id": {"type": "string"},
                                                    "label": {"type": "string"},
                                                    "confidence": {"type": "number"},
                                                    "emotion": {"type": "string"},
                                                    "description": {"type": "string"}
                                                },
                                                "required": ["part_id", "label", "confidence", "emotion", "description"],
                                                "additionalProperties": False
                                            }
                                        }
                                    },
                                    "required": ["message_id", "content", "parts"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["classified_messages"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )

        # ✅ Convert response from OpenAI into a Python dictionary
        classified_messages = json.loads(response.choices[0].message.content)
        
        # ✅ Assign unique IDs to each message and part
        for message in classified_messages["classified_messages"]:
            message["message_id"] = str(uuid.uuid4())
            for part in message["parts"]:
                part["part_id"] = str(uuid.uuid4())

        # ✅ Store event
        event_id = str(uuid.uuid4())
        event_store[event_id] = classified_messages

        return EventResponse(
            event_id=event_id,
            classified_messages=[
                ClassifiedMessage(
                    message_id=message["message_id"],
                    content=message["content"],
                    parts=[
                        ClassifiedPart(
                            part_id=part["part_id"],
                            label=part["label"],
                            confidence=part["confidence"],
                            emotion=part["emotion"],
                            description=part["description"],
                        )
                        for part in message["parts"]
                    ]
                ) for message in classified_messages["classified_messages"]
            ]
        )


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing GPT response: {str(e)}")
