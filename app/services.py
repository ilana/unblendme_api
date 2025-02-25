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
    Processes a single user-provided text blob and classifies its psychological parts.
    """
    system_prompt = """You are an Internal Family Systems (IFS) therapy assistant.
Your job is to classify each user message into psychological 'parts' (such as exile, firefighter, manager, and self). 

### Classification Requirements:
- Each user message may contain **multiple parts**, each with:
  1. **Label**: One of ["Exile", "Firefighter", "Manager", "Self"].
  2. **Blended Amount**: A **float between 0.0 and 0.99**, representing how strongly the user is identifying with the part, confidence level.
  3. **Emotion**: A lowercase, **single-word** emotion from the **mood meter list** (see below) or `"none"` if none apply.
  4. **Quadrant**: One of **["red", "blue", "green", "yellow"]** based on the emotion or `"none"` if none apply.
  5. **Description**: A brief explanation of the part's motivation.

### Mood Quadrant Classification:
Each emotion **must** be assigned to a **quadrant** based on energy and valence:

- **Red Quadrant (high energy, unpleasant)**: ["enraged", "panicked", "stressed", "jittery", "shocked", "livid", "furious", "frustrated", "tense", "fuming", "frightened", "angry", "nervous", "restless", "anxious", "apprehensive", "worried", "irritated", "annoyed"]
- **Yellow Quadrant (high energy, pleasant)**: ["surprised", "upbeat", "festive", "exhilarated", "ecstatic", "hyper", "cheerful", "motivated", "inspired", "elated", "energized", "lively", "excited", "optimistic", "enthusiastic", "pleased", "focused", "happy", "proud", "thrilled", "joyful", "hopeful", "playful", "blissful"]
- **Blue Quadrant (low energy, unpleasant)**: ["repulsed", "troubled", "concerned", "uneasy", "peeved", "disgusted", "glum", "disappointed", "down", "apathetic", "pessimistic", "morose", "discouraged", "sad", "bored", "alienated", "miserable", "lonely", "disheartened", "tired", "despondent", "depressed", "sullen", "exhausted", "fatigued", "despairing", "hopeless", "desolate", "spent", "drained"]
- **Green Quadrant (low energy, pleasant)**: ["at ease", "easygoing", "content", "loving", "fulfilled", "calm", "secure", "satisfied", "grateful", "touched", "relaxed", "chill", "restful", "blessed", "balanced", "mellow", "thoughtful", "peaceful", "comfortable", "carefree", "sleepy", "complacent", "tranquil", "cozy", "serene"]
- **Neutral (if no emotion applies)**: `"neutral"`

### Additional Rules:
- **Ignore assistant messages for classification but use them for context.**
- **Always return structured JSON with valid fields.**
"""

    chat_text = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_log])

    try:
        response = client.chat.completions.create(
            model="o1",
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
                                                    "blended": {"type": "number"},
                                                    "emotion": {"type": "string"},
                                                    "quadrant": {"type": "string"},
                                                    "description": {"type": "string"}
                                                },
                                                "required": ["part_id", "label", "blended", "emotion", "quadrant","description"],
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
        print(response.choices[0].message.content)
        # ✅ Convert response from OpenAI into a Python dictionary
        classified_messages = json.loads(response.choices[0].message.content)

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
                            blended=part["blended"],
                            emotion=part["emotion"],
                            quadrent=part["quadrant"],
                            description=part["description"],
                        )
                        for part in message["parts"]
                    ]
                ) for message in classified_messages["classified_messages"]
            ]
        )


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing GPT response: {str(e)}")

def classify_parts_from_text(user_text: str):
    """
    Processes a single user-provided text blob and classifies its psychological parts.
    """
    system_prompt = """You are an Internal Family Systems (IFS) therapy assistant.
    Your job is to classify the given text into psychological 'parts' (such as Exile, Firefighter, and Manager).
    
    - The text may contain multiple 'parts' with different emotions, needs and confidence levels, reffered to as blended amount as a float. 
    - emotions can be in the four quandrents of mood meter (
    Enraged,Panicked,Stressed,Jittery,Shocked,Surprised,Upbeat,Festive,Exhilarated,Ecstatic,
    Livid,Furious,Frustrated,Tense,Stunned,Hyper,Cheerful,Motivated,Inspired,Elated,
    Fuming,Frightened,Angry,Nervous,Restless,Energized,Lively,Excited,Optimistic,Enthusiastic,
    Anxious,Apprehensive,Worried,Irritated,Annoyed,Pleased,Focused,Happy,Proud,Thrilled,
    Repulsed,Troubled,Concerned,Uneasy,Peeved,Pleasant,Joyful,Hopeful,Playful,Blissful
    Disgusted,Glum,Disappointed,Down,Apathetic,At Ease,Easygoing,Content,Loving,Fulfilled
    Pessimistic,Morose,Discouraged,Sad,Bored,Calm,Secure,Satisfied,Grateful,Touched,
    Alienated,Miserable,Lonely,Disheartened,Tired,Relaxed,Chill,Restful,Blessed,Balanced,
    Despondent,Depressed,Sullen,Exhausted,Fatigued,Mellow,Thoughtful,Peaceful,Comfortable,Carefree,
    Despairing,Hopeless,Desolate,Spent,Drained,Sleepy,Complacent,Tranquil,Cozy,Serene)
    - Ignore assistant messages in classification, but **use them for context**.
    - Return a structured **JSON object** mapping the identified parts within the text.

    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "part_classification",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "classified_parts": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "part_id": {"type": "string"},
                                        "label": {"type": "string"},
                                        "blended": {"type": "number"},
                                        "emotion": {"type": "string"},
                                        "quadrant": {"type": "string"},
                                        "description": {"type": "string"}
                                    },
                                    "required": ["part_id", "label", "blended", "emotion", "quadrant", "description"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["classified_parts"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )

        # ✅ Convert response from OpenAI into a Python dictionary
        classified_parts = json.loads(response.choices[0].message.content)
        
        # ✅ Assign unique IDs to each part
        for part in classified_parts["classified_parts"]:
            part["part_id"] = str(uuid.uuid4())

        # ✅ Store event
        event_id = str(uuid.uuid4())
        event_store[event_id] = classified_parts

        return EventResponse(
            event_id=event_id,
            classified_messages=[
                ClassifiedMessage(
                    message_id=str(uuid.uuid4()),
                    content=user_text,
                    parts=[
                        ClassifiedPart(
                            part_id=part["part_id"],
                            label=part["label"],
                            blended=part["blended"],
                            emotion=part["emotion"],
                            quadrant=part["quadrant"],
                            description=part["description"]
                        ) for part in classified_parts["classified_parts"]
                    ]
                )
            ]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing GPT response: {str(e)}")
