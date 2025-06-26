from uuid import uuid4, UUID
from datetime import datetime
from sqlalchemy.orm import Session
from models import CustomNote
import json
from utils.bedrock_wrapper import call_claude, fetch_embedding  


def summarize_note(note_text: str) -> str:
    """
    Summarizes meeting notes using Claude Sonnet 4.
    """
    system_prompt = (
        "You are a helpful assistant that summarizes meeting notes into 1–2 sentences."
    )
    return call_claude(system_prompt, note_text)


def add_note(
    db: Session,
    customer_id: UUID,
    author: str,
    category: str,
    full_note: dict,
    tags: list,
    source: str,
    timestamp: datetime = None
):
    note_id = uuid4()
    timestamp = timestamp or datetime.utcnow()

    full_note_json = json.dumps(full_note)
    summary = summarize_note(full_note_json)
    embedding = fetch_embedding(summary)

    note = CustomNote(
        id=note_id,
        customer_id=customer_id,
        author=author,
        timestamp=timestamp,
        category=category,
        summary=summary,
        full_note=full_note,
        tags=tags,
        source=source,
        embedding=embedding
    )

    db.add(note)
    db.commit()
    return {"status": "note created", "note_id": str(note_id)}
