# note_service.py
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy.orm import Session
from models import CustomNote  # Make sure you define this model
from embeddings import fetch_embedding
import json

def add_note(
    db: Session,
    customer_id: UUID,
    author: str,
    category: str,
    summary: str,
    full_note: str,
    tags: str,
    source: str,
    timestamp: datetime = None
):
    note_id = uuid4()
    timestamp = timestamp or datetime.utcnow()

    # Generate embedding from full_note (or summary if preferred)
    raw_embedding = fetch_embedding(full_note)
    embedding_value = json.loads(raw_embedding) if isinstance(raw_embedding, str) else raw_embedding

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
        embedding=embedding_value
    )

    db.add(note)
    db.commit()
    return {"status": "note created", "note_id": str(note_id)}
