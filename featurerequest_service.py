from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from models import FeatureRequest
from utils.bedrock_wrapper import call_claude, fetch_embedding


def summarize_feature_request(text: str) -> str:
    """
    Summarizes a feature request description using Claude Sonnet.
    """
    system_prompt = (
        "You are a helpful assistant summarizing feature requests into 1–2 sentences "
        "to help developers and managers understand the main goal quickly."
    )
    return call_claude(system_prompt, text)


def add_feature_request(
    db: Session,
    customer_id: UUID,
    request_title: str,
    description: str,
    priority: str,
    status: str,
    estimated_delivery: datetime,
    internal_notes: str,
):
    # your implementation here
    request_id = uuid4()
    created_at = datetime.utcnow()

    summary = summarize_feature_request(description)
    embedding = fetch_embedding(summary)

    request = FeatureRequest(
        id=request_id,
        customer_id=customer_id,
        request_title=request_title,
        description=description,
        priority=priority,
        status=status,
        estimated_delivery=estimated_delivery,
        internal_notes=internal_notes,
        created_at=created_at,
        updated_at=created_at,
        summary=summary,
        embedding=embedding,
    )

    db.add(request)
    db.commit()
    return {"status": "feature request created", "request_id": str(request_id)}
