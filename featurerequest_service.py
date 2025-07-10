from datetime import datetime
from uuid import UUID, uuid4
import json
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from models import FeatureRequest, Customer
from schemas import (
    FeatureRequestUpdatePayload,
    FeatureRequestOperationRequest,
    FeatureRequestFromRaw,
)
from utils.bedrock_wrapper import call_claude, fetch_embedding


def summarize_feature_request(text: str) -> dict:
    """
    Summarizes a feature request description using Claude Sonnet and returns
    a dict with keys 'title' and 'summary'.
    """
    system_prompt = (
        """
        You are a helpful assistant summarizing software feature requests.

        Based on the provided input:
        1. Generate a clear and concise TITLE — it must be 80 characters or fewer.
        2. Then generate a longer SUMMARY explaining the feature in more detail (1–3 paragraphs).

        Return only the JSON object in this format:
        {
        "title": "<title here>",
        "summary": "<summary here>"
        }
        """
    )
    raw_response = call_claude(system_prompt, text)

    try:
        # 🔽 Strip markdown code block formatting if present
        if raw_response.strip().startswith("```"):
            raw_response = raw_response.strip().strip("`").strip("json").strip()
        
        response_json = json.loads(raw_response)
        return {
            "title": response_json["title"],
            "summary": response_json["summary"]
        }
    except Exception as e:
        raise ValueError(f"Failed to parse Claude response: {e}\nRaw: {raw_response}")



def add_feature_request_from_raw(
    db: Session,
    customer_id: UUID,
    raw_input: str,
    priority: str = "unspecified",
    status: str = "new",
):
    # ✅ Step 1: Check if customer exists
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(
            status_code=404,
            detail=f"Customer ID {customer_id} does not exist. Please create the customer first.",
        )

    # ✅ Step 2: Prepare content
    request_id = uuid4()
    created_at = datetime.utcnow()
    summary_data = summarize_feature_request(raw_input)
    request_title = summary_data["title"]
    summary = summary_data["summary"]
    embedding = fetch_embedding(summary)

    # ✅ Step 3: Save to DB
    request = FeatureRequest(
        id=request_id,
        customer_id=customer_id,
        request_title=request_title,
        summary=summary,
        priority=priority,
        status=status,
        created_at=created_at,
        raw_input=raw_input,
        embedding=embedding,
    )
    try:
        db.add(request)
        db.commit()
        return {"status": "feature request created", "request_id": str(request_id)}
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )


def update_feature_request(db: Session, update: FeatureRequestUpdatePayload):
    request = db.query(FeatureRequest).filter(FeatureRequest.id == update.request_id).first()
    if not request:
        raise HTTPException(status_code=404, detail="Feature request not found")

    if update.raw_input:
        request.raw_input = update.raw_input
        summary_data = summarize_feature_request(update.raw_input)
        request.request_title = summary_data["title"]
        request.summary = summary_data["summary"]
        request.embedding = fetch_embedding(request.summary)

    if update.priority:
        request.priority = update.priority
    if update.status:
        request.status = update.status

    db.commit()
    return {"status": "feature request updated", "request_id": str(request.id)}


def delete_feature_request(db: Session, request_id: UUID):
    request = db.query(FeatureRequest).filter(FeatureRequest.id == request_id).first()
    if not request:
        raise HTTPException(status_code=404, detail="Feature request not found")

    db.delete(request)
    db.commit()
    return {"status": "feature request deleted", "request_id": str(request.id)}


def handle_feature_request_operation(db: Session, payload: FeatureRequestOperationRequest):
    if payload.operation == "add":
        raw = FeatureRequestFromRaw(**payload.payload)
        return add_feature_request_from_raw(
            db=db,
            customer_id=raw.customer_id,
            raw_input=raw.raw_input,
            priority=raw.priority,
            status=raw.status,
        )

    elif payload.operation == "update":
        update = FeatureRequestUpdatePayload(**payload.payload)
        return update_feature_request(db, update)

    elif payload.operation == "delete":
        request_id = UUID(payload.payload.get("request_id"))
        return delete_feature_request(db, request_id)

    else:
        raise HTTPException(status_code=400, detail="Invalid feature request operation")