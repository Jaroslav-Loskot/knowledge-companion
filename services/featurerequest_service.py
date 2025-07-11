from uuid import UUID, uuid4
import json
from datetime import datetime
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from models import FeatureRequest, Customer
from utils.bedrock_wrapper import call_claude, fetch_embedding
from schemas import (
    FeatureRequestUpdatePayload,
    FeatureRequestOperationRequest,
    FeatureRequestFromRaw,
    OperationStatus,
)

def summarize_feature_request(text: str) -> dict:
    system_prompt = """
        You are a helpful assistant summarizing software feature requests.

        Based on the provided input:
        1. Generate a clear and concise TITLE — it must be 80 characters or fewer.
        2. Then generate a longer SUMMARY explaining the feature in more detail (1–3 paragraphs).

        Return only the JSON object in this format — do NOT wrap it in markdown or add commentary:

        {
        "title": "<title here>",
        "summary": "<summary here>"
        }
    """
    raw_response = call_claude(system_prompt, text)

    try:
        if raw_response.strip().startswith("```"):
            raw_response = raw_response.strip().strip("`").strip("json").strip()
        parsed = json.loads(raw_response)
        return {"title": parsed["title"], "summary": parsed["summary"]}
    except Exception as e:
        raise ValueError(f"Failed to parse Claude response: {e}\nRaw: {raw_response}")


def add_feature_request_from_raw(
    db: Session,
    customer_ids: list[UUID],
    raw_input: str,
    priority: str = "unspecified",
    status: str = "new",
) -> OperationStatus:
    # Fetch all customers found in the DB
    customers = db.query(Customer).filter(Customer.id.in_(customer_ids)).all()

    # Determine which ones are missing
    found_ids = {c.id for c in customers}
    missing_ids = [str(cid) for cid in customer_ids if cid not in found_ids]

    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Some customer IDs were not found.",
                "missing_customer_ids": missing_ids
            }
        )

    request_id = uuid4()
    created_at = datetime.utcnow()
    summary_data = summarize_feature_request(raw_input)

    request = FeatureRequest(
        id=request_id,
        request_title=summary_data["title"],
        summary=summary_data["summary"],
        priority=priority,
        status=status,
        created_at=created_at,
        raw_input=raw_input,
        embedding=fetch_embedding(summary_data["summary"]),
        customers=customers,
    )

    try:
        db.add(request)
        db.commit()
        return OperationStatus(status="created", entity="feature_request", id=str(request_id))
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")



def update_feature_request(db: Session, update: FeatureRequestUpdatePayload) -> OperationStatus:
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

    if update.customer_ids is not None:
        customers = db.query(Customer).filter(Customer.id.in_(update.customer_ids)).all()
        found_ids = {c.id for c in customers}
        missing_ids = [str(cid) for cid in update.customer_ids if cid not in found_ids]

        if missing_ids:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "Some customer IDs were not found.",
                    "missing_customer_ids": missing_ids
                }
            )

        request.customers = customers  # full overwrite

    db.commit()
    return OperationStatus(status="updated", entity="feature_request", id=str(request.id))



def delete_feature_request(db: Session, request_id: UUID) -> OperationStatus:
    request = db.query(FeatureRequest).filter(FeatureRequest.id == request_id).first()
    if not request:
        raise HTTPException(status_code=404, detail="Feature request not found")

    db.delete(request)
    db.commit()
    return OperationStatus(status="deleted", entity="feature_request", id=str(request.id))


def handle_feature_request_operation(db: Session, payload: FeatureRequestOperationRequest):
    if payload.operation == "add":
        raw = FeatureRequestFromRaw(**payload.payload)
        return add_feature_request_from_raw(
            db=db,
            customer_ids=raw.customer_ids,  # now list
            raw_input=raw.raw_input,
            priority=raw.priority,
            status=raw.status,
        )
    elif payload.operation == "update":
        return update_feature_request(db, FeatureRequestUpdatePayload(**payload.payload))
    elif payload.operation == "delete":
        request_id = UUID(payload.payload.get("request_id"))
        return delete_feature_request(db, request_id)
    else:
        raise HTTPException(status_code=400, detail="Invalid feature request operation")
