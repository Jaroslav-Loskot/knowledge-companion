# note_service.py

from uuid import uuid4, UUID
from datetime import datetime
from sqlalchemy.orm import Session
from models import CustomNote
import json
import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Initialize Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

def summarize_note(note_text: str) -> str:
    prompt = (
        "You are a helpful assistant. Please summarize the following meeting note in 1–2 sentences:\n\n"
        f"{note_text}\n"
    )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"].strip()
    except Exception as e:
        raise RuntimeError(f"Summarization failed: {str(e)}")

def generate_embedding(text: str) -> list:
    payload = {
        "inputText": text,
        "dimensions": 1024,
        "normalize": True
    }

    try:
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps(payload),
            contentType="application/json"
        )
        response_body = json.loads(response["body"].read())
        return response_body.get("embedding", [])
    except Exception as e:
        raise RuntimeError(f"Embedding generation failed: {str(e)}")

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
    embedding = generate_embedding(summary)

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
