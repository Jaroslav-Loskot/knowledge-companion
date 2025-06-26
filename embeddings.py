import boto3
import json
import os
from fastapi import HTTPException
from dotenv import load_dotenv
import logging

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="eu-north-1",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def fetch_embedding(text: str) -> list[float]:
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty.")

    try:
        payload = {
            "inputText": text
        }

        response = bedrock_client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )
        body = response['body'].read().decode()
        result = json.loads(body)
        return result.get('embedding')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


