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

def fetch_embedding(text: str) -> str:
    try:
        payload = {
            "inputText": text,
            "dimensions": 1024,
            "normalize": True
        }
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )
        body = response['body'].read().decode()
        result = json.loads(body)
        return f"[{','.join(map(str, result.get('embedding')))}]"
    except Exception as e:
        logging.exception("Failed to fetch embedding")
        raise HTTPException(status_code=500, detail={
            "error": "Embedding generation failed",
            "exception": str(e)
        })
