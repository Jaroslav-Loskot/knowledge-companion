import os
import json
import requests
import logging
from dotenv import load_dotenv
from fastapi import HTTPException
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
import boto3

# Load env variables
load_dotenv(override=True)

# Claude Bedrock config
AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
CLAUDE_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "eu.anthropic.claude-sonnet-4-20250514-v1:0")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# --- Claude Generation via signed HTTP request ---
def call_claude(system_prompt: str, user_input: str, max_tokens: int = 1024) -> str:
    """
    Call Claude Sonnet 4 via Bedrock provisioned model and return assistant's response.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [{"type": "text", "text": user_input}]
        }
    ]

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 250,
        "messages": messages
    }

    url = f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com/model/{CLAUDE_MODEL_ID}/invoke"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    creds = Credentials(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    aws_req = AWSRequest(method="POST", url=url, data=json.dumps(payload), headers=headers)
    SigV4Auth(creds, "bedrock", AWS_REGION).add_auth(aws_req)
    signed_headers = dict(aws_req.headers.items())

    response = requests.post(url, headers=signed_headers, data=json.dumps(payload))

    if response.ok:
        response_data = response.json()
        try:
            return response_data["content"][0]["text"].strip()
        except (KeyError, IndexError):
            raise RuntimeError("Unexpected response format from Claude.")
    else:
        raise RuntimeError(f"Claude request failed ({response.status_code}): {response.text}")


# --- Titan Embedding ---
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def fetch_embedding(text: str) -> list[float]:
    """
    Fetch embedding using Amazon Titan model.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty.")

    try:
        payload = {"inputText": text}
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
        logging.error(f"Embedding generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")
