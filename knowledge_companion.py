from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

from models import Base, Customer, CustomerAlias
from embeddings import fetch_embedding

# Load AWS credentials from .env
load_dotenv()

# --- DB SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)
metadata = MetaData()
metadata.reflect(bind=engine)

# --- SCHEMAS ---
class CustomerAliasCreate(BaseModel):
    alias: str
    embedding: Optional[str]

class CustomerCreate(BaseModel):
    id: UUID
    name: str
    industry: Optional[str]
    size: Optional[str]
    region: Optional[str]
    status: Optional[str]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    jira_project_key: Optional[str]
    salesforce_account_id: Optional[str]
    mainpage_url: Optional[str]
    aliases: Optional[List[CustomerAliasCreate]] = []

# --- FASTAPI APP ---
app = FastAPI(
    title="Knowledge Companion Service",
    description="A microservice for managing customer identities, aliases, and embeddings, supporting AI agents and RAG-based systems in an integrated Information Hub.",
    version="1.0.0"
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/customers")
def create_customer(customer: CustomerCreate):
    db = next(get_db())
    db_customer = Customer(**customer.dict(exclude={"aliases"}))
    db.add(db_customer)
    if customer.aliases:
        for alias in customer.aliases:
            embedding_value = alias.embedding or fetch_embedding(alias.alias)
            db_alias = CustomerAlias(
                customer_id=customer.id,
                alias=alias.alias,
                embedding=embedding_value
            )
            db.add(db_alias)
    db.commit()
    return {"status": "created", "customer_id": str(customer.id)}

@app.delete("/customers/{customer_id}")
def delete_customer(customer_id: UUID):
    db = next(get_db())
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    db.delete(customer)
    db.commit()
    return {"status": "deleted"}

@app.get("/customers/{customer_id}")
def get_customer(customer_id: UUID):
    db = next(get_db())
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    aliases = [alias.alias for alias in customer.aliases]
    return {"id": str(customer.id), "name": customer.name, "aliases": aliases}

@app.get("/schema")
def get_schema():
    schema_info = []
    for table_name, table in metadata.tables.items():
        schema_info.append({
            "table": table_name,
            "columns": [col.name for col in table.columns]
        })
    return {"schema": schema_info}
