from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID, uuid4
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
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)
metadata = MetaData()
metadata.reflect(bind=engine)

# --- SCHEMAS ---
class CustomerAliasCreate(BaseModel):
    alias: str
    embedding: Optional[str] = None

class CustomerCreate(BaseModel):
    id: Optional[UUID] = None
    name: str
    industry: Optional[str] = None
    size: Optional[str] = None
    region: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    jira_project_key: Optional[str] = None
    salesforce_account_id: Optional[str] = None
    mainpage_url: Optional[str] = None
    aliases: Optional[List[CustomerAliasCreate]] = []

class AddAliasRequest(BaseModel):
    alias: str

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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/customers")
def create_customer(customer: CustomerCreate):
    db = next(get_db())
    customer_id = customer.id or uuid4()
    db_customer = Customer(id=customer_id, **customer.dict(exclude={"id", "aliases"}))
    db.add(db_customer)

    aliases = customer.aliases or [CustomerAliasCreate(alias=customer.name)]

    for alias in aliases:
        embedding_value = alias.embedding or fetch_embedding(alias.alias)
        db_alias = CustomerAlias(
            customer_id=customer_id,
            alias=alias.alias,
            embedding=embedding_value
        )
        db.add(db_alias)

    db.commit()
    return {"status": "created", "customer_id": str(customer_id)}

@app.post("/customers/{customer_id}/aliases")
def add_alias(customer_id: UUID, alias_data: AddAliasRequest):
    db = next(get_db())
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    embedding_value = fetch_embedding(alias_data.alias)
    db_alias = CustomerAlias(
        customer_id=customer_id,
        alias=alias_data.alias,
        embedding=embedding_value
    )
    db.add(db_alias)
    db.commit()
    return {"status": "alias added", "customer_id": str(customer_id), "alias": alias_data.alias}

@app.delete("/customers/{customer_id}")
def delete_customer(customer_id: UUID):
    db = next(get_db())
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    db.delete(customer)
    db.commit()
    return {"status": "deleted"}

@app.get("/customers")
def get_customer(id: Optional[UUID] = Query(None), name: Optional[str] = Query(None)):
    db = next(get_db())
    query = db.query(Customer)
    if id:
        query = query.filter(Customer.id == id)
    if name:
        query = query.filter(Customer.name.ilike(f"%{name}%"))
    customers = query.all()

    if not customers:
        raise HTTPException(status_code=404, detail="Customer not found")

    result = []
    for customer in customers:
        aliases = [alias.alias for alias in customer.aliases]
        result.append({"id": str(customer.id), "name": customer.name, "aliases": aliases})

    return result

@app.get("/schema")
def get_schema():
    schema_info = []
    for table_name, table in metadata.tables.items():
        schema_info.append({
            "table": table_name,
            "columns": [col.name for col in table.columns]
        })
    return {"schema": schema_info}
