import os
from uuid import UUID, uuid4
from datetime import datetime
from typing import List, Optional, Literal
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from psycopg2.errors import ForeignKeyViolation
from fastapi import HTTPException

from models import (
    Base,
    Customer,
    CustomerAlias,
    CustomNote,
    Contact
)
from utils.bedrock_wrapper import fetch_embedding
from utils.search import apply_dynamic_filters, SearchFiltegitr
from contact_service import add_contact, search_contacts
from note_service import add_note
from featurerequest_service import add_feature_request
from task_service import add_task
from schemas import (
    ContactCreate,
    ContactSearchRequest,
    ContactSearchFilter,
    FeatureRequestCreate,
    CustomerAliasCreate,
    CustomerCreate,
    AliasOperationRequest,
    CustomerUpdateRequest,
    NoteCreateRequest,
    TaskCreate
)


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


@app.post("/tasks")
def create_task(payload: TaskCreate):
    db = next(get_db())
    try:
        result = add_task(
            db=db,
            customer_id=payload.customer_id,
            title=payload.title,
            due_date=payload.due_date,
            status=payload.status,
            assigned_to=payload.assigned_to
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task creation failed: {str(e)}")


@app.post("/customers")
def create_customer(payload: CustomerCreate):
    db = next(get_db())
    try:
        # Try to create the customer using your service
        customer = Customer(
            id=payload.id or uuid4(),
            name=payload.name,
            industry=payload.industry,
            size=payload.size,
            region=payload.region,
            status=payload.status,
            created_at=payload.created_at or datetime.utcnow(),
            updated_at=payload.updated_at or datetime.utcnow(),
            jira_project_key=payload.jira_project_key,
            salesforce_account_id=payload.salesforce_account_id,
            mainpage_url=payload.mainpage_url,
        )

        db.add(customer)
        db.flush()  # ensure the customer is created and has an ID

        # Add aliases (if any)
        for alias in payload.aliases:
                db.add(CustomerAlias(id=uuid4(), customer_id=customer.id, alias=alias.alias))


        db.commit()
        return {"status": "customer created", "customer_id": str(customer.id)}

    except Exception as e:
        db.rollback()
        logging.error("Customer creation failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Customer creation failed: {str(e)}")




@app.post("/aliases")
def alias_operation(payload: AliasOperationRequest):
    db = next(get_db())
    customer = db.query(Customer).filter(Customer.id == payload.customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    if payload.operation == "add":
        for alias_text in payload.aliases:
            embedding_value = fetch_embedding(alias_text)
            db_alias = CustomerAlias(
                customer_id=payload.customer_id,
                alias=alias_text,
                embedding=embedding_value
            )
            db.add(db_alias)

    elif payload.operation == "delete":
        db.query(CustomerAlias).filter(
            CustomerAlias.customer_id == payload.customer_id,
            CustomerAlias.alias.in_(payload.aliases)
        ).delete(synchronize_session=False)

    elif payload.operation == "update":
        for alias_text in payload.aliases:
            db_alias = db.query(CustomerAlias).filter(
                CustomerAlias.customer_id == payload.customer_id,
                CustomerAlias.alias == alias_text
            ).first()
            if db_alias:
                db_alias.embedding = fetch_embedding(alias_text)

    db.commit()
    return {
        "status": f"aliases {payload.operation}d",
        "customer_id": str(payload.customer_id),
        "aliases": payload.aliases
    }


@app.post("/contacts/search")
def search_contacts_api(payload: ContactSearchRequest):
    db = next(get_db())
    query = db.query(Contact)

    if payload.customer_id:
        query = query.filter(Contact.customer_id == payload.customer_id)

    query = search_contacts(query, payload.filters or [])

    results = query.all()

    return [
        {
            "id": str(contact.id),
            "customer_id": str(contact.customer_id),
            "name": contact.name,
            "role": contact.role,
            "email": contact.email,
            "phone": contact.phone,
            "notes": contact.notes
        }
        for contact in results
    ]



@app.post("/feature-requests")
def create_feature_request(payload: FeatureRequestCreate):
    db = next(get_db())
    try:
        result = add_feature_request(
            db=db,
            customer_id=payload.customer_id,
            request_title=payload.request_title,
            description=payload.description,
            priority=payload.priority,
            status=payload.status,
            estimated_delivery=payload.estimated_delivery,
            internal_notes=payload.internal_notes
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature request creation failed: {str(e)}")


@app.patch("/customers/{customer_id}")
def update_customer(customer_id: UUID, update: CustomerUpdateRequest):
    db = next(get_db())
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    if update.name:
        customer.name = update.name

    db.commit()
    return {"status": "updated", "customer_id": str(customer.id)}

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

@app.post("/notes")
def create_note(payload: NoteCreateRequest):
    db = next(get_db())
    try:
        result = add_note(
            db=db,
            customer_id=payload.customer_id,
            author=payload.author,
            category=payload.category or "",
            full_note=payload.full_note,
            tags=payload.tags,
            source=payload.source or "",
            timestamp=payload.timestamp
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Note creation failed: {str(e)}")

@app.get("/schema")
def get_schema():
    schema_info = []
    for table_name, table in metadata.tables.items():
        schema_info.append({
            "table": table_name,
            "columns": [col.name for col in table.columns]
        })
    return {"schema": schema_info}
