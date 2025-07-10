import logging
import os
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

from models import Base, Contact, Customer, CustomerAlias
from utils.bedrock_wrapper import fetch_embedding

# --- Services ---
from contact_service import (
    ContactUpdatePayload,
    update_contact,
    delete_contact,
    ContactPayload,
    add_contact,
    search_contacts,
)
from featurerequest_service import (
    handle_feature_request_operation,
)
from note_service import add_note
from task_service import add_task

# --- Schemas ---
from schemas import (
    AliasOperationRequest,
    ContactSearchRequest,
    ContactOperationRequest,
    CustomerCreate,
    CustomerUpdateRequest,
    CustomerVectorSearchRequest,
    FeatureRequestOperationRequest,
    NoteCreateRequest,
    TaskCreate,
)

# --- Load environment ---
load_dotenv(override=True)

# --- DB Setup ---
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

# --- FastAPI App ---
app = FastAPI(
    title="Knowledge Companion Service",
    description="A microservice for managing customer identities, aliases, and embeddings, supporting AI agents and RAG-based systems in an integrated Information Hub.",
    version="1.0.0",
)


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
        return add_task(
            db=db,
            customer_id=payload.customer_id,
            title=payload.title,
            due_date=payload.due_date,
            status=payload.status,
            assigned_to=payload.assigned_to,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task creation failed: {str(e)}")


@app.post("/contacts")
def handle_contact_operation(payload: ContactOperationRequest):
    db = next(get_db())
    try:
        if payload.operation == "add":
            contact_payload = ContactPayload(**payload.payload)
            return add_contact(db=db, payload=contact_payload)

        elif payload.operation == "update":
            update_payload = ContactUpdatePayload(**payload.payload)
            return update_contact(db=db, payload=update_payload)

        elif payload.operation == "delete":
            contact_id = UUID(payload.payload.get("contact_id"))
            return delete_contact(db=db, contact_id=contact_id)

        else:
            raise HTTPException(status_code=400, detail="Invalid operation type")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Contact {payload.operation} failed: {str(e)}"
        )


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
            "notes": contact.notes,
        }
        for contact in results
    ]


@app.post("/feature-requests")
def handle_feature_request_op(payload: FeatureRequestOperationRequest):
    db = next(get_db())
    try:
        return handle_feature_request_operation(db, payload)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Feature request operation failed: {str(e)}"
        )


@app.post("/customers")
def create_customer(payload: CustomerCreate):
    db = next(get_db())
    try:
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
        db.flush()

        alias_texts = [payload.name]
        if payload.aliases:
            alias_texts.extend([a.alias for a in payload.aliases])

        for alias_text in set(alias_texts):
            embedding = fetch_embedding(alias_text)
            db.add(
                CustomerAlias(
                    id=uuid4(),
                    customer_id=customer.id,
                    alias=alias_text,
                    embedding=embedding,
                )
            )

        db.commit()
        return {"status": "customer created", "customer_id": str(customer.id)}

    except Exception as e:
        db.rollback()
        logging.error("Customer creation failed", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Customer creation failed: {str(e)}"
        )


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

    return [
        {"id": str(c.id), "name": c.name, "aliases": [a.alias for a in c.aliases]}
        for c in customers
    ]


@app.post("/customers/search")
def vector_search_customers(payload: CustomerVectorSearchRequest):
    db = next(get_db())
    try:
        embedding = fetch_embedding(payload.query)
        if not embedding:
            raise HTTPException(status_code=400, detail="Embedding could not be generated.")

        sql = text("""
            SELECT customer_id, alias, embedding <-> CAST(:query_vector AS vector) AS distance
            FROM customer_alias
            WHERE embedding IS NOT NULL
            ORDER BY embedding <-> CAST(:query_vector AS vector)
            LIMIT :top_k
        """)

        results = db.execute(sql, {"query_vector": embedding, "top_k": payload.top_k}).fetchall()

        customer_ids = list(set(str(row.customer_id) for row in results))
        customers = db.query(Customer).filter(Customer.id.in_(customer_ids)).all()

        return [
            {"id": str(c.id), "name": c.name, "aliases": [a.alias for a in c.aliases]}
            for c in customers
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Customer search failed: {str(e)}")


@app.post("/aliases")
def alias_operation(payload: AliasOperationRequest):
    db = next(get_db())
    customer = db.query(Customer).filter(Customer.id == payload.customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    try:
        if payload.operation == "add":
            for alias_text in payload.aliases:
                embedding = fetch_embedding(alias_text)
                db.add(CustomerAlias(customer_id=payload.customer_id, alias=alias_text, embedding=embedding))

        elif payload.operation == "delete":
            db.query(CustomerAlias).filter(
                CustomerAlias.customer_id == payload.customer_id,
                CustomerAlias.alias.in_(payload.aliases),
            ).delete(synchronize_session=False)

        elif payload.operation == "update":
            for alias_text in payload.aliases:
                db_alias = db.query(CustomerAlias).filter(
                    CustomerAlias.customer_id == payload.customer_id,
                    CustomerAlias.alias == alias_text,
                ).first()
                if db_alias:
                    db_alias.embedding = fetch_embedding(alias_text)

        db.commit()
        return {
            "status": f"aliases {payload.operation}d",
            "customer_id": str(payload.customer_id),
            "aliases": payload.aliases,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alias operation failed: {str(e)}")


@app.post("/notes")
def create_note(payload: NoteCreateRequest):
    db = next(get_db())
    try:
        return add_note(
            db=db,
            customer_id=payload.customer_id,
            author=payload.author,
            category=payload.category or "",
            full_note=payload.full_note,
            tags=payload.tags,
            source=payload.source or "",
            timestamp=payload.timestamp,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Note creation failed: {str(e)}")


@app.get("/schema")
def get_schema():
    schema_info = []
    for table_name, table in metadata.tables.items():
        schema_info.append(
            {"table": table_name, "columns": [col.name for col in table.columns]}
        )
    return {"schema": schema_info}
