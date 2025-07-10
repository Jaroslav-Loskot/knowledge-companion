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

from contact_service import add_contact, search_contacts, ContactPayload
from featurerequest_service import add_feature_request
from models import Base, Contact, Customer, CustomerAlias
from note_service import add_note
from schemas import (
    AliasOperationRequest,
    ContactCreate,
    ContactSearchRequest,
    CustomerAliasCreate,
    CustomerCreate,
    CustomerUpdateRequest,
    CustomerVectorSearchRequest,
    FeatureRequestCreate,
    NoteCreateRequest,
    TaskCreate,
)
from task_service import add_task
from utils.bedrock_wrapper import fetch_embedding

# Load AWS credentials from .env
load_dotenv(override=True)

# --- DB SETUP ---
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)
metadata = MetaData()
metadata.reflect(bind=engine)


# --- FASTAPI APP ---
app = FastAPI(
    title="Knowledge Companion Service",
    description="A microservice for managing customer identities, aliases, and embeddings, supporting AI agents and RAG-based systems in an integrated Information Hub.",
    version="1.0.0",
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
            assigned_to=payload.assigned_to,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task creation failed: {str(e)}")


@app.post("/contacts")
def create_contact(payload: ContactCreate):
    db = next(get_db())
    try:
        result = add_contact(db=db, payload=ContactPayload(**payload.dict()))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contact creation failed: {str(e)}")


@app.post("/customers")
def create_customer(payload: CustomerCreate):
    db = next(get_db())
    try:
        # Step 1: Create the customer
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
        db.flush()  # Get generated ID

        # Step 2: Prepare aliases (always include customer name)
        alias_texts = [payload.name]
        if payload.aliases:
            alias_texts.extend([a.alias for a in payload.aliases])

        # Step 3: Add aliases with embeddings
        for alias_text in set(alias_texts):  # avoid duplicates
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


@app.post("/customers/search")
def vector_search_customers(payload: CustomerVectorSearchRequest):
    db = next(get_db())
    try:
        # Step 1: Get embedding for query
        embedding = fetch_embedding(payload.query)
        if not embedding:
            raise HTTPException(
                status_code=400, detail="Embedding could not be generated."
            )

        # Step 2: Run the similarity search using pgvector
        # Note: use ARRAY syntax to cast Python list to PostgreSQL-compatible array
        sql = text(
            """
            SELECT customer_id, alias, embedding <-> CAST(:query_vector AS vector) AS distance
            FROM customer_alias
            WHERE embedding IS NOT NULL
            ORDER BY embedding <-> CAST(:query_vector AS vector)
            LIMIT :top_k
        """
        )

        results = db.execute(
            sql, {"query_vector": embedding, "top_k": payload.top_k}
        ).fetchall()

        if not results:
            return []

        # Step 3: Collect customer IDs and fetch their data
        customer_ids = list(set(str(row.customer_id) for row in results))
        customers = db.query(Customer).filter(Customer.id.in_(customer_ids)).all()

        return [
            {"id": str(c.id), "name": c.name, "aliases": [a.alias for a in c.aliases]}
            for c in customers
        ]

    except Exception as e:
        print(f"Error during vector search: {e}")
        raise HTTPException(status_code=500, detail=f"Customer search failed: {str(e)}")


@app.post("/aliases")
def alias_operation(payload: AliasOperationRequest):
    db = next(get_db())
    customer = db.query(Customer).filter(Customer.id == payload.customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    if payload.operation == "add":
        for alias_text in payload.aliases:
            try:
                embedding_value = fetch_embedding(alias_text)
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Failed to fetch embedding: {str(e)}"
                )

            db_alias = CustomerAlias(
                customer_id=payload.customer_id,
                alias=alias_text,
                embedding=embedding_value,
            )
            db.add(db_alias)

    elif payload.operation == "delete":
        db.query(CustomerAlias).filter(
            CustomerAlias.customer_id == payload.customer_id,
            CustomerAlias.alias.in_(payload.aliases),
        ).delete(synchronize_session=False)

    elif payload.operation == "update":
        for alias_text in payload.aliases:
            db_alias = (
                db.query(CustomerAlias)
                .filter(
                    CustomerAlias.customer_id == payload.customer_id,
                    CustomerAlias.alias == alias_text,
                )
                .first()
            )
            if db_alias:
                db_alias.embedding = fetch_embedding(alias_text)

    db.commit()
    return {
        "status": f"aliases {payload.operation}d",
        "customer_id": str(payload.customer_id),
        "aliases": payload.aliases,
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
            "notes": contact.notes,
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
            internal_notes=payload.internal_notes,
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Feature request creation failed: {str(e)}"
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

    result = []
    for customer in customers:
        aliases = [alias.alias for alias in customer.aliases]
        result.append(
            {"id": str(customer.id), "name": customer.name, "aliases": aliases}
        )

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
            timestamp=payload.timestamp,
        )
        return result
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
