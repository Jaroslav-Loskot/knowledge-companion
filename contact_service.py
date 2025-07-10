from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Query, Session

from models import Contact
from utils.bedrock_wrapper import fetch_embedding
from utils.search import SearchFilter, apply_dynamic_filters


# --- SCHEMAS FOR SERVICE USE ---
class ContactPayload(BaseModel):
    customer_id: UUID
    name: str
    role: str
    email: str
    phone: str
    notes: str


class ContactUpdatePayload(BaseModel):
    contact_id: UUID
    name: Optional[str] = None
    role: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    notes: Optional[str] = None


class ContactSearchPayload(BaseModel):
    filters: List[SearchFilter] = []
    customer_id: Optional[UUID] = None


# --- CONTACT OPERATIONS ---
def add_contact(db: Session, payload: ContactPayload):
    contact_id = uuid4()

    name_embedding = fetch_embedding(payload.name)

    contact = Contact(
        id=contact_id,
        customer_id=payload.customer_id,
        name=payload.name,
        role=payload.role,
        email=payload.email,
        phone=payload.phone,
        notes=payload.notes,
        embedding=name_embedding,
    )

    db.add(contact)
    db.commit()
    return {"status": "contact created", "contact_id": str(contact_id)}


def update_contact(db: Session, payload: ContactUpdatePayload):
    contact = db.query(Contact).filter(Contact.id == payload.contact_id).first()
    if not contact:
        return {"error": "Contact not found", "contact_id": str(payload.contact_id)}

    # Update only provided fields
    for field, value in payload.dict(exclude_unset=True).items():
        if field != "contact_id":
            setattr(contact, field, value)

    db.commit()
    return {"status": "contact updated", "contact_id": str(contact.id)}


def search_contacts(query: Query, payload: ContactSearchPayload) -> Query:
    """
    Applies dynamic filters to a SQLAlchemy query using `utils.search`.
    Optionally narrows by customer_id.
    """
    if payload.customer_id:
        query = query.filter(Contact.customer_id == payload.customer_id)

    if payload.filters:
        query = apply_dynamic_filters(query, Contact, payload.filters)

    return query
