from uuid import uuid4, UUID
from sqlalchemy.orm import Session, Query
from models import Contact
from typing import List
from sqlalchemy import or_
from utils.search import apply_dynamic_filters, SearchFilter


def add_contact(
    db: Session,
    customer_id: UUID,
    name: str,
    role: str,
    email: str,
    phone: str,
    notes: str
):
    contact_id = uuid4()

    contact = Contact(
        id=contact_id,
        customer_id=customer_id,
        name=name,
        role=role,
        email=email,
        phone=phone,
        notes=notes
    )

    db.add(contact)
    db.commit()
    return {"status": "contact created", "contact_id": str(contact_id)}


def search_contacts(query: Query, filters: List[SearchFilter]) -> Query:
    """
    Applies dynamic ILIKE filters to Contact model using utils.search module.
    Example filters:
        [{"field": "name", "value": "Jaroslav"}, {"field": "email", "value": "@gmail.com"}]
    """
    return apply_dynamic_filters(query, Contact, filters)
