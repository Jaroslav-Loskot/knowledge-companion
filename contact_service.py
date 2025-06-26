from uuid import uuid4, UUID
from sqlalchemy.orm import Session
from models import Contact
from datetime import datetime
from sqlalchemy.orm import Query
from sqlalchemy import or_

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

def search_fields(query: Query, model, fields: list[str], search_text: str) -> Query:
    """
    Apply an ILIKE filter to multiple fields in a SQLAlchemy query.
    
    :param query: SQLAlchemy query object
    :param model: The SQLAlchemy model to search on
    :param fields: List of field names (as strings)
    :param search_text: The text to search for
    :return: Modified query with filters applied
    """
    if not search_text.strip():
        return query

    filters = [getattr(model, field).ilike(f"%{search_text}%") for field in fields]
    return query.filter(or_(*filters))