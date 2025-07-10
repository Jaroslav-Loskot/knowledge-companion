from typing import Optional
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy.orm import Session, Query

from models import Contact
from utils.bedrock_wrapper import fetch_embedding
from utils.search import apply_dynamic_filters
from schemas import ContactPayload, ContactUpdatePayload, ContactSearchRequest, OperationStatus


def add_contact(db: Session, payload: ContactPayload) -> OperationStatus:
    """Add a new contact and generate its name embedding."""
    contact_id = uuid4()
    contact = Contact(
        id=contact_id,
        customer_id=payload.customer_id,
        name=payload.name,
        role=payload.role,
        email=payload.email,
        phone=payload.phone,
        notes=payload.notes,
        name_embedding=fetch_embedding(payload.name),
    )
    db.add(contact)
    db.commit()
    return OperationStatus(status="created", entity="contact", id=str(contact_id))


def update_contact(db: Session, payload: ContactUpdatePayload) -> OperationStatus:
    """Update an existing contact and regenerate embedding if name changes."""
    contact = db.query(Contact).filter(Contact.id == payload.contact_id).first()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")

    name_changed = False
    for field, value in payload.dict(exclude_unset=True).items():
        if field != "contact_id" and getattr(contact, field) != value:
            setattr(contact, field, value)
            if field == "name":
                name_changed = True

    if name_changed:
        try:
            contact.name_embedding = fetch_embedding(contact.name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

    db.commit()
    return OperationStatus(status="updated", entity="contact", id=str(contact.id))


def delete_contact(db: Session, contact_id: UUID) -> OperationStatus:
    """Delete an existing contact by ID."""
    contact = db.query(Contact).filter(Contact.id == contact_id).first()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")

    db.delete(contact)
    db.commit()
    return OperationStatus(status="deleted", entity="contact", id=str(contact_id))


def search_contacts(query: Query, payload: ContactSearchRequest) -> Query:
    """Apply dynamic filters to a contact query."""
    if payload.customer_id:
        query = query.filter(Contact.customer_id == payload.customer_id)

    if payload.filters:
        query = apply_dynamic_filters(query, Contact, payload.filters)

    return query
