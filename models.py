import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import TIMESTAMP, Column, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Task(Base):
    __tablename__ = "task"
    id = Column(UUID(as_uuid=True), primary_key=True)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customer.id"))
    title = Column(Text)
    due_date = Column(TIMESTAMP)
    status = Column(Text)
    assigned_to = Column(Text)

    summary = Column(Text)
    embedding = Column(Vector(1024))


class Customer(Base):
    __tablename__ = "customer"
    id = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(Text)
    industry = Column(Text)
    size = Column(Text)
    region = Column(Text)
    status = Column(Text)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)
    jira_project_key = Column(Text)
    salesforce_account_id = Column(Text)
    mainpage_url = Column(Text)
    aliases = relationship(
        "CustomerAlias", back_populates="customer", cascade="all, delete"
    )


class CustomerAlias(Base):
    __tablename__ = "customer_alias"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customer.id"))
    alias = Column(Text)
    embedding = Column(Vector(1024))
    customer = relationship("Customer", back_populates="aliases")


class CustomNote(Base):
    __tablename__ = "custom_notes"
    id = Column(UUID(as_uuid=True), primary_key=True)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customer.id"))
    author = Column(Text)
    timestamp = Column(TIMESTAMP)
    category = Column(Text)
    summary = Column(Text)
    full_note = Column(Text)
    tags = Column(JSONB)
    source = Column(Text)
    embedding = Column(Vector(1024))


class FeatureRequest(Base):
    __tablename__ = "feature_request"
    id = Column(UUID(as_uuid=True), primary_key=True)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customer.id"))
    request_title = Column(Text)
    summary = Column(Text)
    priority = Column(Text)
    status = Column(Text)
    created_at = Column(TIMESTAMP)
    raw_input = Column(Text)              # ✅ renamed from row_input
    embedding = Column(Vector(1024))


class Contact(Base):
    __tablename__ = "contact"
    id = Column(UUID(as_uuid=True), primary_key=True)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customer.id"))
    name = Column(Text)
    role = Column(Text)
    email = Column(Text)
    phone = Column(Text)
    notes = Column(Text)
    name_embedding = Column(Vector(1024))   
