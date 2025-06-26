from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Text, TIMESTAMP, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()

class Customer(Base):
    __tablename__ = 'customer'
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
    aliases = relationship("CustomerAlias", back_populates="customer", cascade="all, delete")

class CustomerAlias(Base):
    __tablename__ = 'customer_alias'
    id = Column(UUID(as_uuid=True), primary_key=True)
    customer_id = Column(UUID(as_uuid=True), ForeignKey('customer.id'))
    alias = Column(Text)
    embedding = Column(Vector(1024))
    customer = relationship("Customer", back_populates="aliases")

class CustomNote(Base):
    __tablename__ = "customnotes"
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
    __tablename__ = "featurerequest"
    id = Column(UUID(as_uuid=True), primary_key=True)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customer.id"))
    request_title = Column(Text)
    description = Column(Text)  # raw input
    priority = Column(Text)
    status = Column(Text)
    estimated_delivery = Column(TIMESTAMP)
    internal_notes = Column(Text)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)

    summary = Column(Text)  # ✅ new
    embedding = Column(Vector(1024))  # ✅ new

