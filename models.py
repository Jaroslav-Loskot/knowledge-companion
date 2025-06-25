from sqlalchemy import Column, Text, Integer, ForeignKey, TIMESTAMP, Uuid
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Customer(Base):
    __tablename__ = 'customer'
    id = Column(Uuid, primary_key=True)
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
    id = Column(Integer, primary_key=True)
    customer_id = Column(Uuid, ForeignKey('customer.id'))
    alias = Column(Text)
    embedding = Column(Vector(1024))  # or the appropriate dimension
    customer = relationship("Customer", back_populates="aliases")
