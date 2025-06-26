from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel

# --- SCHEMAS ---


class CustomerVectorSearchRequest(BaseModel):
    query: str
    top_k: int = 3


class TaskCreate(BaseModel):
    customer_id: UUID
    title: str
    due_date: datetime
    status: str
    assigned_to: str


class ContactCreate(BaseModel):
    customer_id: UUID
    name: str
    role: str
    email: str
    phone: str
    notes: str


class ContactSearchFilter(BaseModel):
    field: str
    value: str


class ContactSearchRequest(BaseModel):
    customer_id: Optional[UUID] = None
    filters: Optional[List[ContactSearchFilter]] = []


class FeatureRequestCreate(BaseModel):
    customer_id: UUID
    request_title: str
    description: str
    priority: str
    status: str
    estimated_delivery: datetime
    internal_notes: str


class CustomerAliasCreate(BaseModel):
    alias: str
    embedding: Optional[List[float]] = None


class AliasOperationRequest(BaseModel):
    operation: str
    customer_id: UUID
    aliases: List[str]


class CustomerCreate(BaseModel):
    id: Optional[UUID] = None
    name: str
    industry: Optional[str] = None
    size: Optional[str] = None
    region: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    jira_project_key: Optional[str] = None
    salesforce_account_id: Optional[str] = None
    mainpage_url: Optional[str] = None
    aliases: Optional[List[CustomerAliasCreate]] = []


class AliasOperationRequest(BaseModel):
    operation: Literal["add", "delete", "update"]
    customer_id: UUID
    aliases: List[str]


class CustomerUpdateRequest(BaseModel):
    name: Optional[str] = None


class NoteCreateRequest(BaseModel):
    customer_id: UUID
    author: str
    category: str
    full_note: str
    tags: List[str]
    source: str
    timestamp: Optional[datetime] = None
