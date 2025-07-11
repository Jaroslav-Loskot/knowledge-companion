from datetime import datetime
from typing import List, Literal, Optional, Dict
from uuid import UUID

from pydantic import BaseModel

# --- COMMON SCHEMAS ---
class OperationStatus(BaseModel):
    status: str
    entity: str
    id: str


# --- CONTACT SCHEMAS ---
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


class ContactSearchFilter(BaseModel):
    field: str
    value: str


class ContactSearchRequest(BaseModel):
    customer_id: Optional[UUID] = None
    filters: Optional[List[ContactSearchFilter]] = []


class ContactOperationRequest(BaseModel):
    operation: Literal["add", "update", "delete"]
    payload: dict


# --- CUSTOMER SCHEMAS ---
class CustomerVectorSearchRequest(BaseModel):
    query: str
    top_k: int = 3


class CustomerAliasCreate(BaseModel):
    alias: str
    embedding: Optional[List[float]] = None


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


class CustomerUpdateRequest(BaseModel):
    name: Optional[str] = None


class AliasOperationRequest(BaseModel):
    operation: Literal["add", "delete", "update"]
    customer_id: UUID
    aliases: List[str]


# --- FEATURE REQUEST SCHEMAS ---
class FeatureRequestFromRaw(BaseModel):
    customer_id: UUID
    raw_input: str
    priority: Optional[str] = "unspecified"
    status: Optional[str] = "new"


class FeatureRequestUpdatePayload(BaseModel):
    request_id: UUID
    raw_input: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None


class FeatureRequestOperationRequest(BaseModel):
    operation: Literal["add", "update", "delete"]
    payload: Dict


# --- NOTE SCHEMAS ---
class NoteCreateRequest(BaseModel):
    customer_id: UUID
    author: str
    category: str
    full_note: str
    tags: List[str]
    source: str
    timestamp: Optional[datetime] = None

class NoteSearchRequest(BaseModel):
    customer_id: Optional[UUID] = None
    query: str
    top_k: int = 5
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None




# --- TASK SCHEMAS ---
class TaskCreate(BaseModel):
    customer_id: UUID
    title: str
    due_date: datetime
    status: str
    assigned_to: str
