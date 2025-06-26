from sqlalchemy.orm import Query
from sqlalchemy import or_
from typing import List
from pydantic import BaseModel


class SearchFilter(BaseModel):
    field: str
    value: str


def apply_dynamic_filters(query: Query, model, filters: List[SearchFilter]) -> Query:
    if not filters:
        return query

    conditions = []
    for f in filters:
        if hasattr(model, f.field):
            conditions.append(getattr(model, f.field).ilike(f"%{f.value}%"))

    if conditions:
        query = query.filter(or_(*conditions))

    return query
