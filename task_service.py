from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from models import Task
from utils.bedrock_wrapper import call_claude, fetch_embedding


def summarize_task(title: str) -> str:
    system_prompt = "You are a task assistant helping summarize tasks."
    return call_claude(system_prompt, title)


def add_task(
    db: Session,
    customer_id: UUID,
    title: str,
    due_date: datetime,
    status: str,
    assigned_to: str,
):
    task_id = uuid4()
    summary = summarize_task(title)
    embedding = fetch_embedding(summary)

    task = Task(
        id=task_id,
        customer_id=customer_id,
        title=title,
        due_date=due_date,
        status=status,
        assigned_to=assigned_to,
        summary=summary,
        embedding=embedding,
    )

    db.add(task)
    db.commit()
    return {"status": "task created", "task_id": str(task_id)}
