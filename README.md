# 🧠 Knowledge Companion

A FastAPI-based microservice for managing customers, contacts, notes, tasks, and feature requests — built to support AI-driven workflows, vector-based search, and real-time summarization using AWS Bedrock (Claude Sonnet 4).

---

## 📦 Features

* **CRUD for Customers, Contacts, Tasks, Notes, Feature Requests**
* **Claude Sonnet 4 Summarization** for Notes & Feature Requests
* **Vector Search** with pgvector for customer aliases
* **Clean FastAPI endpoints** with modular services
* **Embedding Support** for natural language understanding
* **Prompt capture utility** for LLM context packaging

---

## ✨ Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/knowledge-companion.git
cd knowledge-companion
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file:

```ini
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=knowledge_companion

AWS_REGION=eu-north-1
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_SESSION_TOKEN=optional
BEDROCK_INFERENCE_CONFIG_ARN=arn:aws:bedrock:...
```

### 3. Start the server

```bash
uvicorn main:app --reload
```

---

## 🧪 Example Endpoints

### Health Check

```http
GET /health
```

### Create Customer

```http
POST /customers
```

### Add Contact

```http
POST /contacts
{
  "operation": "add",
  "payload": {
    "customer_id": "UUID-HERE",
    "name": "John Doe",
    "role": "Engineer",
    "email": "john@example.com",
    "phone": "123-456",
    "notes": "Important contact"
  }
}
```

### Create Note

```http
POST /notes
{
  "customer_id": "UUID-HERE",
  "author": "Alice",
  "category": "Meeting",
  "full_note": "We discussed Q3 roadmap.",
  "tags": ["meeting", "q3"],
  "source": "slack"
}
```

---

## 🧪 Test Claude Summarization

Test Claude Sonnet 4 directly:

```bash
python test_claude_summarization.py
```

---

## 📁 Code Organization

| Path                       | Purpose                              |
| -------------------------- | ------------------------------------ |
| `main.py`                  | FastAPI routes                       |
| `models.py`                | SQLAlchemy models                    |
| `schemas.py`               | Pydantic request/response models     |
| `services/`                | Business logic split by domain       |
| `utils/bedrock_wrapper.py` | Claude/Bedrock helper functions      |
| `prompt.txt`               | Generated context from project files |

---

## ✨ Prompt Capture Utility

To package code context into `prompt.txt`:

```bash
python generate_prompt.py
```

This scans `.py` files (excluding `.venv`, `.git`, etc.) and writes clean output to `prompt.txt` — ideal for LLM context.

---

## 🧐 Vector Search with pgvector

Used for searching customer aliases via embedding similarity:

```http
POST /customers/search
{
  "query": "Acme Ltd",
  "top_k": 5
}
```

---

## 🛠️ Tech Stack

* **FastAPI**
* **SQLAlchemy**
* **PostgreSQL** + `pgvector`
* **Claude via AWS Bedrock**
* **Uvicorn**
* **dotenv**, **Pydantic**

---

## 📝 License

MIT – free to use, modify, or fork.
