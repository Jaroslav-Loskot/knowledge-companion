````markdown
# Knowledge Companion Service

A microservice built with FastAPI to manage customer information, aliases, and embeddings. Designed to support AI agents and RAG-based systems within an integrated Information Hub.

---

## 🚀 Features

- Create and update customer records
- Add, delete, or update customer aliases
- Automatically generates text embeddings for aliases
- Query customers by ID or name
- Health check and database schema introspection

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
````

---

## ⚙️ Environment Configuration

Copy `.env.example` to `.env` and fill in your configuration:

```env
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=your_db_name

AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

---

## 🐳 Docker

Build and run the container:

```bash
docker-compose up --build
```

The service will be available at: [http://localhost:8010](http://localhost:8010)

---

## 🧪 Health Check

```http
GET /health
```

---

## 📚 Endpoints

### ➕ Create Customer

```http
POST /customers
```

Payload:

```json
{
  "name": "DFCU",
  "aliases": [
    { "alias": "Cedar Point" },
    { "alias": "CPFCU" }
  ]
}
```

### 🔄 Update Customer Name

```http
PATCH /customers/{customer_id}
```

Payload:

```json
{
  "name": "New Name"
}
```

### ❌ Delete Customer

```http
DELETE /customers/{customer_id}
```

---

### 📬 Manage Aliases

```http
POST /aliases
```

Payload:

```json
{
  "operation": "add",         // or "delete", "update"
  "customer_id": "uuid-here",
  "aliases": ["Alias A", "Alias B"]
}
```

---

### 🔍 Query Customers

```http
GET /customers?id=...&name=...
```

Supports filtering by ID, name, or both.

---

### 📊 View Schema

```http
GET /schema
```

Returns a list of all tables and their columns.

---

## 💡 Future Ideas

* Fuzzy alias resolution
* Alias deduplication logic
* Full-text search
* Embedding similarity-based lookups

