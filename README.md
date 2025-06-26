````markdown
# 🧠 Knowledge Companion

**Knowledge Companion** is a FastAPI microservice designed to manage customer metadata, semantic aliases, and meeting notes with automatic summarization and vector embeddings. It serves as a backend support service for AI agents and RAG-based systems in an integrated Information Hub.

## 🚀 Features

- Create, update, and delete customers with associated metadata.
- Manage semantic customer aliases and generate embeddings using Amazon Titan.
- Record, summarize, and embed custom notes using Claude (via Amazon Bedrock).
- PostgreSQL + pgvector support for vector storage.
- REST API powered by FastAPI.
- Schema introspection endpoint for dynamic integration.

## 📦 Tech Stack

- **Python 3.10+**
- **FastAPI** for web service
- **SQLAlchemy + PostgreSQL (pgvector)** for data storage
- **Amazon Bedrock** for LLMs and embeddings
- **Claude Sonnet** for summarization
- **Titan Embed Text** for vector embeddings

## ⚙️ Environment Variables

Create a `.env` file in the root with the following content:

```env
# AWS
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=your_region
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_INFERENCE_CONFIG_ARN=...

# Database
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_db_name
````

## 🧪 API Endpoints

### Health Check

```http
GET /health
```

### Customers

* **Create a customer**

```http
POST /customers
```

* **Update customer name**

```http
PATCH /customers/{customer_id}
```

* **Delete a customer**

```http
DELETE /customers/{customer_id}
```

* **Search customer by ID or name**

```http
GET /customers
```

### Aliases

* Add, update, or delete customer aliases (with embedding)

```http
POST /aliases
```

### Notes

* Add a note with automatic summarization and embedding

```http
POST /notes
```

### Schema

* Get database schema for introspection

```http
GET /schema
```

## 🛠 Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/Jaroslav-Loskot/knowledge-companion.git
cd knowledge-companion
```

### 2. Create virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up PostgreSQL

Ensure you have a PostgreSQL database ready and accessible using the `.env` credentials. Make sure the `pgvector` extension is enabled.

### 4. Run the server

```bash
uvicorn knowledge_companion:app --reload
```

## 🧠 Usage Flow

1. **Create a customer** with one or more aliases.
2. **Aliases** are automatically embedded with Titan for semantic search.
3. **Add notes** to customers – notes are summarized by Claude and stored with embeddings.
4. Use embeddings in downstream RAG or search systems.

## 🧪 Development Notes

* Models defined in `models.py`
* Claude summarization and Titan embeddings in `bedrock_wrapper.py`
* Database-backed notes and customers linked semantically
* `.env` required for sensitive configs

## 📜 License

MIT

## ✨ Future Ideas

* Semantic search over notes and customers
* UI dashboard for browsing customers
* Integration with external tools like Salesforce or Jira

---

Maintained by [@Jaroslav-Loskot](https://github.com/Jaroslav-Loskot)

```

