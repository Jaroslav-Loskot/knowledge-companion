# 🧠 Knowledge Companion

Knowledge Companion is a FastAPI microservice that stores customer information and related knowledge for other applications. It generates embeddings for aliases, notes, feature requests and tasks using Amazon Bedrock models so the data can be semantically searched.

## Features

- **Customer management** – create, update and delete customers with rich metadata
- **Semantic aliases** – add, update or remove aliases that are embedded with Amazon Titan
- **Vector search** – search for customers by text using pgvector similarity
- **Notes** – store lengthy meeting notes which are summarised by Claude and embedded
- **Feature requests** – record product feature requests and automatically summarise them
- **Tasks** – create tasks linked to customers, summarised for easy reference
- **Contact search** – filter stored contacts using dynamic field filters
- **Schema endpoint** – expose database schema for quick introspection

## Tech Stack

- Python 3.11
- FastAPI
- SQLAlchemy with PostgreSQL and `pgvector`
- Amazon Bedrock (Claude Sonnet & Titan Embed Text)

## Environment variables

Create a `.env` file in the project root:

```env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=your_region
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_INFERENCE_CONFIG_ARN=your_inference_arn

DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=knowledge
```

## API

### Health
`GET /health`

### Customers
`POST /customers` – create customer

`POST /customers/search` – vector search

`PATCH /customers/{customer_id}` – update name

`DELETE /customers/{customer_id}` – delete customer

`GET /customers` – search by id or name

### Aliases
`POST /aliases` – add, delete or update aliases

### Notes
`POST /notes` – create note with summarisation

### Contacts
`POST /contacts` – create contact
`POST /contacts/search` – filter contacts

### Feature requests
`POST /feature-requests` – create request

### Tasks
`POST /tasks` – create task

### Schema
`GET /schema` – list all tables and columns

## Running

Install the dependencies and start the server:

```bash
pip install -r requirements.txt
uvicorn knowledge_companion:app --reload
```

A Dockerfile and `docker-compose.yml` are provided for container based deployments.

## Development notes

The core endpoints live in `knowledge_companion.py` with helper services in separate `*_service.py` files. Embedding and summarisation helpers are located in `utils/bedrock_wrapper.py`.

## License

MIT

