# 🧠 Knowledge Companion Service

A FastAPI microservice that manages customer information, aliases, and embedding vectors to support AI agents and Retrieval-Augmented Generation (RAG) systems. Designed as a foundational backend for an integrated Information Hub.

---

## 🚀 Features

- Create and manage customers and their aliases
- Auto-generate alias embeddings using Amazon Bedrock Titan
- Query customer details and aliases
- Explore database schema dynamically via `/schema`
- Easily extendable for future RAG integrations

---

## 🛠 Tech Stack

- Python 3.11
- FastAPI
- SQLAlchemy
- PostgreSQL
- Amazon Bedrock Titan (via boto3)
- Docker (optional)
- `.env` configuration

---

## 📦 Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-org/knowledge-companion.git
cd knowledge-companion
