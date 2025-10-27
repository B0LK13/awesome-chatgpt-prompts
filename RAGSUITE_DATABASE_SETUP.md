# RAGSUITE Database Setup

## Overview
RAGSUITE is configured with PostgreSQL 16 and the pgvector extension to power Retrieval Augmented Generation (RAG) workflows. This setup supports efficient vector similarity search, document storage, and conversation tracking features tailored for AI applications.

## Database Information
- **Host:** localhost
- **Port:** 5433
- **Database:** ragsuite
- **Username:** ragsuite_user
- **Password:** ragsuite_password

## Features
- PostgreSQL 16 with pgvector extension v0.8.1
- Vector similarity search using cosine distance
- Document storage with metadata support
- User management system
- Conversation/session tracking
- Optimized IVFFlat indexes (100 lists) for vector search

## Schema Overview

### Tables
1. **documents** – Stores documents with 1,536-dimensional vector embeddings
2. **users** – Manages application users
3. **sessions** – Tracks user sessions
4. **conversations** – Records chat history across sessions

### Functions
- `search_documents(query_embedding, threshold, limit)` – Returns documents similar to the provided embedding.

## Usage

### Connect to the Database
```bash
# Using environment variables
source database_config.env
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME

# Or directly
PGPASSWORD=ragsuite_password psql -h localhost -p 5433 -U ragsuite_user -d ragsuite
```

### Container Management
```bash
# Start the database
docker compose up -d

# Stop the database
docker compose down

# View logs
docker compose logs -f postgres

# Restart with fresh data (WARNING: destroys all data)
docker compose down -v && docker compose up -d
```

### Example Vector Operations
```sql
-- Insert a document with embedding
INSERT INTO documents (content, metadata, embedding) 
VALUES ('Your document content', '{"source": "api"}', your_vector_array);

-- Search for similar documents
SELECT * FROM search_documents(your_query_vector, 0.7, 10);
```

## Development Notes
- Vector dimensions are fixed at 1,536 (OpenAI embedding size)
- Cosine similarity powers vector comparisons
- Timestamp columns are stored with timezone information
