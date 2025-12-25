# Bio-RAG Implementation Plan (구현 계획서)

> 이 문서는 Bio-RAG 플랫폼의 기술 구현 계획을 정의합니다.
> Specification의 "무엇을"을 "어떻게" 구현할지 상세히 기술합니다.

---

## 1. System Architecture (시스템 아키텍처)

### 1.1 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Web UI      │  │  Mobile UI   │  │  API Client  │              │
│  │  (React.js)  │  │  (Responsive)│  │  (REST)      │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                              │ HTTPS
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  API Gateway (FastAPI)                                       │  │
│  │  - JWT Authentication    - Rate Limiting                     │  │
│  │  - Request Validation    - CORS Handling                     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
│  │  Search     │  │  RAG        │  │  Analytics  │  │  User     │  │
│  │  Service    │  │  Service    │  │  Service    │  │  Service  │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       DATA PROCESSING LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  PubMed      │  │  Embedding   │  │  Batch       │             │
│  │  Collector   │  │  Generator   │  │  Processor   │             │
│  │  (Async)     │  │  (PubMedBERT)│  │  (Celery)    │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         STORAGE LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  PostgreSQL  │  │  Qdrant      │  │  Redis       │             │
│  │  (Metadata)  │  │  (Vector DB) │  │  (Cache)     │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       EXTERNAL SERVICES                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │  PubMed API  │  │  OpenAI API  │  │  Monitoring  │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 기술 스택

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Frontend** | React.js | 18.2+ | UI Framework |
| | TypeScript | 5.0+ | Type Safety |
| | TailwindCSS | 3.3+ | Styling |
| | React Query | 5.0+ | Data Fetching |
| | Zustand | 4.0+ | State Management |
| | Recharts | 2.5+ | Visualization |
| **Backend** | Python | 3.11+ | Primary Language |
| | FastAPI | 0.104+ | Web Framework |
| | Pydantic | 2.0+ | Validation |
| | SQLAlchemy | 2.0+ | ORM |
| | Celery | 5.3+ | Task Queue |
| | Redis | 7.0+ | Cache/Queue |
| **AI/ML** | LangChain | 0.1+ | RAG Framework |
| | Transformers | 4.35+ | Model Loading |
| | PubMedBERT | - | Embeddings |
| | OpenAI API | GPT-4 | LLM |
| **Database** | PostgreSQL | 15+ | Relational DB |
| | Qdrant | 1.7+ | Vector DB |
| **Infra** | Docker | 24+ | Container |
| | AWS ECS | - | Orchestration |

---

## 2. Database Design (데이터베이스 설계)

### 2.1 ERD

```
┌─────────────────────┐       ┌─────────────────────┐
│       users         │       │      papers         │
├─────────────────────┤       ├─────────────────────┤
│ id (PK)             │       │ id (PK)             │
│ email (UNIQUE)      │       │ pmid (UNIQUE)       │
│ password_hash       │       │ title               │
│ name                │       │ abstract            │
│ research_field      │       │ authors (JSONB)     │
│ created_at          │       │ journal             │
│ updated_at          │       │ publication_date    │
└─────────────────────┘       │ keywords (JSONB)    │
         │                    │ mesh_terms (JSONB)  │
         │ 1:N                └─────────────────────┘
         ▼                             │
┌─────────────────────┐                │ 1:N
│   saved_papers      │                ▼
├─────────────────────┤       ┌─────────────────────┐
│ id (PK)             │       │      chunks         │
│ user_id (FK)        │       ├─────────────────────┤
│ paper_id (FK)       │       │ id (PK)             │
│ tags (JSONB)        │       │ paper_id (FK)       │
│ notes               │       │ text                │
└─────────────────────┘       │ section             │
                              │ embedding_id        │
┌─────────────────────┐       └─────────────────────┘
│   chat_sessions     │
├─────────────────────┤
│ id (PK)             │
│ user_id (FK)        │       ┌─────────────────────┐
│ title               │       │   chat_messages     │
│ created_at          │──1:N──├─────────────────────┤
└─────────────────────┘       │ id (PK)             │
                              │ session_id (FK)     │
                              │ role                │
                              │ content             │
                              │ sources (JSONB)     │
                              └─────────────────────┘
```

### 2.2 주요 테이블

```sql
-- Papers 테이블
CREATE TABLE papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pmid VARCHAR(20) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    authors JSONB,
    journal VARCHAR(255),
    publication_date DATE,
    doi VARCHAR(100),
    keywords JSONB,
    mesh_terms JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_papers_pmid ON papers(pmid);
CREATE INDEX idx_papers_keywords ON papers USING GIN(keywords);

-- Chunks 테이블
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id UUID REFERENCES papers(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    section VARCHAR(50),
    chunk_index INTEGER,
    embedding_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chat Sessions 테이블
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chat Messages 테이블
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    sources JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 3. API Design (API 설계)

### 3.1 Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| **Auth** | | |
| POST | `/api/v1/auth/register` | 회원가입 |
| POST | `/api/v1/auth/login` | 로그인 |
| POST | `/api/v1/auth/refresh` | 토큰 갱신 |
| **Search** | | |
| POST | `/api/v1/search` | 의미 기반 검색 |
| GET | `/api/v1/papers/{pmid}` | 논문 상세 |
| GET | `/api/v1/papers/{pmid}/similar` | 유사 논문 |
| **Chat** | | |
| POST | `/api/v1/chat/query` | AI 질의 |
| POST | `/api/v1/chat/sessions` | 세션 생성 |
| GET | `/api/v1/chat/sessions` | 세션 목록 |
| **Library** | | |
| POST | `/api/v1/library/papers` | 논문 저장 |
| GET | `/api/v1/library/papers` | 저장 목록 |
| **Trends** | | |
| GET | `/api/v1/trends/keywords` | 키워드 트렌드 |
| GET | `/api/v1/trends/hot` | 핫 토픽 |

### 3.2 Request/Response Schemas

**Search API**
```json
// POST /api/v1/search
// Request
{
  "query": "cancer immunotherapy",
  "filters": {
    "year_from": 2020,
    "journal": "Nature"
  },
  "limit": 10
}

// Response
{
  "total": 247,
  "took_ms": 312,
  "results": [
    {
      "pmid": "38123456",
      "title": "Immune-related adverse events...",
      "abstract": "...",
      "relevance_score": 0.94,
      "authors": ["Kim S", "Lee J"],
      "journal": "Nature Medicine",
      "publication_date": "2024-03-15"
    }
  ]
}
```

**Chat API**
```json
// POST /api/v1/chat/query
// Request
{
  "question": "What are the latest methods to reduce CRISPR off-target effects?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}

// Response
{
  "answer": "Based on recent research, there are several approaches...",
  "sources": [
    {
      "pmid": "38123456",
      "title": "High-fidelity CRISPR-Cas9...",
      "relevance": 0.95,
      "excerpt": "SpCas9-HF1 variants showed..."
    }
  ],
  "confidence": 0.92,
  "processing_time_ms": 1823
}
```

---

## 4. Core Components (핵심 컴포넌트)

### 4.1 RAG Pipeline

```
┌──────────────────────────────────────────────────────────┐
│                     RAG Pipeline                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. Query Input                                          │
│     "CRISPR off-target 효과를 줄이는 방법?"               │
│                        │                                 │
│                        ▼                                 │
│  2. Query Embedding (PubMedBERT)                        │
│     [0.12, -0.34, 0.56, ...] (768 dims)                 │
│                        │                                 │
│                        ▼                                 │
│  3. Vector Search (Qdrant)                              │
│     - Cosine Similarity                                  │
│     - Top-K = 10                                         │
│                        │                                 │
│                        ▼                                 │
│  4. Re-ranking (CrossEncoder)                           │
│     - ms-marco-MiniLM                                    │
│     - Select Top-5                                       │
│                        │                                 │
│                        ▼                                 │
│  5. Context Building                                     │
│     [Paper 1] PMID: 123 ...                             │
│     [Paper 2] PMID: 456 ...                             │
│                        │                                 │
│                        ▼                                 │
│  6. LLM Generation (GPT-4)                              │
│     - System Prompt + Context + Question                 │
│     - Temperature: 0.1                                   │
│                        │                                 │
│                        ▼                                 │
│  7. Response Validation                                  │
│     - PMID Citation Check                                │
│     - Confidence Score                                   │
│                        │                                 │
│                        ▼                                 │
│  8. Output                                               │
│     Answer + Sources + Confidence                        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 4.2 Data Collection Pipeline

```
┌──────────────────────────────────────────────────────────┐
│                 Data Collection Pipeline                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. Scheduled Trigger (UTC 02:00)                       │
│                        │                                 │
│                        ▼                                 │
│  2. PubMed API Search                                    │
│     - Keywords: ["CAR-T", "CRISPR", "immunotherapy"]    │
│     - Date: Yesterday                                    │
│     - Rate Limit: 10 req/sec                            │
│                        │                                 │
│                        ▼                                 │
│  3. Fetch Metadata                                       │
│     - Title, Abstract, Authors                           │
│     - Journal, Keywords, MeSH                            │
│                        │                                 │
│                        ▼                                 │
│  4. Store in PostgreSQL                                  │
│                        │                                 │
│                        ▼                                 │
│  5. Trigger Embedding Task (Celery)                     │
│                        │                                 │
│                        ▼                                 │
│  6. Text Chunking                                        │
│     - 512 tokens, 50 overlap                            │
│                        │                                 │
│                        ▼                                 │
│  7. Generate Embeddings (PubMedBERT)                    │
│                        │                                 │
│                        ▼                                 │
│  8. Store in Qdrant                                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 5. Project Structure (프로젝트 구조)

```
bio-rag/
├── frontend/                      # React Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/           # Button, Input, Modal
│   │   │   ├── search/           # SearchBar, SearchResults
│   │   │   ├── chat/             # ChatWindow, Message
│   │   │   └── layout/           # Header, Sidebar
│   │   ├── pages/
│   │   │   ├── SearchPage.tsx
│   │   │   ├── ChatPage.tsx
│   │   │   └── PaperDetailPage.tsx
│   │   ├── hooks/                # useSearch, useChat
│   │   ├── services/             # API clients
│   │   ├── store/                # Zustand stores
│   │   └── types/                # TypeScript types
│   ├── package.json
│   └── vite.config.ts
│
├── backend/                       # FastAPI Backend
│   ├── src/
│   │   ├── api/v1/               # API routers
│   │   │   ├── auth.py
│   │   │   ├── search.py
│   │   │   └── chat.py
│   │   ├── core/                 # Config, DB, Security
│   │   ├── models/               # SQLAlchemy models
│   │   ├── schemas/              # Pydantic schemas
│   │   ├── services/
│   │   │   ├── collector/        # PubMed API
│   │   │   ├── embedding/        # PubMedBERT
│   │   │   ├── rag/              # RAG Service
│   │   │   ├── search/           # Semantic Search
│   │   │   └── storage/          # Vector Store
│   │   ├── tasks/                # Celery tasks
│   │   └── main.py
│   ├── tests/
│   ├── migrations/
│   └── requirements.txt
│
├── infra/
│   ├── docker/
│   │   ├── docker-compose.yml
│   │   └── Dockerfile.*
│   └── terraform/
│
└── .specify/
    ├── memory/constitution.md
    └── specs/001-bio-rag-mvp/
        ├── spec.md
        ├── plan.md
        └── tasks.md
```

---

## 6. Environment Configuration

### 6.1 환경 변수

```bash
# .env.example

# Application
APP_NAME=bio-rag
APP_ENV=development
DEBUG=true

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/bio_rag

# Redis
REDIS_URL=redis://localhost:6379/0

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# External APIs
PUBMED_API_KEY=your_pubmed_api_key
OPENAI_API_KEY=your_openai_api_key

# Security
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440

# CORS
CORS_ORIGINS=["http://localhost:3000"]

# Celery
CELERY_BROKER_URL=redis://localhost:6379/1
```

### 6.2 Docker Compose

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: bio_rag
      POSTGRES_PASSWORD: password
      POSTGRES_DB: bio_rag
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
      - qdrant

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  postgres_data:
  qdrant_data:
```

---

## 7. Performance Requirements

| Metric | Target | Strategy |
|--------|--------|----------|
| AI Chat Response | < 5s | Caching, Streaming |
| Search Response | < 1s | Vector Index, Caching |
| Page Load | < 1s | CDN, Code Splitting |
| Concurrent Users | 500 | Horizontal Scaling |
| Cache Hit Rate | > 60% | Redis TTL Strategy |

---

## 8. Security Measures

| Area | Implementation |
|------|----------------|
| Authentication | JWT (24h expiry) |
| Password | bcrypt hashing |
| API Keys | Environment variables |
| Transport | TLS 1.3 |
| Storage | AES-256 |
| CORS | Whitelist origins |
| Rate Limiting | 100 req/min per user |

---

## 9. Monitoring & Logging

```yaml
# Prometheus metrics
metrics:
  - request_count
  - request_latency_seconds
  - error_rate
  - llm_api_cost
  - cache_hit_rate

# Logging
logging:
  level: INFO
  format: JSON
  retention: 30 days
```

---

## 10. Timeline Summary

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Setup | Project structure, Docker, CI/CD |
| 2 | Data | PubMed collector, DB schema |
| 3 | Embedding | PubMedBERT, Qdrant, Chunking |
| 4 | RAG | RAG service, Validation |
| 5 | Search | Semantic search, Recommendations |
| 6 | Frontend | Search, Chat, Paper pages |
| 7 | Deploy | Testing, AWS deployment |

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 1.0 | 2024.12 | 초기 Plan 작성 |
