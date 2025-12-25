# Bio-RAG

**Biomedical Research AI-Guided Analytics**

바이오 연구자를 위한 AI 기반 논문 분석 및 인사이트 도출 플랫폼

## Features

- **AI 논문 Q&A**: RAG 기반 자연어 질의응답
- **의미 기반 검색**: 벡터 유사도 기반 논문 검색
- **유사 논문 추천**: 코사인 유사도 기반 추천
- **연구 트렌드 대시보드**: 시계열 키워드 분석

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18 + TypeScript + TailwindCSS |
| Backend | Python 3.11 + FastAPI |
| AI/ML | LangChain + PubMedBERT + OpenAI GPT-4 |
| Database | PostgreSQL + Qdrant (Vector DB) + Redis |
| Infra | Docker + AWS |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose
- OpenAI API Key
- PubMed API Key

### Development Setup

```bash
# 1. Clone and setup
git clone <repository-url>
cd bio-rag

# 2. Start with Docker
make docker-up

# 3. Or run locally
make setup
make dev
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp backend/.env.example backend/.env
```

## Project Structure

```
bio-rag/
├── backend/                # FastAPI Backend
│   ├── src/
│   │   ├── api/v1/        # API endpoints
│   │   ├── core/          # Config, DB, Security
│   │   ├── models/        # SQLAlchemy models
│   │   ├── schemas/       # Pydantic schemas
│   │   ├── services/      # Business logic
│   │   └── tasks/         # Celery tasks
│   ├── tests/
│   └── migrations/
├── frontend/              # React Frontend
│   └── src/
├── infra/                 # Infrastructure
│   ├── docker/
│   └── terraform/
└── docs/
```

## API Documentation

After starting the backend, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

MIT License
