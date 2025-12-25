# Bio-RAG Implementation Guide (구현 가이드)

> 이 문서는 Tasks.md의 작업을 실행할 때 참조하는 상세 구현 가이드입니다.
> `/speckit.implement` 명령 실행 시 이 가이드를 따릅니다.

---

## 1. Quick Start (빠른 시작)

### 1.1 프로젝트 초기화 명령

```bash
# 1. 프로젝트 디렉토리 생성
mkdir bio-rag && cd bio-rag

# 2. 백엔드 구조 생성
mkdir -p backend/src/{api/v1,core,models,schemas,services/{collector,embedding,rag,search,storage},tasks}
mkdir -p backend/tests/{unit,integration}
mkdir -p backend/migrations/versions

# 3. 프론트엔드 구조 생성 (Vite + React + TypeScript)
npm create vite@latest frontend -- --template react-ts
cd frontend && npm install

# 4. 인프라 구조 생성
mkdir -p infra/{docker,kubernetes,terraform}

# 5. 문서 및 설정
mkdir docs .specify/{memory,specs,scripts,templates}
```

### 1.2 Docker 환경 실행

```bash
# 개발 환경 전체 실행
docker-compose -f infra/docker/docker-compose.yml up -d

# 개별 서비스 확인
docker-compose ps

# 로그 확인
docker-compose logs -f backend
```

---

## 2. Backend Core Implementation

### 2.1 FastAPI 앱 설정

**파일: `/backend/src/main.py`**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.core.config import settings
from src.api.v1 import auth, search, chat, library, trends

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Bio-RAG API...")
    yield
    logger.info("Shutting down Bio-RAG API...")

app = FastAPI(
    title="Bio-RAG API",
    description="AI-powered biomedical research platform",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(search.router, prefix="/api/v1", tags=["Search"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

### 2.2 환경 설정

**파일: `/backend/src/core/config.py`**
```python
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    APP_NAME: str = "bio-rag"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str
    
    # Redis
    REDIS_URL: str
    
    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    
    # APIs
    PUBMED_API_KEY: str
    OPENAI_API_KEY: str
    
    # Security
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440
    
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 2.3 데이터베이스 연결

**파일: `/backend/src/core/database.py`**
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from src.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=20,
    echo=settings.DEBUG
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
```

---

## 3. Core Services Implementation

### 3.1 PubMed Collector

**파일: `/backend/src/services/collector/pubmed_collector.py`**
```python
from typing import List, Optional, Tuple
from datetime import datetime
import asyncio
import httpx
import xml.etree.ElementTree as ET
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class PaperMetadata(BaseModel):
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: Optional[datetime] = None
    doi: Optional[str] = None
    keywords: List[str] = []
    mesh_terms: List[str] = []

class PubMedCollector:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._semaphore = asyncio.Semaphore(10)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10))
    async def _api_call(self, endpoint: str, params: dict) -> str:
        async with self._semaphore:
            params["api_key"] = self.api_key
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.BASE_URL}{endpoint}",
                    params=params,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.text
    
    async def search_papers(
        self, query: str, max_results: int = 100
    ) -> List[PaperMetadata]:
        # Search
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        
        import json
        response = await self._api_call("esearch.fcgi", search_params)
        result = json.loads(response)
        pmids = result.get("esearchresult", {}).get("idlist", [])
        
        if not pmids:
            return []
        
        # Fetch details
        return await self.batch_fetch(pmids)
    
    async def batch_fetch(self, pmids: List[str]) -> List[PaperMetadata]:
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml"
        }
        
        xml_response = await self._api_call("efetch.fcgi", fetch_params)
        return self._parse_xml(xml_response)
    
    def _parse_xml(self, xml_text: str) -> List[PaperMetadata]:
        papers = []
        root = ET.fromstring(xml_text)
        
        for article in root.findall(".//PubmedArticle"):
            try:
                pmid = article.find(".//PMID").text
                title = article.find(".//ArticleTitle").text or ""
                
                abstract_parts = []
                for ab in article.findall(".//AbstractText"):
                    if ab.text:
                        abstract_parts.append(ab.text)
                abstract = " ".join(abstract_parts)
                
                authors = []
                for author in article.findall(".//Author"):
                    ln = author.find("LastName")
                    if ln is not None and ln.text:
                        authors.append(ln.text)
                
                journal = article.find(".//Journal/Title")
                journal = journal.text if journal is not None else ""
                
                papers.append(PaperMetadata(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    journal=journal
                ))
            except Exception as e:
                logger.warning(f"Parse error: {e}")
        
        return papers
```

### 3.2 Embedding Generator

**파일: `/backend/src/services/embedding/generator.py`**
```python
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    EMBEDDING_DIM = 768
    
    def __init__(self, device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.model.eval()
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model.to(self.device)
        logger.info(f"Model loaded on {self.device}")
    
    def encode(self, text: str, max_length: int = 512) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.squeeze()
    
    def batch_encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embs = [self.encode(t) for t in batch]
            embeddings.extend(batch_embs)
        return np.array(embeddings)
```

### 3.3 Vector Store (Qdrant)

**파일: `/backend/src/services/storage/vector_store.py`**
```python
from typing import List, Optional, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    COLLECTION_NAME = "biomedical_papers"
    VECTOR_SIZE = 768
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection()
    
    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        if not any(c.name == self.COLLECTION_NAME for c in collections):
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
    
    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        points = [
            models.PointStruct(
                id=chunk['id'],
                vector=emb.tolist(),
                payload={"text": chunk['text'], **chunk.get('metadata', {})}
            )
            for chunk, emb in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)
    
    def search(
        self, query_embedding: np.ndarray, top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        query_filter = None
        if filter_dict:
            query_filter = models.Filter(must=[
                models.FieldCondition(key=k, match=models.MatchValue(value=v))
                for k, v in filter_dict.items()
            ])
        
        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter
        )
        
        return [
            {
                "id": str(r.id),
                "text": r.payload.get("text", ""),
                "score": r.score,
                "metadata": {k: v for k, v in r.payload.items() if k != "text"}
            }
            for r in results
        ]
```

### 3.4 RAG Service

**파일: `/backend/src/services/rag/service.py`**
```python
from typing import List
from pydantic import BaseModel
import openai
from sentence_transformers import CrossEncoder
import re
import time
import logging

logger = logging.getLogger(__name__)

class RAGResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float
    processing_time_ms: int

class RAGService:
    SYSTEM_PROMPT = """You are an expert biomedical researcher assistant.

RULES:
1. Only use information from the provided context
2. Cite sources using [PMID: xxxxx] format
3. If context is insufficient, say "I cannot find sufficient information"
4. Be precise and factual"""
    
    def __init__(self, vector_store, embedding_gen, openai_key: str, model: str = "gpt-4"):
        self.vector_store = vector_store
        self.embedding_gen = embedding_gen
        self.openai_client = openai.AsyncOpenAI(api_key=openai_key)
        self.model = model
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    
    async def query(self, question: str, top_k: int = 5, rerank: bool = True) -> RAGResponse:
        start = time.time()
        
        # 1. Embed query
        query_emb = self.embedding_gen.encode(question)
        
        # 2. Search
        results = self.vector_store.search(query_emb, top_k * 2 if rerank else top_k)
        
        if not results:
            return RAGResponse(
                answer="No relevant papers found.",
                sources=[],
                confidence=0.0,
                processing_time_ms=int((time.time() - start) * 1000)
            )
        
        # 3. Rerank
        if rerank:
            results = self._rerank(question, results)[:top_k]
        
        # 4. Build context
        context = self._build_context(results)
        
        # 5. Generate answer
        answer = await self._generate(question, context)
        
        # 6. Validate
        confidence = self._validate(answer, results)
        
        return RAGResponse(
            answer=answer,
            sources=self._format_sources(results),
            confidence=confidence,
            processing_time_ms=int((time.time() - start) * 1000)
        )
    
    def _build_context(self, results: List[dict]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            pmid = r['metadata'].get('pmid', 'N/A')
            title = r['metadata'].get('title', 'N/A')
            parts.append(f"[Paper {i}] PMID: {pmid}\nTitle: {title}\nContent: {r['text']}\n")
        return "\n".join(parts)
    
    def _rerank(self, question: str, results: List[dict]) -> List[dict]:
        pairs = [(question, r['text']) for r in results]
        scores = self.reranker.predict(pairs)
        return [r for _, r in sorted(zip(scores, results), key=lambda x: x[0], reverse=True)]
    
    async def _generate(self, question: str, context: str) -> str:
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Error generating response."
    
    def _validate(self, answer: str, sources: List[dict]) -> float:
        cited = re.findall(r'PMID:\s*(\d+)', answer)
        source_pmids = [s['metadata'].get('pmid', '') for s in sources]
        valid = all(p in source_pmids for p in cited)
        return 0.9 if valid and cited else 0.5
    
    def _format_sources(self, results: List[dict]) -> List[dict]:
        return [
            {
                "pmid": r['metadata'].get('pmid', ''),
                "title": r['metadata'].get('title', ''),
                "relevance": round(r['score'], 3),
                "excerpt": r['text'][:200] + "..."
            }
            for r in results
        ]
```

---

## 4. API Endpoints

### 4.1 Search API

**파일: `/backend/src/api/v1/search.py`**
```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    filters: Optional[dict] = None

class PaperResult(BaseModel):
    pmid: str
    title: str
    abstract: str
    relevance_score: float
    authors: List[str] = []
    journal: str = ""

class SearchResponse(BaseModel):
    total: int
    took_ms: int
    results: List[PaperResult]

@router.post("/search", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    """의미 기반 논문 검색"""
    # TODO: Implement with SemanticSearchService
    return SearchResponse(total=0, took_ms=0, results=[])

@router.get("/papers/{pmid}")
async def get_paper(pmid: str):
    """논문 상세 조회"""
    # TODO: Implement
    return {"pmid": pmid}

@router.get("/papers/{pmid}/similar")
async def get_similar_papers(pmid: str, limit: int = 5):
    """유사 논문 추천"""
    # TODO: Implement
    return {"papers": []}
```

### 4.2 Chat API

**파일: `/backend/src/api/v1/chat.py`**
```python
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class ChatQueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    context_pmids: List[str] = []

class ChatQueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float
    processing_time_ms: int

@router.post("/query", response_model=ChatQueryResponse)
async def chat_query(request: ChatQueryRequest):
    """AI 질의응답"""
    # TODO: Implement with RAGService
    return ChatQueryResponse(
        answer="Sample answer",
        sources=[],
        confidence=0.9,
        processing_time_ms=1000
    )

@router.get("/sessions")
async def get_sessions():
    """대화 세션 목록"""
    return {"sessions": []}

@router.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    """세션 메시지 조회"""
    return {"messages": []}
```

---

## 5. Frontend Implementation

### 5.1 프로젝트 설정

**파일: `/frontend/package.json`**
```json
{
  "name": "bio-rag-frontend",
  "version": "1.0.0",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "@tanstack/react-query": "^5.8.0",
    "zustand": "^4.4.0",
    "axios": "^1.6.0",
    "recharts": "^2.10.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "tailwindcss": "^3.3.0",
    "typescript": "^5.3.0",
    "vite": "^5.0.0"
  }
}
```

### 5.2 API Service

**파일: `/frontend/src/services/api.ts`**
```typescript
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for auth token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Search API
export const searchPapers = async (query: string, limit = 10) => {
  const response = await api.post('/search', { query, limit });
  return response.data;
};

// Chat API
export const sendChatQuery = async (question: string, sessionId?: string) => {
  const response = await api.post('/chat/query', { question, session_id: sessionId });
  return response.data;
};

// Paper API
export const getPaperDetail = async (pmid: string) => {
  const response = await api.get(`/papers/${pmid}`);
  return response.data;
};

export const getSimilarPapers = async (pmid: string) => {
  const response = await api.get(`/papers/${pmid}/similar`);
  return response.data;
};
```

### 5.3 Search Page

**파일: `/frontend/src/pages/SearchPage.tsx`**
```tsx
import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { searchPapers } from '../services/api';

interface PaperResult {
  pmid: string;
  title: string;
  abstract: string;
  relevance_score: number;
  authors: string[];
  journal: string;
}

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  
  const { data, isLoading, error } = useQuery({
    queryKey: ['search', searchTerm],
    queryFn: () => searchPapers(searchTerm),
    enabled: !!searchTerm,
  });
  
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setSearchTerm(query);
  };
  
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">논문 검색</h1>
      
      {/* Search Bar */}
      <form onSubmit={handleSearch} className="mb-8">
        <div className="flex gap-4">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="검색어를 입력하세요 (예: cancer immunotherapy)"
            className="flex-1 px-4 py-3 border rounded-lg focus:outline-none focus:ring-2"
          />
          <button
            type="submit"
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            검색
          </button>
        </div>
      </form>
      
      {/* Results */}
      {isLoading && <p>검색 중...</p>}
      
      {error && <p className="text-red-500">검색 중 오류가 발생했습니다.</p>}
      
      {data && (
        <div>
          <p className="text-gray-600 mb-4">
            {data.total}건의 결과 ({data.took_ms}ms)
          </p>
          
          <div className="space-y-4">
            {data.results.map((paper: PaperResult) => (
              <div key={paper.pmid} className="p-6 border rounded-lg hover:shadow-md">
                <div className="flex justify-between items-start mb-2">
                  <h3 className="text-lg font-semibold text-blue-600">
                    {paper.title}
                  </h3>
                  <span className="text-sm bg-green-100 text-green-800 px-2 py-1 rounded">
                    {Math.round(paper.relevance_score * 100)}%
                  </span>
                </div>
                <p className="text-sm text-gray-500 mb-2">
                  PMID: {paper.pmid} | {paper.journal}
                </p>
                <p className="text-gray-700 line-clamp-3">
                  {paper.abstract}
                </p>
                <div className="mt-4 flex gap-2">
                  <button className="text-sm text-blue-600 hover:underline">
                    상세보기
                  </button>
                  <button className="text-sm text-blue-600 hover:underline">
                    유사 논문
                  </button>
                  <button className="text-sm text-blue-600 hover:underline">
                    AI 질문
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

### 5.4 Chat Page

**파일: `/frontend/src/pages/ChatPage.tsx`**
```tsx
import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { sendChatQuery } from '../services/api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Array<{
    pmid: string;
    title: string;
    relevance: number;
  }>;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  
  const mutation = useMutation({
    mutationFn: sendChatQuery,
    onSuccess: (data) => {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          sources: data.sources,
        },
      ]);
    },
  });
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    setMessages((prev) => [...prev, { role: 'user', content: input }]);
    mutation.mutate(input);
    setInput('');
  };
  
  return (
    <div className="flex flex-col h-screen">
      <header className="p-4 border-b">
        <h1 className="text-xl font-bold">🤖 Bio-RAG AI Assistant</h1>
      </header>
      
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-3xl p-4 rounded-lg ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-800'
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
              
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-300">
                  <p className="text-sm font-semibold mb-2">📚 참고 문헌:</p>
                  {msg.sources.map((source, i) => (
                    <p key={i} className="text-sm">
                      [{i + 1}] PMID: {source.pmid} - {source.title}
                    </p>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        
        {mutation.isPending && (
          <div className="flex justify-start">
            <div className="bg-gray-100 p-4 rounded-lg">
              <p>답변 생성 중...</p>
            </div>
          </div>
        )}
      </div>
      
      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t">
        <div className="flex gap-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="질문을 입력하세요..."
            className="flex-1 px-4 py-3 border rounded-lg"
          />
          <button
            type="submit"
            disabled={mutation.isPending}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg disabled:opacity-50"
          >
            전송
          </button>
        </div>
      </form>
    </div>
  );
}
```

---

## 6. Docker Configuration

**파일: `/infra/docker/docker-compose.yml`**
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
    build:
      context: ../../backend
      dockerfile: ../infra/docker/Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://bio_rag:password@postgres:5432/bio_rag
      - REDIS_URL=redis://redis:6379/0
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    depends_on:
      - postgres
      - redis
      - qdrant

  frontend:
    build:
      context: ../../frontend
      dockerfile: ../infra/docker/Dockerfile.frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  postgres_data:
  qdrant_data:
```

**파일: `/infra/docker/Dockerfile.backend`**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 7. Testing Examples

**파일: `/backend/tests/unit/test_embedding.py`**
```python
import pytest
import numpy as np
from src.services.embedding.generator import EmbeddingGenerator

class TestEmbeddingGenerator:
    @pytest.fixture
    def generator(self):
        return EmbeddingGenerator(device='cpu')
    
    def test_encode_dimension(self, generator):
        text = "CRISPR gene editing"
        embedding = generator.encode(text)
        assert embedding.shape == (768,)
    
    def test_similar_texts_high_similarity(self, generator):
        text1 = "cancer immunotherapy"
        text2 = "immunotherapy for cancer"
        
        emb1 = generator.encode(text1)
        emb2 = generator.encode(text2)
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        assert similarity > 0.8
```

---

## 8. Execution Commands

```bash
# Backend 개발 서버
cd backend
pip install -r requirements.txt
uvicorn src.main:app --reload

# Frontend 개발 서버
cd frontend
npm install
npm run dev

# Docker 전체 실행
docker-compose -f infra/docker/docker-compose.yml up -d

# 테스트 실행
cd backend
pytest tests/ -v --cov=src

# 마이그레이션
cd backend
alembic upgrade head
```

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 1.0 | 2024.12 | 초기 Implementation Guide 작성 |
