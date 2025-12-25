# Bio-RAG Tasks (작업 목록)

> 이 문서는 Bio-RAG 플랫폼 구현을 위한 상세 작업 목록입니다.
> Plan.md의 기술 설계를 기반으로 실행 가능한 단위 작업으로 분해되었습니다.
> `/speckit.implement` 명령으로 실행됩니다.

---

## Task Overview (작업 개요)

### 프로젝트 통계
- **총 작업 수**: 47개
- **예상 소요 시간**: 7주 (MVP 기준)
- **우선순위**: P0 (필수) > P1 (중요) > P2 (선택)

### 작업 상태 범례
- `[ ]` 대기 중
- `[P]` 병렬 실행 가능
- `[D]` 의존성 있음 (선행 작업 완료 필요)
- `[✓]` 완료

---

## Phase 1: 프로젝트 설정 (Week 1)

### 1.1 개발 환경 설정

#### Task 1.1.1: 프로젝트 구조 생성
```
상태: [ ]
우선순위: P0
예상 시간: 2시간
파일 경로: /bio-rag/

작업 내용:
1. 루트 디렉토리 구조 생성
   - /frontend (React)
   - /backend (FastAPI)
   - /infra (Docker, K8s)
   - /docs
   - /.specify

2. Git 저장소 초기화
   - .gitignore 설정
   - README.md 작성
   - LICENSE 추가

3. Makefile 작성
   - make setup
   - make dev
   - make test
   - make build

수용 기준:
- [ ] 모든 디렉토리가 생성됨
- [ ] git init 완료
- [ ] make help 명령 동작
```

#### Task 1.1.2: Backend 프로젝트 초기화 [P]
```
상태: [ ]
우선순위: P0
예상 시간: 3시간
파일 경로: /bio-rag/backend/

작업 내용:
1. Python 프로젝트 설정
   - pyproject.toml 작성
   - requirements.txt 생성
   - requirements-dev.txt 생성

2. FastAPI 앱 기본 구조
   - src/main.py (앱 진입점)
   - src/core/config.py (환경 설정)
   - src/api/v1/__init__.py

3. 환경 변수 설정
   - .env.example 작성
   - config.py에 Pydantic Settings 적용

수용 기준:
- [ ] uvicorn src.main:app --reload 실행 가능
- [ ] /health 엔드포인트 응답
- [ ] 환경 변수 로드 정상
```

#### Task 1.1.3: Frontend 프로젝트 초기화 [P]
```
상태: [ ]
우선순위: P0
예상 시간: 3시간
파일 경로: /bio-rag/frontend/

작업 내용:
1. Vite + React + TypeScript 설정
   - npm create vite@latest
   - TypeScript strict mode
   - Path alias 설정

2. TailwindCSS 설정
   - tailwind.config.js
   - postcss.config.js
   - 기본 스타일 적용

3. 프로젝트 구조
   - /src/components
   - /src/pages
   - /src/hooks
   - /src/services
   - /src/types

수용 기준:
- [ ] npm run dev 실행 가능
- [ ] TypeScript 컴파일 에러 없음
- [ ] TailwindCSS 클래스 적용됨
```

#### Task 1.1.4: Docker 개발 환경 설정
```
상태: [ ]
우선순위: P0
예상 시간: 4시간
파일 경로: /bio-rag/infra/docker/
의존성: Task 1.1.2, 1.1.3

작업 내용:
1. Dockerfile 작성
   - Dockerfile.backend (Python 3.11)
   - Dockerfile.frontend (Node 20)

2. docker-compose.yml 작성
   - PostgreSQL 15
   - Redis 7
   - Qdrant
   - Backend API
   - Frontend

3. 볼륨 및 네트워크 설정

수용 기준:
- [ ] docker-compose up 으로 전체 스택 실행
- [ ] 서비스 간 통신 정상
- [ ] 데이터 영속성 유지
```

---

## Phase 2: 데이터 수집 파이프라인 (Week 2)

### 2.1 PubMed 데이터 수집

#### Task 2.1.1: PubMed API 클라이언트 구현
```
상태: [ ]
우선순위: P0
예상 시간: 6시간
파일 경로: /bio-rag/backend/src/services/collector/pubmed_collector.py

작업 내용:
1. PubMedCollector 클래스 구현
   - __init__(api_key)
   - async search_papers(query, max_results, date_range)
   - async fetch_abstract(pmid)
   - async batch_fetch(pmid_list)

2. Rate Limiting 구현
   - asyncio.Semaphore (10 req/sec)
   - 재시도 로직 (tenacity)

3. 에러 처리
   - PubMedAPIError 커스텀 예외
   - 로깅

수용 기준:
- [ ] "cancer immunotherapy" 검색 시 결과 반환
- [ ] Rate limit 준수 (10 req/sec)
- [ ] 3회 재시도 후 실패 시 예외 발생
```

#### Task 2.1.2: 논문 메타데이터 모델 정의 [P]
```
상태: [ ]
우선순위: P0
예상 시간: 2시간
파일 경로: 
  - /bio-rag/backend/src/schemas/paper.py
  - /bio-rag/backend/src/models/paper.py

작업 내용:
1. Pydantic 스키마
   - PaperMetadata
   - PaperCreate
   - PaperResponse

2. SQLAlchemy 모델
   - Paper 테이블
   - 인덱스 설정

수용 기준:
- [ ] 스키마 검증 동작
- [ ] DB 마이그레이션 성공
```

#### Task 2.1.3: 데이터베이스 설정
```
상태: [ ]
우선순위: P0
예상 시간: 4시간
파일 경로: /bio-rag/backend/src/core/database.py
의존성: Task 2.1.2

작업 내용:
1. SQLAlchemy 연결 설정
   - async 세션 팩토리
   - Connection pooling

2. Alembic 마이그레이션 설정
   - alembic.ini
   - migrations/env.py

3. 초기 마이그레이션
   - users, papers, chunks 테이블

수용 기준:
- [ ] alembic upgrade head 성공
- [ ] DB 연결 테스트 통과
```

#### Task 2.1.4: Celery 배치 작업 설정
```
상태: [ ]
우선순위: P0
예상 시간: 4시간
파일 경로: /bio-rag/backend/src/tasks/

작업 내용:
1. Celery 앱 설정
   - celery.py (브로커 연결)
   - __init__.py

2. 크롤링 태스크
   - crawler.py
     - daily_paper_crawl()
     - process_paper_async()

3. Celery Beat 스케줄
   - UTC 02:00 일일 크롤링

수용 기준:
- [ ] celery worker 실행 가능
- [ ] celery beat 스케줄 동작
- [ ] 태스크 실행 및 결과 확인
```

---

## Phase 3: 임베딩 및 벡터 DB (Week 3)

### 3.1 임베딩 생성

#### Task 3.1.1: 임베딩 생성기 구현
```
상태: [ ]
우선순위: P0
예상 시간: 6시간
파일 경로: /bio-rag/backend/src/services/embedding/generator.py

작업 내용:
1. EmbeddingGenerator 클래스
   - __init__(model_name, device)
   - encode(text) -> np.ndarray (768,)
   - batch_encode(texts, batch_size) -> np.ndarray

2. PubMedBERT 모델 로드
   - HuggingFace Transformers
   - GPU 지원 (선택적)

3. 성능 최적화
   - 배치 처리
   - 토큰 캐싱

수용 기준:
- [ ] 단일 텍스트 임베딩 생성
- [ ] 배치 임베딩 생성
- [ ] 768차원 벡터 반환
```

#### Task 3.1.2: 텍스트 청킹 유틸리티 구현 [P]
```
상태: [ ]
우선순위: P0
예상 시간: 4시간
파일 경로: /bio-rag/backend/src/services/embedding/chunker.py

작업 내용:
1. 청킹 함수
   - chunk_by_tokens(text, chunk_size=512, overlap=50)
   - chunk_by_section(paper)

2. 텍스트 전처리
   - clean_text(text)
   - 특수문자 제거
   - 참조 번호 제거

3. Chunk 데이터 모델
   - Pydantic 스키마
   - SQLAlchemy 모델

수용 기준:
- [ ] 긴 텍스트를 청크로 분할
- [ ] 오버랩 적용 확인
- [ ] 섹션 기반 청킹 동작
```

### 3.2 벡터 데이터베이스

#### Task 3.2.1: Qdrant 클라이언트 구현
```
상태: [ ]
우선순위: P0
예상 시간: 6시간
파일 경로: /bio-rag/backend/src/services/storage/vector_store.py
의존성: Task 3.1.1

작업 내용:
1. VectorStore 클래스
   - __init__(host, port)
   - _ensure_collection()
   - add_documents(chunks, embeddings)
   - search(query_embedding, top_k, filter_dict)
   - delete(ids)

2. 컬렉션 설정
   - 코사인 유사도
   - 768차원 벡터

3. 메타데이터 필터링
   - section, pmid 등

수용 기준:
- [ ] 문서 추가/검색/삭제 동작
- [ ] 메타데이터 필터링 동작
- [ ] 상위 K개 결과 반환
```

#### Task 3.2.2: 임베딩 파이프라인 통합
```
상태: [ ]
우선순위: P0
예상 시간: 4시간
파일 경로: /bio-rag/backend/src/tasks/embedding.py
의존성: Task 3.1.1, 3.1.2, 3.2.1

작업 내용:
1. Celery 태스크
   - process_paper_embeddings(paper_id)
   - batch_process_embeddings(paper_ids)

2. 파이프라인 통합
   - 논문 → 청킹 → 임베딩 → Qdrant 저장

3. 진행 상태 추적

수용 기준:
- [ ] 논문 저장 시 자동 임베딩 생성
- [ ] Qdrant에 저장 확인
- [ ] 에러 발생 시 재시도
```

---

## Phase 4: RAG 서비스 (Week 4)

### 4.1 RAG 파이프라인

#### Task 4.1.1: RAG 서비스 핵심 구현
```
상태: [ ]
우선순위: P0
예상 시간: 8시간
파일 경로: /bio-rag/backend/src/services/rag/service.py
의존성: Task 3.1.1, 3.2.1

작업 내용:
1. RAGService 클래스
   - __init__(vector_store, embedding_gen, openai_key)
   - async query(question, top_k, rerank)

2. RAG 파이프라인
   - Query Embedding
   - Vector Search
   - Context Building
   - LLM Generation
   - Response Validation

3. 프롬프트 템플릿
   - SYSTEM_PROMPT
   - Few-shot 예시

수용 기준:
- [ ] 질문에 대한 답변 생성
- [ ] 출처(PMID) 포함
- [ ] 5초 이내 응답
```

#### Task 4.1.2: 리랭킹 모듈 구현 [P]
```
상태: [ ]
우선순위: P1
예상 시간: 4시간
파일 경로: /bio-rag/backend/src/services/rag/reranker.py

작업 내용:
1. CrossEncoder 리랭커
   - ms-marco-MiniLM-L-12-v2 모델
   - rerank(question, results)

2. 성능 최적화
   - 배치 처리

수용 기준:
- [ ] 검색 결과 재정렬
- [ ] 관련도 점수 개선
```

#### Task 4.1.3: 할루시네이션 검증 모듈
```
상태: [ ]
우선순위: P0
예상 시간: 4시간
파일 경로: /bio-rag/backend/src/services/rag/validator.py
의존성: Task 4.1.1

작업 내용:
1. ResponseValidator 클래스
   - validate(answer, sources)
   - _extract_citations(answer)
   - _check_consistency(answer, sources)

2. 검증 결과
   - is_valid: bool
   - confidence: float
   - warnings: List[str]

수용 기준:
- [ ] 인용된 PMID 검증
- [ ] 신뢰도 점수 계산
- [ ] 할루시네이션 탐지
```

#### Task 4.1.4: 캐싱 레이어 구현
```
상태: [ ]
우선순위: P1
예상 시간: 4시간
파일 경로: /bio-rag/backend/src/services/rag/cache.py

작업 내용:
1. Redis 캐시
   - 질문+컨텍스트 해시 키
   - TTL: 7일

2. 캐시 전략
   - get_cached_or_generate()
   - invalidate()

수용 기준:
- [ ] 동일 질문 캐시 히트
- [ ] TTL 만료 후 재생성
- [ ] LLM API 호출 60% 감소
```

---

## Phase 5: 검색 및 추천 (Week 5)

### 5.1 의미 기반 검색

#### Task 5.1.1: 검색 서비스 구현
```
상태: [ ]
우선순위: P0
예상 시간: 6시간
파일 경로: /bio-rag/backend/src/services/search/semantic_search.py
의존성: Task 3.1.1, 3.2.1

작업 내용:
1. SemanticSearchService 클래스
   - search(query, filters, top_k)
   - _expand_query(query) - 동의어 확장
   - _apply_filters(results, filters)
   - _aggregate_chunks(results)

2. 검색 필터
   - 연도, 저널, 저자

3. 결과 집계
   - 청크 → 논문 그룹핑
   - 관련도 점수 계산

수용 기준:
- [ ] 자연어 검색 동작
- [ ] 필터 적용 동작
- [ ] 1초 이내 응답
```

#### Task 5.1.2: 유사 논문 추천 구현
```
상태: [ ]
우선순위: P1
예상 시간: 4시간
파일 경로: /bio-rag/backend/src/services/search/recommender.py
의존성: Task 5.1.1

작업 내용:
1. PaperRecommender 클래스
   - recommend_similar(paper_id, top_k)
   - _content_based(paper_id, top_k)
   - _get_paper_embedding(paper_id)

2. 코사인 유사도 계산

수용 기준:
- [ ] 논문에서 유사 논문 5개 추천
- [ ] 유사도 점수 포함
```

### 5.2 검색 API

#### Task 5.2.1: Search API 엔드포인트
```
상태: [ ]
우선순위: P0
예상 시간: 4시간
파일 경로: /bio-rag/backend/src/api/v1/search.py
의존성: Task 5.1.1, 5.1.2

작업 내용:
1. API 라우터
   - POST /api/v1/search
   - GET /api/v1/papers/{pmid}
   - GET /api/v1/papers/{pmid}/similar

2. 요청/응답 스키마
   - SearchRequest
   - SearchResponse
   - PaperDetailResponse

수용 기준:
- [ ] API 문서 자동 생성
- [ ] 입력 검증 동작
- [ ] 적절한 에러 응답
```

---

## Phase 6: 프론트엔드 개발 (Week 6)

### 6.1 공통 컴포넌트

#### Task 6.1.1: 레이아웃 컴포넌트 [P]
```
상태: [ ]
우선순위: P0
예상 시간: 4시간
파일 경로: /bio-rag/frontend/src/components/layout/

작업 내용:
1. 컴포넌트 구현
   - Header.tsx (로고, 네비게이션, 사용자 메뉴)
   - Sidebar.tsx (메인 메뉴)
   - Footer.tsx

2. 반응형 디자인
   - 모바일 메뉴

수용 기준:
- [ ] 데스크톱/모바일 레이아웃 동작
- [ ] 네비게이션 동작
```

#### Task 6.1.2: 공통 UI 컴포넌트 [P]
```
상태: [ ]
우선순위: P0
예상 시간: 6시간
파일 경로: /bio-rag/frontend/src/components/common/

작업 내용:
1. 기본 컴포넌트
   - Button.tsx
   - Input.tsx
   - Modal.tsx
   - Card.tsx
   - Loading.tsx
   - Badge.tsx

2. Tailwind 스타일링

수용 기준:
- [ ] 일관된 디자인 시스템
- [ ] 접근성 (WCAG 2.1 AA)
```

### 6.2 핵심 페이지

#### Task 6.2.1: 검색 페이지 구현
```
상태: [ ]
우선순위: P0
예상 시간: 8시간
파일 경로: 
  - /bio-rag/frontend/src/pages/SearchPage.tsx
  - /bio-rag/frontend/src/components/search/
의존성: Task 6.1.1, 6.1.2

작업 내용:
1. 컴포넌트
   - SearchBar.tsx (검색 입력)
   - SearchFilters.tsx (필터 패널)
   - SearchResults.tsx (결과 목록)
   - PaperCard.tsx (논문 카드)

2. API 연동
   - useSearch 훅
   - React Query 캐싱

3. 페이지네이션

수용 기준:
- [ ] 검색 및 결과 표시
- [ ] 필터 적용 동작
- [ ] 관련도 점수 시각화
```

#### Task 6.2.2: AI 챗봇 페이지 구현
```
상태: [ ]
우선순위: P0
예상 시간: 10시간
파일 경로:
  - /bio-rag/frontend/src/pages/ChatPage.tsx
  - /bio-rag/frontend/src/components/chat/
의존성: Task 6.1.1, 6.1.2

작업 내용:
1. 컴포넌트
   - ChatWindow.tsx (메인 컨테이너)
   - MessageBubble.tsx (메시지 표시)
   - ChatInput.tsx (입력창)
   - SourceCard.tsx (출처 카드)

2. 상태 관리
   - Zustand chatStore
   - 대화 히스토리

3. API 연동
   - useChat 훅
   - 스트리밍 응답 (선택적)

수용 기준:
- [ ] 질문/답변 UI
- [ ] 출처 표시 및 링크
- [ ] 대화 히스토리 유지
```

#### Task 6.2.3: 논문 상세 페이지 구현
```
상태: [ ]
우선순위: P0
예상 시간: 6시간
파일 경로:
  - /bio-rag/frontend/src/pages/PaperDetailPage.tsx
  - /bio-rag/frontend/src/components/paper/

작업 내용:
1. 컴포넌트
   - PaperDetail.tsx (상세 정보)
   - SimilarPapers.tsx (유사 논문)
   - PaperActions.tsx (저장, 공유)

2. API 연동
   - 논문 상세 조회
   - 유사 논문 추천

수용 기준:
- [ ] 논문 정보 전체 표시
- [ ] 유사 논문 목록
- [ ] PDF 링크 (가능한 경우)
```

### 6.3 사용자 인증

#### Task 6.3.1: 인증 서비스 구현
```
상태: [ ]
우선순위: P0
예상 시간: 6시간
파일 경로: /bio-rag/backend/src/api/v1/auth.py

작업 내용:
1. API 엔드포인트
   - POST /auth/register
   - POST /auth/login
   - POST /auth/logout
   - POST /auth/refresh

2. JWT 토큰 관리
   - Access Token (24시간)
   - Refresh Token (7일)

3. 비밀번호 해싱
   - bcrypt

수용 기준:
- [ ] 회원가입 동작
- [ ] 로그인/로그아웃 동작
- [ ] 토큰 갱신 동작
```

#### Task 6.3.2: 로그인/회원가입 UI [P]
```
상태: [ ]
우선순위: P0
예상 시간: 4시간
파일 경로:
  - /bio-rag/frontend/src/pages/LoginPage.tsx
  - /bio-rag/frontend/src/pages/RegisterPage.tsx

작업 내용:
1. 로그인 폼
2. 회원가입 폼 (연구 분야 선택 포함)
3. 인증 상태 관리 (authStore)

수용 기준:
- [ ] 폼 검증 동작
- [ ] 에러 메시지 표시
- [ ] 로그인 후 리다이렉트
```

---

## Phase 7: 테스트 및 배포 (Week 7)

### 7.1 테스트

#### Task 7.1.1: Backend Unit 테스트
```
상태: [ ]
우선순위: P0
예상 시간: 8시간
파일 경로: /bio-rag/backend/tests/unit/

작업 내용:
1. 서비스 테스트
   - test_embedding_generator.py
   - test_rag_service.py
   - test_semantic_search.py
   - test_pubmed_collector.py

2. pytest 설정
   - conftest.py (fixtures)
   - pytest.ini

수용 기준:
- [ ] 80% 이상 커버리지
- [ ] 모든 테스트 통과
```

#### Task 7.1.2: Backend Integration 테스트 [P]
```
상태: [ ]
우선순위: P0
예상 시간: 6시간
파일 경로: /bio-rag/backend/tests/integration/

작업 내용:
1. API 테스트
   - test_search_api.py
   - test_chat_api.py
   - test_auth_api.py

2. DB 연동 테스트

수용 기준:
- [ ] API 엔드포인트 테스트 커버
- [ ] 에러 케이스 테스트
```

#### Task 7.1.3: Frontend 테스트 [P]
```
상태: [ ]
우선순위: P1
예상 시간: 4시간
파일 경로: /bio-rag/frontend/src/**/__tests__/

작업 내용:
1. 컴포넌트 테스트 (Vitest + React Testing Library)
2. 훅 테스트

수용 기준:
- [ ] 주요 컴포넌트 테스트
- [ ] 훅 테스트
```

### 7.2 배포

#### Task 7.2.1: CI/CD 파이프라인 설정
```
상태: [ ]
우선순위: P0
예상 시간: 6시간
파일 경로: /.github/workflows/

작업 내용:
1. GitHub Actions 워크플로우
   - ci.yml (테스트, 린트)
   - cd.yml (빌드, 배포)

2. Docker 이미지 빌드
3. AWS ECR 푸시

수용 기준:
- [ ] PR 시 자동 테스트
- [ ] main 머지 시 자동 배포
```

#### Task 7.2.2: 프로덕션 환경 설정
```
상태: [ ]
우선순위: P0
예상 시간: 8시간
파일 경로: /bio-rag/infra/

작업 내용:
1. Terraform 스크립트
   - ECS/EKS 클러스터
   - RDS PostgreSQL
   - ElastiCache Redis
   - S3

2. 환경 변수 관리
   - AWS Secrets Manager

3. 모니터링
   - Prometheus 메트릭
   - CloudWatch 로그

수용 기준:
- [ ] terraform apply 성공
- [ ] 서비스 헬스 체크 통과
- [ ] 로그 수집 동작
```

#### Task 7.2.3: 성능 테스트 및 최적화
```
상태: [ ]
우선순위: P1
예상 시간: 4시간

작업 내용:
1. 부하 테스트
   - Locust 또는 k6
   - 동시 사용자 500명 시뮬레이션

2. 성능 최적화
   - 캐시 히트율 확인
   - 슬로우 쿼리 최적화

수용 기준:
- [ ] AI 챗봇 응답 < 5초
- [ ] 검색 응답 < 1초
- [ ] 동시 500명 처리 가능
```

---

## Dependency Graph (의존성 그래프)

```
Phase 1 ─────────────────────────────────────────────────────────────
   │
   ├─ Task 1.1.1 (프로젝트 구조)
   │     │
   │     ├─ Task 1.1.2 (Backend 초기화) ──┬── Task 1.1.4 (Docker)
   │     │                                │
   │     └─ Task 1.1.3 (Frontend 초기화) ─┘
   │
Phase 2 ─────────────────────────────────────────────────────────────
   │
   ├─ Task 2.1.1 (PubMed 클라이언트)
   │     │
   │     └─ Task 2.1.2 (모델 정의) ── Task 2.1.3 (DB 설정) ── Task 2.1.4 (Celery)
   │
Phase 3 ─────────────────────────────────────────────────────────────
   │
   ├─ Task 3.1.1 (임베딩 생성기) ──┬── Task 3.2.1 (Qdrant) ── Task 3.2.2 (파이프라인)
   │                              │
   └─ Task 3.1.2 (청킹) ──────────┘
   │
Phase 4 ─────────────────────────────────────────────────────────────
   │
   └─ Task 4.1.1 (RAG 서비스) ──┬── Task 4.1.3 (검증)
                               │
                               ├── Task 4.1.2 (리랭킹)
                               │
                               └── Task 4.1.4 (캐싱)
   │
Phase 5 ─────────────────────────────────────────────────────────────
   │
   └─ Task 5.1.1 (검색 서비스) ── Task 5.1.2 (추천) ── Task 5.2.1 (API)
   │
Phase 6 ─────────────────────────────────────────────────────────────
   │
   ├─ Task 6.1.1 (레이아웃) ──┬── Task 6.2.1 (검색 페이지)
   │                         │
   ├─ Task 6.1.2 (공통 UI) ──┼── Task 6.2.2 (챗봇 페이지)
   │                         │
   │                         └── Task 6.2.3 (논문 상세)
   │
   └─ Task 6.3.1 (인증 서비스) ── Task 6.3.2 (인증 UI)
   │
Phase 7 ─────────────────────────────────────────────────────────────
   │
   ├─ Task 7.1.1 (Unit 테스트) ──┬── Task 7.2.1 (CI/CD)
   │                            │
   ├─ Task 7.1.2 (Integration) ─┴── Task 7.2.2 (프로덕션)
   │
   └─ Task 7.1.3 (Frontend 테스트) ── Task 7.2.3 (성능 테스트)
```

---

## Checkpoint Validation (체크포인트 검증)

### Week 1 완료 기준
- [ ] 프로젝트 구조 완성
- [ ] docker-compose up 동작
- [ ] Backend /health 응답
- [ ] Frontend 개발 서버 실행

### Week 2 완료 기준
- [ ] PubMed 검색 동작
- [ ] 논문 데이터 DB 저장
- [ ] Celery 크롤링 태스크 실행

### Week 3 완료 기준
- [ ] 텍스트 → 768차원 임베딩 변환
- [ ] Qdrant 저장/검색 동작
- [ ] 논문 저장 시 자동 임베딩

### Week 4 완료 기준
- [ ] RAG 질의응답 동작
- [ ] 출처 포함 응답
- [ ] 5초 이내 응답

### Week 5 완료 기준
- [ ] 의미 기반 검색 API 동작
- [ ] 유사 논문 추천 동작
- [ ] 검색 1초 이내 응답

### Week 6 완료 기준
- [ ] 검색/챗봇/논문상세 페이지 동작
- [ ] 로그인/회원가입 동작
- [ ] 전체 사용자 플로우 완성

### Week 7 완료 기준
- [ ] 테스트 커버리지 80%+
- [ ] CI/CD 파이프라인 동작
- [ ] 프로덕션 배포 완료

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 1.0 | 2024.12 | 초기 Tasks 작성 |
